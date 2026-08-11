"""ROCKET (random convolutional kernel transform) feature extraction and classification."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

from rocket2d.config import default_device

KernelGroup = dict[tuple[int, int], list[dict[str, object]]]
KernelTensors = dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]

DEFAULT_FEATURE_TYPES = ["ppv", "max", "mpv", "mipv_y", "mipv_x"]
DEFAULT_K_CHOICES = [3, 5, 7, 9]
DEFAULT_D_CHOICES = [1, 2]
_MASK_FEATURES = ("ppv", "mpv", "mipv_y", "mipv_x", "lspv")
_POS_COUNT_FEATURES = ("mpv", "mipv_y", "mipv_x")


class RocketClassifier:
    """ROCKET feature extraction + RidgeClassifier for 2D image classification.

    Randomly generates convolution kernels, applies them to extract pooled
    features (proportion of positive values, max, mean of positive values,
    mean position of positive values), and fits a ``RidgeClassifier`` on top.
    """

    def __init__(
        self,
        n_kernels: int = 5000,
        k_choices: list[int] | None = None,
        d_choices: list[int] | None = None,
        feature_types: list[str] | None = None,
        alpha: float = 1200,
        seed: int = 42,
        batch_size: int = 128,
        device: str | None = None,
        lspv_grid: int = 3,
    ) -> None:
        """Initialize a RocketClassifier instance.

        Parameters
        ----------
        n_kernels : int
            Number of random convolution kernels.
        k_choices : list of int, optional
            Kernel sizes to sample from (default ``[3, 5, 7, 9]``).
        d_choices : list of int, optional
            Dilation factors to sample from (default ``[1, 2]``).
        feature_types : list of str, optional
            Feature types to compute (default ``["ppv", "max", "mpv", "mipv_y", "mipv_x"]``).
        alpha : float
            Ridge regularization strength.
        seed : int
            Random seed for kernel generation.
        batch_size : int
            Batch size used when computing features.
        device : str, optional
            Torch device for feature extraction (default: CUDA if available, else CPU).
        lspv_grid : int
            Grid resolution for the Local Share of Positive Values (LSPV) feature,
            if ``"lspv"`` is included in ``feature_types``: the kernel's activation
            map is adaptively pooled into an ``lspv_grid x lspv_grid`` grid of
            regional PPV values, giving each kernel coarse positional resolution
            instead of a single global statistic (default 3, i.e. 9 regions).
        """
        self.n_kernels = n_kernels
        self.k_choices = k_choices or list(DEFAULT_K_CHOICES)
        self.d_choices = d_choices or list(DEFAULT_D_CHOICES)
        self.feature_types = feature_types or list(DEFAULT_FEATURE_TYPES)
        self.alpha = alpha
        self.seed = seed
        self.batch_size = batch_size
        self.device = device or default_device()
        self.lspv_grid = lspv_grid
        self.kernels_by_group: KernelGroup | None = None
        self.clf = RidgeClassifier(alpha=self.alpha)
        self.is_fitted = False
        # Lazily built, keyed by id(kernels_by_group): stacking numpy kernel
        # weights into torch tensors is redundant work if repeated per batch.
        self._kernel_tensor_cache: dict[int, KernelTensors] = {}
        # Position-index tensors for mipv_x/mipv_y depend only on the conv
        # output's spatial size, which is constant across batches of a run.
        self._pos_idx_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def make_random_kernels_2d(
        n_kernels: int,
        in_channels: int,
        k_choices: list[int],
        d_choices: list[int],
        seed: int = 42,
    ) -> KernelGroup:
        """Generate random 2D convolution kernels and biases, grouped by (size, dilation).

        Parameters
        ----------
        n_kernels : int
            Number of kernels to generate.
        in_channels : int
            Number of input channels.
        k_choices : list of int
            Kernel sizes to sample from.
        d_choices : list of int
            Dilation factors to sample from.
        seed : int
            Random seed.

        Returns
        -------
        dict
            Mapping ``(kernel_size, dilation) -> [{"weight": ..., "bias": ...}, ...]``.
        """
        rng = np.random.RandomState(seed)
        kernels_by_group: KernelGroup = {}
        for _ in range(n_kernels):
            k = int(rng.choice(k_choices))
            d = int(rng.choice(d_choices))
            key = (k, d)
            fan_in = in_channels * (k * k)
            std = np.sqrt(2.0 / max(1, fan_in))
            weights = rng.normal(0.0, std, size=(in_channels, k, k)).astype(np.float32)
            bias = float(rng.normal(0, 0.1))
            kernels_by_group.setdefault(key, []).append({"weight": weights, "bias": bias})
        return kernels_by_group

    def _get_kernel_tensors(self, kernels_by_group: KernelGroup) -> KernelTensors:
        """Return cached per-group (weight, bias) tensors, building them once on first use.

        Stacking the numpy kernel weights into a torch tensor is the same work
        every single batch if done inside the hot loop; since ``kernels_by_group``
        does not change across batches (or across ``fit`` -> ``predict``), we
        build the stacked tensors exactly once per kernel set and reuse them.
        """
        cache_key = id(kernels_by_group)
        cached = self._kernel_tensor_cache.get(cache_key)
        if cached is not None:
            return cached
        tensors: KernelTensors = {}
        for key, kernels in kernels_by_group.items():
            weight = torch.stack([torch.from_numpy(kern["weight"]) for kern in kernels])
            bias = torch.tensor([kern["bias"] for kern in kernels], dtype=torch.float32)
            tensors[key] = (weight.to(self.device), bias.to(self.device))
        self._kernel_tensor_cache[cache_key] = tensors
        return tensors

    def _get_pos_indices(self, out_h: int, out_w: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached ``(h_idx, w_idx)`` broadcast tensors for a given output spatial size."""
        cache_key = (out_h, out_w)
        cached = self._pos_idx_cache.get(cache_key)
        if cached is not None:
            return cached
        h_idx = torch.arange(out_h, device=self.device).reshape(1, 1, -1, 1)
        w_idx = torch.arange(out_w, device=self.device).reshape(1, 1, 1, -1)
        self._pos_idx_cache[cache_key] = (h_idx, w_idx)
        return h_idx, w_idx

    def compute_features_2d(
        self, X: np.ndarray, kernels_by_group: KernelGroup, feature_types: list[str]
    ) -> np.ndarray:
        """Apply grouped random kernels to a batch of images and pool features.

        Parameters
        ----------
        X : np.ndarray
            Input images, shape ``(N, C, H, W)``.
        kernels_by_group : dict
            Kernels grouped by ``(kernel_size, dilation)``, from :meth:`make_random_kernels_2d`.
        feature_types : list of str
            Which pooled features to compute per kernel.

        Returns
        -------
        np.ndarray
            Feature matrix, shape ``(N, n_features)``.
        """
        needs_mask = any(ft in feature_types for ft in _MASK_FEATURES)
        needs_pos_counts = any(ft in feature_types for ft in _POS_COUNT_FEATURES)
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).to(self.device, dtype=torch.float32)
            kernel_tensors = self._get_kernel_tensors(kernels_by_group)
            group_features = []
            for (_, d), (weight, bias) in kernel_tensors.items():
                out = F.conv2d(X_tensor, weight, bias=bias, stride=1, padding=0, dilation=d)

                mask = (out > 0).float() if needs_mask else None
                pos_counts = (
                    mask.sum(dim=[2, 3]).clamp(min=1.0)
                    if mask is not None and needs_pos_counts
                    else None
                )

                feats = []
                if "ppv" in feature_types:
                    assert mask is not None
                    feats.append(mask.mean(dim=[2, 3]))
                if "max" in feature_types:
                    feats.append(out.amax(dim=[2, 3]))
                if "mpv" in feature_types:
                    assert mask is not None and pos_counts is not None
                    feats.append((out * mask).sum(dim=[2, 3]) / pos_counts)
                if "mipv_y" in feature_types or "mipv_x" in feature_types:
                    assert mask is not None and pos_counts is not None
                    h_idx, w_idx = self._get_pos_indices(out.shape[2], out.shape[3])
                    mipv_y = (h_idx * mask).sum(dim=[2, 3]) / pos_counts
                    mipv_x = (w_idx * mask).sum(dim=[2, 3]) / pos_counts
                    h, w = out.shape[2], out.shape[3]
                    if h > 1:
                        mipv_y = mipv_y / (h - 1)
                    if w > 1:
                        mipv_x = mipv_x / (w - 1)
                    if "mipv_y" in feature_types:
                        feats.append(mipv_y)
                    if "mipv_x" in feature_types:
                        feats.append(mipv_x)
                if "lspv" in feature_types:
                    assert mask is not None
                    grid = min(self.lspv_grid, out.shape[2], out.shape[3])
                    lspv = F.adaptive_avg_pool2d(mask, output_size=(grid, grid))
                    feats.append(lspv.flatten(start_dim=1))
                group_features.append(torch.cat(feats, dim=1))
            return torch.cat(group_features, dim=1).cpu().numpy()

    def compute_features_in_batches(
        self,
        X: np.ndarray,
        kernels_by_group: KernelGroup | None = None,
        feature_types: list[str] | None = None,
    ) -> np.ndarray:
        """Compute ROCKET features for all rows of ``X`` in batches of ``self.batch_size``.

        Parameters
        ----------
        X : np.ndarray
            Input images, shape ``(N, C, H, W)``.
        kernels_by_group : dict, optional
            Defaults to the classifier's fitted kernels.
        feature_types : list of str, optional
            Defaults to the classifier's configured feature types.

        Returns
        -------
        np.ndarray
            Feature matrix, shape ``(N, n_features)``.
        """
        kernels_by_group = kernels_by_group or self.kernels_by_group
        feature_types = feature_types or self.feature_types
        if kernels_by_group is None:
            raise RuntimeError("No kernels available. Call fit() first.")
        n = X.shape[0]
        feats = [
            self.compute_features_2d(X[i : i + self.batch_size], kernels_by_group, feature_types)
            for i in range(0, n, self.batch_size)
        ]
        return np.concatenate(feats, axis=0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RocketClassifier:
        """Generate random kernels, extract features, and fit the ridge classifier.

        Parameters
        ----------
        X : np.ndarray
            Training images, shape ``(N, H, W)`` or ``(N, C, H, W)``.
        y : np.ndarray
            Training labels.

        Returns
        -------
        RocketClassifier
            self

        Raises
        ------
        ValueError
            If the image height/width is smaller than the largest configured
            kernel/dilation combination can be applied to without padding.
        """
        if X.ndim == 3:
            X = X[:, None, :, :]

        max_span = max((k - 1) * d + 1 for k in self.k_choices for d in self.d_choices)
        min_side = min(X.shape[2], X.shape[3])
        if max_span > min_side:
            raise ValueError(
                f"Image size ({X.shape[2]}x{X.shape[3]}) is too small for the configured "
                f"kernel/dilation choices: the largest combination (k={self.k_choices}, "
                f"d={self.d_choices}) needs at least {max_span}px per side. Increase the "
                f"image size or reduce k_choices/d_choices."
            )

        self.kernels_by_group = self.make_random_kernels_2d(
            self.n_kernels, X.shape[1], self.k_choices, self.d_choices, seed=self.seed
        )
        features = self.compute_features_in_batches(X, self.kernels_by_group, self.feature_types)
        self.clf.fit(features, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for unseen images.

        Parameters
        ----------
        X : np.ndarray
            Images, shape ``(N, H, W)`` or ``(N, C, H, W)``.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit the model first!")
        if X.ndim == 3:
            X = X[:, None, :, :]
        features = self.compute_features_in_batches(X)
        return self.clf.predict(features)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy on ``(X, y)``."""
        return accuracy_score(y, self.predict(X))
