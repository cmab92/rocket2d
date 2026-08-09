"""ROCKET (random convolutional kernel transform) feature extraction and classification."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

KernelGroup = dict[tuple[int, int], list[dict[str, object]]]

DEFAULT_FEATURE_TYPES = ["ppv", "max", "mpv", "mipv_y", "mipv_x"]
DEFAULT_K_CHOICES = [3, 5, 7, 9]
DEFAULT_D_CHOICES = [1, 2]


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
        """
        self.n_kernels = n_kernels
        self.k_choices = k_choices or list(DEFAULT_K_CHOICES)
        self.d_choices = d_choices or list(DEFAULT_D_CHOICES)
        self.feature_types = feature_types or list(DEFAULT_FEATURE_TYPES)
        self.alpha = alpha
        self.seed = seed
        self.batch_size = batch_size
        self.kernels_by_group: KernelGroup | None = None
        self.clf = RidgeClassifier(alpha=self.alpha)
        self.is_fitted = False

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

    @staticmethod
    def compute_features_2d(
        X: np.ndarray, kernels_by_group: KernelGroup, feature_types: list[str]
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
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).to(torch.float32)
            group_features = []
            for (_, d), kernels in kernels_by_group.items():
                weight = torch.stack([torch.from_numpy(kern["weight"]) for kern in kernels])
                bias = torch.tensor([kern["bias"] for kern in kernels], dtype=torch.float32)
                out = F.conv2d(X_tensor, weight, bias=bias, stride=1, padding=0, dilation=d)

                feats = []
                if "ppv" in feature_types:
                    feats.append((out > 0).float().mean(dim=[2, 3]))
                if "max" in feature_types:
                    feats.append(out.amax(dim=[2, 3]))
                if "mpv" in feature_types:
                    mask = (out > 0).float()
                    feats.append((out * mask).sum(dim=[2, 3]) / mask.sum(dim=[2, 3]).clamp(min=1.0))
                if "mipv_y" in feature_types or "mipv_x" in feature_types:
                    mask = (out > 0).float()
                    h_idx = torch.arange(out.shape[2]).reshape(1, 1, -1, 1)
                    w_idx = torch.arange(out.shape[3]).reshape(1, 1, 1, -1)
                    pos_counts = mask.sum(dim=[2, 3]).clamp(min=1.0)
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
