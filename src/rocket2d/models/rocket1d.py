"""Genuine 1D-ROCKET (random convolutional kernel transform) for real-valued
time series, following Dempster et al. (2020) closely: kernel lengths sampled
from a small set, a per-kernel-group random choice between 'same' and 'valid'
padding, dilation, PPV+Max pooling, and a RidgeClassifier on top.

This mirrors the "minimal 1D-ROCKET baseline" described in Sec. 5.1 of the
paper (kernel sizes {7,9,11}, same/valid padding chosen per kernel group) but
is a fresh implementation for genuine 1D time-series data (SAR raw echo
lines), not the flattened-2D-image variant used there. Kernels are grouped by
(length, dilation, padding) and each group's convolution is run as a single
batched ``conv1d`` call, mirroring the grouping strategy in
:mod:`rocket2d.models.rocket` for tractable runtime at thousands of kernels.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

from rocket2d.config import default_device

DEFAULT_K_CHOICES = [7, 9, 11]
DEFAULT_D_CHOICES = [1, 2, 4]

GroupKey = tuple[int, int, str]  # (kernel_length, dilation, padding)


class Rocket1DClassifier:
    """1D-ROCKET feature extraction + RidgeClassifier for time-series classification."""

    def __init__(
        self,
        n_kernels: int = 5000,
        k_choices: list[int] | None = None,
        d_choices: list[int] | None = None,
        alpha: float = 1200,
        seed: int = 42,
        batch_size: int = 128,
        device: str | None = None,
    ) -> None:
        self.n_kernels = n_kernels
        self.k_choices = k_choices or list(DEFAULT_K_CHOICES)
        self.d_choices = d_choices or list(DEFAULT_D_CHOICES)
        self.alpha = alpha
        self.seed = seed
        self.batch_size = batch_size
        self.device = device or default_device()
        self.clf = RidgeClassifier(alpha=self.alpha)
        self.is_fitted = False
        self._groups: dict[GroupKey, tuple[torch.Tensor, torch.Tensor]] | None = None

    def _make_kernel_groups(
        self, in_channels: int
    ) -> dict[GroupKey, tuple[torch.Tensor, torch.Tensor]]:
        rng = np.random.RandomState(self.seed)
        raw: dict[GroupKey, list[tuple[np.ndarray, float]]] = {}
        for _ in range(self.n_kernels):
            k = int(rng.choice(self.k_choices))
            d = int(rng.choice(self.d_choices))
            padding = "same" if rng.random() < 0.5 else "valid"
            fan_in = in_channels * k
            std = np.sqrt(2.0 / max(1, fan_in))
            weight = rng.normal(0.0, std, size=(in_channels, k)).astype(np.float32)
            bias = float(rng.normal(0, 0.1))
            raw.setdefault((k, d, padding), []).append((weight, bias))

        groups: dict[GroupKey, tuple[torch.Tensor, torch.Tensor]] = {}
        for key, kernels in raw.items():
            weight_t = torch.stack([torch.from_numpy(w) for w, _ in kernels])
            bias_t = torch.tensor([b for _, b in kernels], dtype=torch.float32)
            groups[key] = (weight_t.to(self.device), bias_t.to(self.device))
        return groups

    def _compute_features(self, X: np.ndarray) -> np.ndarray:
        """X: (N, L) or (N, C, L) real-valued series -> (N, 2 * n_kernels) features."""
        if X.ndim == 2:
            X = X[:, None, :]
        assert self._groups is not None
        n = X.shape[0]
        feats_batches = []
        with torch.no_grad():
            for i in range(0, n, self.batch_size):
                batch_np = X[i : i + self.batch_size]
                batch = torch.from_numpy(batch_np).to(self.device, dtype=torch.float32)
                group_feats = []
                for (k, d, padding), (weight, bias) in self._groups.items():
                    span = (k - 1) * d + 1
                    if batch.shape[-1] < span:
                        n_kern = weight.shape[0]
                        group_feats.append(
                            torch.zeros((batch.shape[0], 2 * n_kern), device=self.device)
                        )
                        continue
                    pad = ((k - 1) * d) // 2 if padding == "same" else 0
                    out = F.conv1d(batch, weight, bias=bias, dilation=d, padding=pad)
                    ppv = (out > 0).float().mean(dim=2)
                    mx = out.amax(dim=2)
                    group_feats.append(torch.stack([ppv, mx], dim=2).flatten(start_dim=1))
                feats_batches.append(torch.cat(group_feats, dim=1).cpu().numpy())
        return np.concatenate(feats_batches, axis=0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Rocket1DClassifier:
        in_channels = 1 if X.ndim == 2 else X.shape[1]
        self._groups = self._make_kernel_groups(in_channels)
        features = self._compute_features(X)
        self.clf.fit(features, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Must fit the model first!")
        features = self._compute_features(X)
        return self.clf.predict(features)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))
