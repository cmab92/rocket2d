"""A simple 1D CNN baseline for real-valued time-series classification,
mirroring :class:`rocket2d.models.cnn.SimpleCNN`'s shallow, two-layer design."""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNN1D(nn.Module):
    """A small 1D convolutional network: two conv blocks followed by two FC layers."""

    def __init__(self, num_classes: int, seq_len: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (seq_len // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of series ``(B, 1, seq_len)``."""
        x = self.features(x)
        return self.classifier(x)
