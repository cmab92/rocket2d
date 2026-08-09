"""A simple CNN baseline for grayscale image classification."""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """A small convolutional network: two conv blocks followed by two FC layers."""

    def __init__(self, num_classes: int, img_size: int) -> None:
        """Initialize the model.

        Parameters
        ----------
        num_classes : int
            Number of classification output classes.
        img_size : int
            Height/width of the (square) input images.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (img_size // 4) * (img_size // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of images ``(B, 1, img_size, img_size)``."""
        x = self.features(x)
        return self.classifier(x)
