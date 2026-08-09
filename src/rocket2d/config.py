"""Global configuration and reproducibility helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch

IMG_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")


def default_device() -> str:
    """Return "cuda" if a GPU is available, otherwise "cpu"."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    """Run configuration shared across data loading, training, and evaluation."""

    seed: int = 42
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 10
    device: str = field(default_factory=default_device)
    base_data_dir: str = "data"
    n_kernels: int = 5000

    @property
    def dataset_dirs(self) -> dict[str, str]:
        """Mapping of dataset name to its root directory under ``base_data_dir``."""
        return {
            "neu": f"{self.base_data_dir}/neu",
            "xray": f"{self.base_data_dir}/xray",
            "lc": f"{self.base_data_dir}/lc25000",
            "dtd": f"{self.base_data_dir}/dtd",
        }


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
