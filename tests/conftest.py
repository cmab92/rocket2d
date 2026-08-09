"""Shared pytest fixtures: synthetic image datasets on disk."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless test environments have no display

import numpy as np
import pytest
from PIL import Image


def _write_random_images(class_dir: Path, n: int, size: tuple[int, int] = (32, 32)) -> None:
    class_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 256, size=(size[1], size[0]), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(class_dir / f"img_{i}.png")


@pytest.fixture
def neu_dataset_dir(tmp_path: Path) -> Path:
    """Build a synthetic NEU-style dataset: root/{train,validation}/{class}/*.png."""
    root = tmp_path / "neu"
    for split in ("train", "validation"):
        for cls in ("crazing", "scratches"):
            _write_random_images(root / split / cls, n=6)
    return root


@pytest.fixture
def xray_dataset_dir(tmp_path: Path) -> Path:
    """Build a synthetic chest X-ray dataset: root/{train,val,test}/{NORMAL,PNEUMONIA}/*.png."""
    root = tmp_path / "xray"
    for split in ("train", "val", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            _write_random_images(root / split / cls, n=6)
    return root


@pytest.fixture
def dtd_dataset_dir(tmp_path: Path) -> Path:
    """Build a synthetic DTD-style dataset: root/images/{class}/*.png."""
    root = tmp_path / "dtd"
    for cls in ("bumpy", "striped"):
        _write_random_images(root / "images" / cls, n=6)
    return root
