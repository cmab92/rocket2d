"""Folder-structured image dataset loading for NEU, chest X-ray, and DTD."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from rocket2d.config import IMG_EXTENSIONS


def get_images(root: str | Path) -> tuple[list[str], list[str]]:
    """Collect all image file paths and their associated class labels.

    Parameters
    ----------
    root : str or Path
        Root directory containing class subfolders.

    Returns
    -------
    tuple
        ``(image_paths, labels)`` — parallel lists of file paths and class names.
    """
    root = Path(root)
    paths: list[str] = []
    labels: list[str] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for file_path in class_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in IMG_EXTENSIONS:
                paths.append(str(file_path))
                labels.append(class_dir.name)
    return paths, labels


class ImageDataset:
    """Unified dataset loader for image classification tasks.

    Loads folder-structured datasets and converts them to NumPy arrays and
    PyTorch tensors, applying consistent preprocessing and class encoding.

    Supported variants: ``"neu"``, ``"xray"``, ``"dtd"``.
    """

    def __init__(
        self,
        root: str | Path,
        size: int = 128,
        grayscale: bool = True,
        dataset: str = "",
    ) -> None:
        """Initialize the dataset loader.

        Parameters
        ----------
        root : str or Path
            Directory containing the dataset.
        size : int, optional
            Target image size (default 128).
        grayscale : bool, optional
            If True, converts images to grayscale (default True).
        dataset : str
            Dataset name (``"neu"``, ``"xray"``, or ``"dtd"``).
        """
        self.root = Path(root)
        self.size = size
        self.grayscale = grayscale
        self.dataset = dataset
        self.class_to_idx: dict[str, int] = {}
        self.samples = self._load_paths()
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def _load_paths(self) -> list[tuple[str, int, str]]:
        """Build the list of ``(image_path, class_index, split)`` samples."""
        loaders = {
            "neu": self._load_neu,
            "xray": self._load_xray,
            "dtd": self._load_dtd,
        }
        try:
            loader = loaders[self.dataset]
        except KeyError:
            raise ValueError(f"Unknown dataset: {self.dataset}") from None
        return loader()

    def _load_split_classes(
        self, splits: list[str], class_to_idx: dict[str, int] | None = None
    ) -> list[tuple[str, int, str]]:
        """Walk ``root/<split>/<class>/*`` for each split and collect labeled samples."""
        samples: list[tuple[str, int, str]] = []
        for split in splits:
            split_path = self.root / split
            if not split_path.exists():
                continue
            for class_dir in sorted(p for p in split_path.iterdir() if p.is_dir()):
                if class_to_idx is not None and class_dir.name not in class_to_idx:
                    continue
                label = self.class_to_idx[class_dir.name]
                for file_path in class_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in IMG_EXTENSIONS:
                        samples.append((str(file_path), label, split))
        return samples

    def _load_neu(self) -> list[tuple[str, int, str]]:
        """Load NEU surface defect dataset image paths, labels, and splits."""
        train_path = self.root / "train"
        classes = sorted(p.name for p in train_path.iterdir() if p.is_dir())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return self._load_split_classes(["train", "validation"])

    def _load_xray(self) -> list[tuple[str, int, str]]:
        """Load Chest X-Ray Pneumonia dataset image paths, labels, and splits."""
        self.class_to_idx = {"NORMAL": 0, "PNEUMONIA": 1}
        return self._load_split_classes(["train", "val", "test"], self.class_to_idx)

    def _load_dtd(self) -> list[tuple[str, int, str]]:
        """Load Describable Textures Dataset (DTD) image paths and labels."""
        images_path = self.root / "images"
        classes = sorted(p.name for p in images_path.iterdir() if p.is_dir())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        samples: list[tuple[str, int, str]] = []
        for class_dir in sorted(p for p in images_path.iterdir() if p.is_dir()):
            label = self.class_to_idx[class_dir.name]
            for file_path in class_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in IMG_EXTENSIONS:
                    samples.append((str(file_path), label, "all"))
        return samples

    def _preprocess_image(self, path: str) -> np.ndarray:
        """Load, resize, and normalize a single image to ``[0, 1]`` float32."""
        with Image.open(path) as raw:
            img: Image.Image = raw.convert("L") if self.grayscale else raw.convert("RGB")
        img = img.resize((self.size, self.size))
        return np.asarray(img, dtype=np.float32) / 255.0

    def _load_all(self) -> tuple[np.ndarray, np.ndarray]:
        """Load and preprocess all samples into ``self.X`` / ``self.y``."""
        X_list = [self._preprocess_image(path) for path, _, _ in self.samples]
        y_list = [label for _, label, _ in self.samples]
        self.X = np.stack(X_list).astype(np.float32)
        self.y = np.array(y_list, dtype=np.int64)
        return self.X, self.y

    def numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Get data as NumPy arrays: images ``(N, H, W)`` or ``(N, H, W, C)``, labels ``(N,)``."""
        return self.prepare()

    def torch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get data as PyTorch tensors with channel-first layout ``(N, C, H, W)``."""
        X, y = self.prepare()
        X = X[:, None, :, :] if self.grayscale else X.transpose(0, 3, 1, 2)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def prepare(self) -> tuple[np.ndarray, np.ndarray]:
        """Ensure images and labels are loaded, loading them lazily on first call."""
        if self.X is None or self.y is None:
            self._load_all()
        assert self.X is not None and self.y is not None
        return self.X, self.y

    def stats(self) -> dict[str, Any]:
        """Compute basic dataset statistics (counts, class/split distributions)."""
        labels = [s[1] for s in self.samples]
        splits = [s[2] for s in self.samples]
        class_dist = {k: labels.count(k) for k in set(labels)}
        split_dist = {k: splits.count(k) for k in set(splits)}
        return {
            "num_samples": len(self.samples),
            "num_classes": len(self.class_to_idx),
            "class_distribution": class_dist,
            "split_distribution": split_dist,
            "sample_shape": (self.size, self.size),
            "dtype": "float32",
            "value_range": (0.0, 1.0),
        }

    def debug(self, max_samples: int = 5) -> None:
        """Print debug information: distributions, sample entries, and path validity."""
        samples = self.samples
        print("\n" + "=" * 50)
        print(f"DATASET DEBUG: {self.dataset}")
        print("=" * 50)
        print(f"Total samples: {len(samples)}")

        labels = [s[1] for s in samples]
        print("\nLabel distribution:")
        for k in sorted(set(labels)):
            print(k, labels.count(k))

        splits = [s[2] for s in samples]
        print("\nSplit distribution:")
        for split_name in sorted(set(splits)):
            print(split_name, splits.count(split_name))

        print("\nSample entries:")
        for s in samples[:max_samples]:
            print(s)

        print("\nPath validity:")
        for s in samples[:max_samples]:
            print(os.path.exists(s[0]), s[0])

        img = self._preprocess_image(samples[0][0])
        print("\nSingle image shape:", img.shape)
        print("dtype:", img.dtype)
        print("min/max:", img.min(), img.max())

    def validate(self, min_samples: int = 10) -> bool:
        """Run basic sanity checks on the dataset (sample count, class count, value range).

        Parameters
        ----------
        min_samples : int, optional
            Minimum required sample count (default 10).

        Returns
        -------
        bool
            True if the dataset passes all checks.
        """
        stats = self.stats()
        print("\n=== DATASET VALIDATION ===")
        print("Samples:", stats["num_samples"])
        print("Classes:", stats["num_classes"])
        print("Shape:", stats["sample_shape"])
        print("Range:", stats["value_range"])

        if stats["num_samples"] < min_samples:
            print("Too few samples.")
            return False
        if len(stats["class_distribution"]) < 2:
            print("Only one class detected.")
            return False
        vmin, vmax = stats["value_range"]
        if vmax <= 0 or vmin < 0:
            print("Image normalization issue.")
            return False

        print("Dataset valid.")
        return True
