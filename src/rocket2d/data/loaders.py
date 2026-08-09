"""Train/test splitting and PyTorch DataLoader construction."""

from __future__ import annotations

import logging

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from rocket2d.data.datasets import ImageDataset

logger = logging.getLogger(__name__)


def prepare_image_dataset_for_training(
    dataset_name: str,
    dataset_paths: dict[str, str],
    image_size: int,
    seed: int,
    grayscale: bool = True,
    test_size: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load, validate, split, and tensorize an image dataset for training.

    Parameters
    ----------
    dataset_name : str
        The key for the dataset (``"neu"``, ``"xray"``, ``"dtd"``).
    dataset_paths : dict
        Mapping of dataset names to their root directories.
    image_size : int
        Height/width to which images are resized.
    seed : int
        Random seed for reproducibility.
    grayscale : bool, optional
        Whether to convert images to grayscale (default True).
    test_size : float, optional
        Proportion of the dataset assigned to the test split (default 0.2).

    Returns
    -------
    tuple of torch.Tensor
        ``(X_train, X_test, y_train, y_test)``, images shaped ``(N, C, H, W)``.
    """
    logger.info("Loading dataset '%s' from %s", dataset_name, dataset_paths[dataset_name])
    dataset = ImageDataset(
        root=dataset_paths[dataset_name],
        dataset=dataset_name,
        size=image_size,
        grayscale=grayscale,
    )
    dataset.debug()
    dataset.validate()

    X, y = dataset.prepare()
    logger.info("Dataset loaded: X=%s y=%s", X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    logger.info("Train split: %s, test split: %s", X_train.shape, X_test.shape)

    X_train = X_train[:, None, :, :]
    X_test = X_test[:, None, :, :]

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
    )


def create_data_loaders(
    X_train_tensor: torch.Tensor,
    y_train_tensor: torch.Tensor,
    X_test_tensor: torch.Tensor,
    y_test_tensor: torch.Tensor,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training and test datasets.

    Parameters
    ----------
    X_train_tensor, y_train_tensor : torch.Tensor
        Training images ``(N_train, C, H, W)`` and labels ``(N_train,)``.
    X_test_tensor, y_test_tensor : torch.Tensor
        Test images ``(N_test, C, H, W)`` and labels ``(N_test,)``.
    batch_size : int
        Batch size for both loaders.
    seed : int
        Random seed for the training shuffle.

    Returns
    -------
    tuple of DataLoader
        ``(train_loader, test_loader)``.
    """
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info("Train batches: %d, test batches: %d", len(train_loader), len(test_loader))
    return train_loader, test_loader
