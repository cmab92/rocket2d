"""End-to-end training/evaluation loops for the CNN and ROCKET pipelines."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from rocket2d.config import set_seed
from rocket2d.data.datasets import ImageDataset
from rocket2d.models.cnn import SimpleCNN
from rocket2d.models.rocket import RocketClassifier
from rocket2d.visualization import evaluate_metrics, plot_learning_curves

logger = logging.getLogger(__name__)


def run_cnn(
    dataset_name: str,
    root: str,
    img_size: int = 128,
    batch_size: int = 32,
    epochs: int = 10,
    seed: int = 42,
    device: str = "cpu",
    show_plots: bool = True,
    save_dir: str | None = None,
) -> dict[str, float]:
    """Train and evaluate :class:`SimpleCNN` on a given image dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset key (``"neu"``, ``"xray"``, or ``"dtd"``).
    root : str
        Path to the dataset root directory.
    img_size : int, optional
        Image resize target (default 128).
    batch_size : int, optional
        Training batch size (default 32).
    epochs : int, optional
        Number of training epochs (default 10).
    seed : int, optional
        Random seed (default 42).
    device : str, optional
        Torch device string (default "cpu").
    show_plots : bool, optional
        Whether to display matplotlib figures interactively (default True).
    save_dir : str, optional
        If given, save generated plots (confusion matrix, ROC/PR, misclassified
        examples, learning curve) to this directory.

    Returns
    -------
    dict
        Evaluation metrics returned by :func:`evaluate_metrics`.
    """
    set_seed(seed)
    dataset = ImageDataset(root=root, dataset=dataset_name, size=img_size, grayscale=True)
    X, y = dataset.prepare()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    X_tr = X_tr[:, None, :, :].astype("float32")
    X_te = X_te[:, None, :, :].astype("float32")
    y_tr = y_tr.astype("int64")
    y_te = y_te.astype("int64")

    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    test_ds = torch.utils.data.TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
    loader_train = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    loader_test = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(set(y))
    model = SimpleCNN(num_classes=num_classes, img_size=img_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    model.train()
    for epoch in range(epochs):
        loss_sum = 0.0
        for xb, yb in loader_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        epoch_loss = loss_sum / len(loader_train)
        train_losses.append(epoch_loss)
        logger.info("Epoch %2d/%d | Loss: %.4f", epoch + 1, epochs, epoch_loss)

    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    X_te_collected = []
    with torch.no_grad():
        for xb, yb in loader_test:
            X_te_collected.append(xb.cpu().numpy())
            pred = torch.argmax(model(xb.to(device)), dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(yb.numpy())
    X_te_full = np.concatenate(X_te_collected)

    print(f"\n--- CNN Evaluation for {dataset_name.upper()} ---")
    if not dataset.class_to_idx or len(dataset.class_to_idx) != num_classes:
        print("WARNING: Cannot automatically assign class labels. Using default names.")
        target_names = [str(i) for i in range(num_classes)]
    else:
        target_names = list(dataset.class_to_idx.keys())

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    metrics = evaluate_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        labels=list(range(num_classes)),
        target_names=target_names,
        X_sample=X_te_full,
        show_confusion=True,
        show_misclassified=True,
        max_misclassified=5,
        show=show_plots,
        save_dir=save_dir,
    )
    plot_learning_curves(
        train_losses,
        y_label="Training Loss",
        title=f"{dataset_name.upper()} CNN Learning Curve",
        show=show_plots,
        save_path=f"{save_dir}/learning_curve.png" if save_dir else None,
    )
    print("-" * 60)
    return metrics


def run_rocket(
    dataset_name: str,
    root: str,
    img_size: int = 128,
    seed: int = 42,
    n_kernels: int = 5000,
    device: str | None = None,
    show_plots: bool = True,
    save_dir: str | None = None,
) -> dict[str, float]:
    """Train and evaluate a :class:`RocketClassifier` pipeline on a given dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset key (``"neu"``, ``"xray"``, or ``"dtd"``).
    root : str
        Path to the dataset root directory.
    img_size : int, optional
        Image resize target (default 128).
    seed : int, optional
        Random seed (default 42).
    n_kernels : int, optional
        Number of random ROCKET kernels (default 5000).
    device : str, optional
        Torch device for feature extraction (default: CUDA if available, else CPU).
    show_plots : bool, optional
        Whether to display matplotlib figures interactively (default True).
    save_dir : str, optional
        If given, save generated plots (confusion matrix, ROC/PR, misclassified
        examples) to this directory.

    Returns
    -------
    dict
        Evaluation metrics returned by :func:`evaluate_metrics`.
    """
    set_seed(seed)
    dataset = ImageDataset(root=root, dataset=dataset_name, size=img_size, grayscale=True)
    X, y = dataset.prepare()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    model = RocketClassifier(n_kernels=n_kernels, seed=seed, device=device)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n--- ROCKET Evaluation for {dataset_name.upper()} ---")
    metrics = evaluate_metrics(
        y_true=y_te,
        y_pred=y_pred,
        labels=list(range(len(set(y)))),
        target_names=list(dataset.class_to_idx.keys()),
        X_sample=X_te,
        show_confusion=True,
        show_misclassified=True,
        max_misclassified=5,
        show=show_plots,
        save_dir=save_dir,
    )
    print("-" * 60)
    return metrics
