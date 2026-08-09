"""Plotting and metrics reporting for dataset inspection and model evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from rocket2d.config import IMG_EXTENSIONS


def show_samples(
    dataset_path: str | Path,
    n_samples: int = 6,
    grayscale: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """Visualize random sample images from a dataset directory.

    Parameters
    ----------
    dataset_path : str or Path
        Root directory to search for images (recursively).
    n_samples : int, optional
        Number of samples to show (default 6).
    grayscale : bool, optional
        Whether to convert images to grayscale for display (default True).
    show : bool, optional
        Whether to call ``plt.show()`` (default True). Set False for headless runs.
    save_path : str or Path, optional
        If given, save the figure to this path.
    """
    images = [
        str(p)
        for p in Path(dataset_path).rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    ]
    if not images:
        print(f"No images found in: {dataset_path}")
        return

    sample_paths = np.random.choice(images, min(n_samples, len(images)), replace=False)

    plt.figure(figsize=(12, 4))
    for i, path in enumerate(sample_paths):
        img: Image.Image = Image.open(path)
        if grayscale:
            img = img.convert("L")
        plt.subplot(1, len(sample_paths), i + 1)
        plt.imshow(img, cmap="gray" if grayscale else None)
        plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close()


def image_size_stats(image_paths: Sequence[str], n_samples: int = 200) -> None:
    """Print average width/height over a sample of images.

    Parameters
    ----------
    image_paths : sequence of str
        Image paths to sample from.
    n_samples : int, optional
        Maximum number of images to sample (default 200).
    """
    if not image_paths:
        print("No images found.")
        return
    sample = np.random.choice(image_paths, min(n_samples, len(image_paths)), replace=False)
    sizes = [Image.open(p).size for p in sample]
    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    print(f"Average width: {np.mean(widths):.1f}")
    print(f"Average height: {np.mean(heights):.1f}")


def grayscale_histogram(
    image_paths: Sequence[str],
    n_samples: int = 5,
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """Plot grayscale pixel-intensity histograms for a sample of images.

    Parameters
    ----------
    image_paths : sequence of str
        Image paths to sample from.
    n_samples : int, optional
        Maximum number of images to display a histogram for (default 5).
    show : bool, optional
        Whether to call ``plt.show()`` (default True).
    save_path : str or Path, optional
        If given, save the figure to this path.
    """
    if not image_paths:
        print("No images found.")
        return
    sample = np.random.choice(image_paths, min(n_samples, len(image_paths)), replace=False)
    plt.figure(figsize=(12, 3))
    for i, p in enumerate(sample):
        img = Image.open(p).convert("L")
        plt.subplot(1, len(sample), i + 1)
        plt.hist(np.array(img).flatten(), bins=50)
        plt.title("Histogram")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close()


def evaluate_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: list[int] | None = None,
    target_names: list[str] | None = None,
    y_score: Sequence[float] | np.ndarray | None = None,
    prefix: str = "",
    show_confusion: bool = True,
    show_misclassified: bool = True,
    X_sample: np.ndarray | None = None,
    max_misclassified: int = 5,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """Print and visualize classification metrics (accuracy, F1, confusion matrix, ROC/PR, errors).

    Parameters
    ----------
    y_true, y_pred : sequence of int
        True and predicted class labels.
    labels : list of int, optional
        Label indices to include in evaluation.
    target_names : list of str, optional
        Human-readable label names for the classification report.
    y_score : sequence of float, optional
        Continuous scores for the positive class, for binary ROC/PR curves.
    prefix : str, optional
        Prefix prepended to printed metric lines.
    show_confusion : bool, optional
        Whether to plot the confusion matrix (default True).
    show_misclassified : bool, optional
        Whether to plot misclassified examples; requires ``X_sample`` (default True).
    X_sample : np.ndarray, optional
        Input images aligned with ``y_true``/``y_pred``, shape ``(N, H, W)`` or ``(N, 1, H, W)``.
    max_misclassified : int, optional
        Number of misclassified samples to display (default 5).
    save_dir : str or Path, optional
        Directory to save generated plots.
    show : bool, optional
        Whether to call ``plt.show()`` for generated figures (default True).

    Returns
    -------
    dict
        Scalar metrics: ``accuracy``, ``macro_f1``, ``weighted_f1``, ``report_dict``,
        and (if ``y_score`` given on a binary problem) ``roc_auc``.
    """
    if target_names is None or labels is None or len(target_names) != len(labels):
        print("WARNING: Cannot automatically assign class labels. Using default names.")
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))
        target_names = [str(i) for i in labels]

    acc = accuracy_score(y_true, y_pred)
    clf_rep = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, output_dict=True
    )
    print(f"{prefix}Accuracy: {acc:.4f}")
    print(f"{prefix}Macro F1: {clf_rep['macro avg']['f1-score']:.4f}")
    print(f"{prefix}Weighted F1: {clf_rep['weighted avg']['f1-score']:.4f}")
    print(
        f"{prefix}Classification Report:\n"
        f"{classification_report(y_true, y_pred, labels=labels, target_names=target_names)}"
    )
    cm_array = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"{prefix}Confusion matrix (as array):\n", cm_array)

    metrics: dict[str, Any] = {
        "accuracy": acc,
        "macro_f1": clf_rep["macro avg"]["f1-score"],
        "weighted_f1": clf_rep["weighted avg"]["f1-score"],
        "report_dict": clf_rep,
    }

    if show_confusion:
        fig, ax = plt.subplots(figsize=(6, 6))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/confusion_matrix.png", bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    if y_score is not None and len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        if save_dir:
            plt.savefig(f"{save_dir}/roc_curve.png", bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        plt.close()
        metrics["roc_auc"] = auc

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        if save_dir:
            plt.savefig(f"{save_dir}/pr_curve.png", bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        plt.close()

    if show_misclassified and X_sample is not None:
        y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
        mis_idx = np.where(y_true_arr != y_pred_arr)[0]
        if len(mis_idx) > 0:
            print(f"\n{prefix}Showing up to {max_misclassified} misclassified examples:")
            n_show = min(len(mis_idx), max_misclassified)
            plt.figure(figsize=(min(12, max_misclassified * 2.5), 2.5))
            for idx, mi in enumerate(mis_idx[:max_misclassified]):
                plt.subplot(1, n_show, idx + 1)
                img = X_sample[mi]
                if img.ndim == 3 and img.shape[0] == 1:
                    img = img[0]
                plt.imshow(img, cmap="gray")
                plt.title(f"True: {y_true_arr[mi]}\nPred: {y_pred_arr[mi]}")
                plt.axis("off")
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/misclassified.png", bbox_inches="tight", dpi=150)
            if show:
                plt.show()
            plt.close()
        else:
            print(f"{prefix}No misclassified samples to display.")

    return metrics


def plot_learning_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
    y_label: str = "Loss",
    title: str = "Learning Curve",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot training (and optionally validation) loss curves over epochs.

    Parameters
    ----------
    train_losses : sequence of float
        Training loss (or score) per epoch.
    val_losses : sequence of float, optional
        Validation loss (or score) per epoch.
    y_label : str, optional
        Y-axis label (default "Loss").
    title : str, optional
        Plot title.
    save_path : str or Path, optional
        If given, save the figure to this path.
    show : bool, optional
        Whether to call ``plt.show()`` (default True).
    """
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close()
