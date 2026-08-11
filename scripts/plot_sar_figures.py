"""Generate figures for the SAR RFI detection section: an example clean-vs-
contaminated echo line, and confusion matrices for the three methods on the
representative seed-42 split. Not part of the installed package.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from scripts.run_sar_rfi_experiment import BLOCK_SIZE, load_burst, valid_realizations


def plot_example_signal(out_path: str) -> None:
    echo, rfi = load_burst()
    valid_k = valid_realizations(rfi)
    # Find a line with strong, easy-to-see interference for illustration.
    best = max(
        ((i, k) for i in range(echo.shape[0]) for k in valid_k if np.count_nonzero(rfi[k, i, :]) > 0),
        key=lambda ik: np.abs(rfi[ik[1], ik[0], :]).max(),
    )
    i, k = best
    clean_mag = np.abs(echo[i])
    contaminated_mag = np.abs(echo[i] + rfi[k, i, :])

    fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True, sharey=True)
    axes[0].plot(clean_mag, linewidth=0.5, color="#4C72B0")
    axes[0].set_title("Clean echo line")
    axes[0].set_ylabel("Magnitude")
    axes[1].plot(contaminated_mag, linewidth=0.5, color="#C44E52")
    axes[1].set_title(f"Same line + injected RFI (realization {k})")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_xlabel("Range sample")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(out_path: str, seed: int = 42) -> None:
    r = json.loads(Path(f"results/sar_rfi/seed{seed}.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    labels = ["clean", "RFI"]
    for ax, model, title in zip(
        axes, ["cnn", "rocket", "svm"], ["1D CNN", "1D-ROCKET", "Linear SVM"]
    ):
        cm = np.array(r[model]["confusion_matrix"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_example_signal("docs/figures/sar_example_signal.png")
    plot_confusion_matrices("docs/figures/sar_confusion_matrices.png")
    print("Saved docs/figures/sar_example_signal.png and sar_confusion_matrices.png")
