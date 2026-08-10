"""Regenerate compact, print-quality CNN-vs-ROCKET confusion-matrix figures
for NEU and Chest X-Ray, from the confusion matrices logged in
results/run_full_eval.log, for inclusion in docs/figures/.

Not part of the installed package; a one-off plotting helper.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

NEU_LABELS = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
XRAY_LABELS = ["NORMAL", "PNEUMONIA"]

NEU_CNN = np.array(
    [
        [60, 0, 0, 0, 0, 0],
        [0, 60, 0, 0, 0, 0],
        [2, 0, 54, 2, 0, 2],
        [1, 0, 0, 59, 0, 0],
        [0, 0, 0, 0, 59, 1],
        [0, 2, 0, 0, 2, 56],
    ]
)
NEU_ROCKET = np.array(
    [
        [60, 0, 0, 0, 0, 0],
        [0, 53, 0, 5, 2, 0],
        [0, 0, 60, 0, 0, 0],
        [1, 1, 0, 58, 0, 0],
        [2, 0, 0, 3, 55, 0],
        [0, 2, 0, 0, 0, 58],
    ]
)
XRAY_CNN = np.array([[288, 29], [23, 832]])
XRAY_ROCKET = np.array([[272, 45], [39, 816]])


def plot_pair(cm_cnn, cm_rocket, labels, out_path, rotate=False):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    for ax, cm, title in zip(axes, [cm_cnn, cm_rocket], ["Lightweight CNN", "2D-ROCKET"]):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title(title, fontsize=10)
        if rotate:
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(ax.get_xlabel(), fontsize=9)
        ax.set_ylabel(ax.get_ylabel(), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_pair(NEU_CNN, NEU_ROCKET, NEU_LABELS, "docs/figures/cm_neu.png", rotate=True)
    plot_pair(XRAY_CNN, XRAY_ROCKET, XRAY_LABELS, "docs/figures/cm_xray.png", rotate=False)
    print("Saved docs/figures/cm_neu.png and docs/figures/cm_xray.png")
