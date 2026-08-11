"""SAR RFI detection as a binary time-series classification problem, using
raw Sentinel-1 Level-0 echo lines from the RFInject-v1-L0 dataset (ESA
Phi-lab, Hugging Face bucket ESA-philab/RFInject-v1-L0).

Each azimuth line of the burst is a genuine 1D radar time series. RFInject
stores the interference component separately from the clean echo (additive
model): a contaminated line is ``echo + rfi[k]`` for one of several injected
interference realizations, several of which are all-zero (no injection) and
are excluded. For each seed, every one of the burst's lines is paired with
one randomly chosen non-empty interference realization to build a balanced
clean/RFI dataset -- deliberately small (a single burst, ~100 lines), to
test whether 1D-ROCKET's sample efficiency (established on DTD in Sec. 5.4)
carries over to a genuinely small-n, real time-series regime, following
Dempster et al.'s original ROCKET design rather than the flattened-2D-image
variant used elsewhere in this paper.

Compares 1D-ROCKET, a shallow 1D CNN, and a linear SVM, over ten seeds with
paired significance testing, matching the protocol in Sec. 5.4/5.5.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch import nn

from rocket2d.config import set_seed
from rocket2d.models.cnn1d import SimpleCNN1D
from rocket2d.models.rocket1d import Rocket1DClassifier

BURST_PATH = (
    "data/rfinject/s1a-iw-raw-s-vh-20190610t052901-20190610t052933-027613-031dc6.zarr/burst_36"
)
BLOCK_SIZE = 4
SEEDS = list(range(42, 52))
N_KERNELS = 5000

RESULTS_DIR = Path("results/sar_rfi")


def load_burst() -> tuple[np.ndarray, np.ndarray]:
    z = zarr.open(BURST_PATH, mode="r")
    echo = np.asarray(z["echo"])  # (n_lines, n_range)
    rfi = np.asarray(z["rfi"])  # (n_realizations, n_lines, n_range)
    return echo, rfi


def valid_realizations(rfi: np.ndarray) -> list[int]:
    return [k for k in range(rfi.shape[0]) if np.count_nonzero(rfi[k]) > 0]


def block_max_magnitude(signal: np.ndarray, block_size: int) -> np.ndarray:
    """Magnitude of a complex 1D signal, block-max-pooled to reduce length
    while preserving sparse, localized interference peaks (better suited to
    this than block-averaging, which would dilute narrow RFI signatures)."""
    mag = np.abs(signal)
    n_blocks = mag.shape[-1] // block_size
    trimmed = mag[..., : n_blocks * block_size]
    return trimmed.reshape(*trimmed.shape[:-1], n_blocks, block_size).max(axis=-1)


def build_dataset(echo: np.ndarray, rfi: np.ndarray, valid_k: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Build a fixed clean-vs-RFI dataset (no seed dependence): interference in
    RFInject is sparse not just across range samples but across azimuth lines
    too -- most realizations inject energy into only a handful of a burst's
    lines, not all of them. Pairing every line with a randomly chosen
    realization (an earlier version of this function) therefore mislabeled
    most "contaminated" examples as positive when the chosen realization had
    no actual energy at that specific line, which produced a systematically
    anti-correlated (sub-chance) classifier that was in fact learning the
    right signal, just against ~90% wrong labels. Instead: every genuine
    (line, realization) pair with nonzero injected energy becomes one positive
    example; every original line becomes one negative (clean) example. Only
    the train/test split and model randomness vary by seed, matching the
    dataset-fixed, split-varies-by-seed protocol used elsewhere in this paper.
    """
    n_lines = echo.shape[0]
    clean = block_max_magnitude(echo, BLOCK_SIZE)  # (n_lines, L)

    contaminated_rows = [
        echo[i] + rfi[k, i, :]
        for i in range(n_lines)
        for k in valid_k
        if np.count_nonzero(rfi[k, i, :]) > 0
    ]
    contaminated_complex = np.stack(contaminated_rows, axis=0)
    contaminated = block_max_magnitude(contaminated_complex, BLOCK_SIZE)

    X = np.concatenate([clean, contaminated], axis=0).astype("float32")
    y = np.concatenate([np.zeros(len(clean)), np.ones(len(contaminated))]).astype("int64")
    # Per-series z-normalization, standard ROCKET practice.
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True).clip(min=1e-6)
    X = (X - mu) / sigma
    return X, y


def run_rocket(X_tr, y_tr, X_te, y_te, seed: int) -> dict:
    t0 = time.perf_counter()
    clf = Rocket1DClassifier(n_kernels=N_KERNELS, seed=seed, device="cpu")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    elapsed = time.perf_counter() - t0
    return _metrics(y_te, y_pred, elapsed)


def run_svm(X_tr, y_tr, X_te, y_te, seed: int) -> dict:
    t0 = time.perf_counter()
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LinearSVC(C=0.1, max_iter=20000, random_state=seed)
    clf.fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_te_s)
    elapsed = time.perf_counter() - t0
    return _metrics(y_te, y_pred, elapsed)


def run_cnn(X_tr, y_tr, X_te, y_te, seed: int) -> dict:
    set_seed(seed)
    t0 = time.perf_counter()
    device = "cpu"
    seq_len = X_tr.shape[1]
    model = SimpleCNN1D(num_classes=2, seq_len=seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    X_tr_t = torch.tensor(X_tr[:, None, :], dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.int64)
    ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)
    model.train()
    for _epoch in range(30):
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te[:, None, :], dtype=torch.float32).to(device)
        y_pred = model(X_te_t).argmax(dim=1).cpu().numpy()
    elapsed = time.perf_counter() - t0
    return _metrics(y_te, y_pred, elapsed)


def _metrics(y_true, y_pred, elapsed: float) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": cm,
        "seconds": elapsed,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    echo, rfi = load_burst()
    valid_k = valid_realizations(rfi)
    X, y = build_dataset(echo, rfi, valid_k)
    print(f"Burst: {echo.shape[0]} lines, {echo.shape[1]} range samples, "
          f"{len(valid_k)}/{rfi.shape[0]} non-empty RFI realizations: {valid_k}")
    print(f"Dataset: {int((y == 0).sum())} clean, {int((y == 1).sum())} contaminated, "
          f"seq_len={X.shape[1]}")

    for seed in SEEDS:
        out_path = RESULTS_DIR / f"seed{seed}.json"
        if out_path.exists():
            print(f"[skip] seed={seed} already done")
            continue
        print(f"[start] seed={seed}", flush=True)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        result = {
            "seed": seed,
            "n_train": len(y_tr),
            "n_test": len(y_te),
            "seq_len": X.shape[1],
            "rocket": run_rocket(X_tr, y_tr, X_te, y_te, seed),
            "svm": run_svm(X_tr, y_tr, X_te, y_te, seed),
            "cnn": run_cnn(X_tr, y_tr, X_te, y_te, seed),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(
            f"[done] seed={seed}: "
            f"rocket={result['rocket']['accuracy']:.4f} "
            f"svm={result['svm']['accuracy']:.4f} "
            f"cnn={result['cnn']['accuracy']:.4f}",
            flush=True,
        )

    print("All seeds complete.", flush=True)


if __name__ == "__main__":
    main()
