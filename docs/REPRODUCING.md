# Reproducing the results in `paper.tex`

This file documents, section by section, how to reproduce every numeric result
reported in the paper — and, just as importantly, which results *cannot* be
reproduced from this repository at all.

## 0. Scope: what is and isn't reproducible here

**Reproducible from this repository:** Sec. 5.4 (Real-world validation: NEU,
Chest X-Ray, DTD), Sec. 5.5 (LSPV) and the capacity-control experiment
referenced there, and Sec. 5.6 (SAR RFI detection) — everything built on the
`rocket2d` package under `src/`.

**Not reproducible from this repository:** Table 1 and Sec. 5.1–5.3 (the
MNIST/CIFAR-10/1D-vs-2D/padding/preprocessing/kernel-count ablations and the
KTH-TIPS proof of concept). These numbers come from Felicitas Lock's original
bachelor-thesis work; `src/rocket2d` has no MNIST, CIFAR-10, or KTH-TIPS
dataset support at all (`Config.dataset_dirs` only knows `neu`, `xray`, `lc`,
`dtd`), and `legacy/Untitled7.ipynb` — the one notebook kept in this repo —
only covers NEU/X-Ray/LC25000/DTD as well, not MNIST/CIFAR/KTH-TIPS. That
ablation study lives in a separate notebook that is not part of this
repository; `docs/bachelor_thesis_felicitas_lock_36434.pdf` is the closest
available source for its methodology. Any attempt to reproduce Table 1
requires reconstructing that pipeline from scratch, or obtaining the original
notebook from the thesis author.

## 1. Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python >= 3.10. All experiments below were run on CPU; no GPU is
required (though `rocket2d`'s `--device` flag will use CUDA if available).

## 2. Download datasets

```bash
rocket2d download   # requires Kaggle API credentials at ~/.kaggle/kaggle.json
```

This populates `data/neu`, `data/xray`, `data/dtd` (`data/lc25000` is fetched
too but unused by anything in the paper). Expected sizes after normalization:
NEU 1,800 images / 6 classes, Chest X-Ray ~5,860 images / 2 classes, DTD
5,640 images / 47 classes.

## 3. Table 2 — real-world validation (Sec. 5.4)

The paper's numbers are mean ± std over 10 seeds (42–51), plus a paired
$t$-test per dataset. Reproduce with:

```bash
python scripts/run_full_eval.py       # seed 42 only, all 3 datasets, both models
python scripts/run_multiseed_eval.py  # seeds 43-51, all 3 datasets, both models
python scripts/aggregate_multiseed.py # aggregates all 10 seeds + paired t-tests
```

Notes:
- `run_multiseed_eval.py` runs jobs in parallel (`N_WORKERS = 2`,
  `THREADS_PER_WORKER = 4` by default). Each ROCKET job can transiently use
  10–15GB RSS; do not raise `N_WORKERS` without checking `free -h` during a
  run, or a 5-worker attempt at these settings pushed the machine into swap
  and stalled two of five workers indefinitely on this repo's dev machine.
- CNN uses 30 epochs (not the 10 used for MNIST/CIFAR-10 in Table 1) — an
  initial 10-epoch pass left DTD's training loss still falling steeply
  (Fig. 5), so the longer schedule was adopted uniformly across all three
  datasets. `run_cnn`/`run_rocket` accept `save_dir` to persist confusion
  matrices, misclassified-example grids, and (for CNN) the learning curve.
- Total wall time: budget several hours. Feature extraction for ROCKET on
  Chest X-Ray/DTD (~4,700–4,700 training images, 5,000 kernels) is the
  dominant cost; a single uncontended run is ~1,000–1,100s, but concurrent
  workers on shared hardware inflate this — the "Time (s)" column in Table 2
  is a single uncontended seed-42 run precisely to avoid reporting that
  inflated, contention-dependent number.
- Confusion-matrix figures (Fig. 3/4) and the DTD learning-curve figure
  (Fig. 5) are regenerated from `results/run_full_eval.log` (or by directly
  editing the hardcoded arrays in `scripts/plot_confusion_figures.py` if you
  change the seed) via `python scripts/plot_confusion_figures.py`, which
  writes to `docs/figures/`.
- The linear-SVM row in Table 2 is produced by `scripts/run_svm_multiseed.py`
  (`run_svm` in `src/rocket2d/training.py`: flattened, standardized pixels,
  Gaussian random projection to 2,048 dims, then `SGDClassifier(loss="hinge")`
  with `alpha` chosen from `[1e-4, 1e-3, 1e-2]` on an internal validation
  split). This is *not* the same solver used for MNIST/CIFAR-10 in Table 1
  (`LinearSVC`, `C=0.1`, `max_iter=20000`): `LinearSVC`'s liblinear solver did
  not converge within a practical time budget on DTD's 4,512-sample, 47-class
  one-vs-rest fit (13+ minutes, still running), whereas `SGDClassifier` scales
  near-linearly and matched or beat a fully-converged `LinearSVC` on NEU in a
  direct side-by-side check, so it is used for all three real-world datasets
  instead. Budget ~30–40 min for NEU/10 seeds, ~1.5h for Chest X-Ray, and
  ~2.5–3h for DTD (each DTD seed is ~900–1,100s even after the fix below).
- **`SGDClassifier(n_jobs=-1)` inside a `multiprocessing`-parallel sweep
  causes severe CPU oversubscription**, especially for DTD's 47-class
  one-vs-rest fits (6 parallel workers x unbounded per-worker OVR threads).
  `run_svm` therefore hardcodes `n_jobs=2`; do not raise this without also
  lowering the sweep's own worker count, or per-worker CPU% will blow past
  500% and wall time will not improve.
- **The SAR RFI experiment (Sec. 6 below) is a deliberate exception**: its
  `run_svm` (local to `scripts/run_sar_rfi_experiment.py`) uses `LinearSVC`,
  not `SGDClassifier`. That dataset is small-n/high-p (172 training examples,
  4,913 features) and binary, the regime `LinearSVC`'s dual solver is fast
  and accurate in — checked directly, `LinearSVC` reached 79.5% there against
  70.4% best-of-grid for `SGDClassifier`. Do not "fix" this to use
  `SGDClassifier` for consistency; it is a verified regime-dependent choice,
  not an oversight.

## 4. Table 3 — LSPV (Sec. 5.5)

```bash
python scripts/run_spatial_rocket_xray.py
```

Runs `RocketClassifier` with `feature_types=["ppv","max","mpv","mipv_y",
"mipv_x","lspv"]`, `lspv_grid=3`, `n_kernels=5000` on Chest X-Ray, seeds
42–51 (`N_WORKERS = 2`). The LSPV feature itself lives in
`src/rocket2d/models/rocket.py::RocketClassifier.compute_features_2d` — it
adaptively average-pools each kernel's binary activation mask into an
`lspv_grid x lspv_grid` grid via `F.adaptive_avg_pool2d`, giving each kernel
coarse positional resolution instead of one global PPV number. Aggregate with
the same pattern as `aggregate_multiseed.py` (adjust the glob/paths — this
result was aggregated ad hoc, not via a checked-in aggregation script).

## 5. Capacity-vs-locality control (referenced in Sec. 5.5's discussion)

Tests whether LSPV's improvement is specifically about restored positional
information, or just about having more features for the Ridge fit to work
with:

```bash
python scripts/run_capacity_control_xray.py
```

Runs plain `RocketClassifier` (no LSPV) at `n_kernels=8000` (40,000 features,
a partial match to LSPV's 70,000 — the original plan of `n_kernels=14000` for
an exact match pushed a single job to ~41GB RSS and was abandoned after it
started swapping the machine) on Chest X-Ray, 5 seeds (42–46),
`N_WORKERS = 1` (sequential — this configuration is not safe to parallelize
on a 60GB machine). Compare its accuracy distribution against both the
original 5,000-kernel ROCKET (no LSPV) and the LSPV run above.

## 6. Table 4 — SAR RFI detection (Sec. 5.6)

This section uses real, third-party data (RFInject-v1-L0, an ESA Φ-lab
Hugging Face bucket of Sentinel-1 Level-0 raw bursts with simulated RFI) and
package dependencies not required anywhere else in this repository.

```bash
git clone https://github.com/sirbastiano/rfinject-utils.git /tmp/rfinject-utils
pip install zarr huggingface-hub folium pandas
pip install -e /tmp/rfinject-utils --no-deps
```

Download one small burst (~14MB; the bucket also contains a single monolithic
411GB mirror at `RFInject/v1` on Hugging Face — do not use that, use the
bucket-native selective access below instead):

```python
from rfinject import DEFAULT_HF_BUCKET_ID, download_hf_bucket_path
download_hf_bucket_path(
    DEFAULT_HF_BUCKET_ID,
    "s1a-iw-raw-s-vh-20190610t052901-20190610t052933-027613-031dc6.zarr/burst_36",
    "data/rfinject",
)
```

Then run:

```bash
python scripts/run_sar_rfi_experiment.py   # 10 seeds, ~3 min total
python -m scripts.plot_sar_figures         # Figs. 6-7
```

Important, hard-won details:
- **Interference is sparse across azimuth lines, not just range samples.**
  Most of the burst's 15 interference realizations inject energy into only a
  handful of its 102 lines each, not all of them. `build_dataset()` in
  `scripts/run_sar_rfi_experiment.py` only labels a `(line, realization)` pair
  as contaminated if it actually contains non-zero injected energy (114 such
  pairs in this burst) — an earlier version paired every line with a randomly
  chosen realization regardless, which mislabeled ~90% of the "contaminated"
  class and produced classifiers that scored *below chance* despite correctly
  learning the real signal (see Sec. 5.6 for the full account; verify this
  isn't silently reintroduced if you modify the pairing logic, e.g. by
  checking that `(y_true == y_pred).mean() + (y_true == (1 - y_pred)).mean()
  == 1.0` and that the former, not the latter, is the larger of the two).
- `rfi[k]` is the interference-only array (additive model); a contaminated
  signal is `echo + rfi[k]`, not `rfi[k]` alone.
- The genuine 1D-ROCKET implementation lives in
  `src/rocket2d/models/rocket1d.py` (kernel lengths `{7,9,11}`, dilations
  `{1,2,4}`, random same/valid padding per kernel, PPV+Max, grouped/batched
  `conv1d` calls for tractable runtime at thousands of kernels) — this is a
  fresh implementation, not a reuse of the 2D `RocketClassifier`, since PyTorch
  `conv2d` kernels in that class are always square and cannot represent a
  length-only 1D kernel without reshaping hacks.
- The dataset itself (216 examples: 102 clean + 114 genuinely contaminated) is
  fixed; only the seed on `main()`'s per-seed loop is used to vary the
  train/test split and model initialization, matching the split-varies-not-
  the-data protocol used for Table 2/3.

## 7. Building the PDF

```bash
cd docs
make          # -> build/paper.pdf, via latexmk + pdflatex + bibtex
```

Requires `latexmk` and a LaTeX distribution with the Elsevier `cas-sc` class
(see `docs/README.md` for the exact `texlive-*` package list). After a
successful build, `docs/paper.pdf` is a checked-in convenience copy of
`docs/build/paper.pdf` (the latter is git-ignored as a build artifact) —
remember to `cp docs/build/paper.pdf docs/paper.pdf` before committing if
you rebuild it.

## 8. Tests

```bash
pytest            # synthetic-image fixtures, no dataset download needed
ruff check .
mypy src
```
