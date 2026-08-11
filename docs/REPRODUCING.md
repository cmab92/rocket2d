# Reproducing the results in `paper.tex`

This file documents, section by section, how to reproduce every numeric result
reported in the paper — and, just as importantly, which results *cannot* be
reproduced from this repository at all.

## 0. Scope: what is and isn't reproducible here

**Reproducible from this repository:** Sec. 5.4 (Real-world validation: NEU,
Chest X-Ray, DTD), Sec. 5.5 (LSPV) and the capacity-control experiment
referenced there — everything built on the `rocket2d` package under `src/`.

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

## 6. Building the PDF

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

## 7. Tests

```bash
pytest            # synthetic-image fixtures, no dataset download needed
ruff check .
mypy src
```
