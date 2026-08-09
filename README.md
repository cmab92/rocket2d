# rocket2d

ROCKET (Random Convolutional Kernel Transform) and a CNN baseline for
texture-dominant image classification — industrial surface defects (NEU),
chest X-rays (pneumonia), and describable textures (DTD).

This is a from-scratch Python package reimplementation of the exploratory
notebook in [`legacy/`](legacy/); it is kept only as historical reference and
is not used by this package.

## Project layout

```
src/rocket2d/
├── config.py          # Config dataclass, reproducibility (set_seed)
├── data/
│   ├── kaggle.py       # Kaggle download + dataset directory normalization
│   ├── datasets.py     # ImageDataset: folder-structured loading -> numpy/torch
│   └── loaders.py      # train/test split, PyTorch DataLoader construction
├── models/
│   ├── cnn.py           # SimpleCNN baseline
│   └── rocket.py        # RocketClassifier (random kernels + RidgeClassifier)
├── visualization.py     # sample grids, histograms, confusion matrix, ROC/PR
├── training.py           # run_cnn / run_rocket end-to-end pipelines
└── cli.py                 # `rocket2d` console entry point
tests/                      # pytest suite using synthetic image fixtures
```

## Installation

Requires Python >= 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

The `rocket2d` CLI mirrors the stages of the original notebook:

```bash
# 1. Download and normalize the Kaggle datasets into ./data
rocket2d download

# 2. Inspect a dataset: class distribution, sample grid, size/histogram stats
rocket2d inspect neu

# 3. Train and evaluate a model
rocket2d train cnn neu
rocket2d train rocket neu

# 4. Or run everything (CNN + ROCKET on neu, xray, dtd)
rocket2d train all
```

`rocket2d download` requires Kaggle API credentials at `~/.kaggle/kaggle.json`
(see the [Kaggle API docs](https://www.kaggle.com/docs/api)).

Global options: `--data-dir` (default `data`), `--seed` (default `42`),
`-v/--verbose`. Run `rocket2d <command> --help` for command-specific options
(image size, batch size, epochs, device, number of ROCKET kernels, ...).

## Development

```bash
pytest          # run the test suite (uses synthetic images, no dataset download needed)
ruff check .    # lint
mypy src        # type-check
```
