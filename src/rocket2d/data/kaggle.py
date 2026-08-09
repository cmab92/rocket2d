"""Kaggle dataset acquisition and directory normalization.

Replaces the notebook's shell-magic cells (``!kaggle ...``, ``!cp``, ``!find``)
with plain subprocess/pathlib calls so it runs as a regular Python module.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

KAGGLE_DATASETS: dict[str, str] = {
    "neu": "kaustubhdikshit/neu-surface-defect-database",
    "xray": "paultimothymooney/chest-xray-pneumonia",
    "dtd": "jmexpert/describable-textures-dataset-dtd",
}


def ensure_kaggle_credentials(source: Path | str = "kaggle.json") -> None:
    """Install a ``kaggle.json`` API token into ``~/.kaggle`` if not already present.

    Parameters
    ----------
    source : Path or str
        Path to a local ``kaggle.json`` credentials file, used only if
        ``~/.kaggle/kaggle.json`` does not already exist.
    """
    target = Path.home() / ".kaggle" / "kaggle.json"
    if target.exists():
        return
    source = Path(source)
    if not source.exists():
        logger.warning(
            "kaggle.json not found at %s. Please provide a Kaggle API token "
            "for dataset downloads (see https://www.kaggle.com/docs/api).",
            source,
        )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)
    target.chmod(0o600)


def move_if_exists(src: Path | str, dst: Path | str) -> None:
    """Move a source directory or file to ``dst`` if ``src`` exists and ``dst`` does not.

    Parameters
    ----------
    src : Path or str
        Source path.
    dst : Path or str
        Destination path.
    """
    src, dst = Path(src), Path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.move(str(src), str(dst))


def _download_dataset(slug: str, out_dir: Path) -> None:
    """Download and unzip a single Kaggle dataset via the ``kaggle`` CLI."""
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir), "--unzip"],
        check=True,
    )


def _remove_junk_files(base_dir: Path) -> None:
    """Remove macOS resource-fork artifacts (``__MACOSX``, ``._*``) recursively."""
    for junk_dir in base_dir.rglob("__MACOSX"):
        shutil.rmtree(junk_dir, ignore_errors=True)
    for junk_file in base_dir.rglob("._*"):
        junk_file.unlink(missing_ok=True)


def download_and_prepare_datasets(
    base_dir: Path | str, datasets: dict[str, str] | None = None
) -> None:
    """Download the required Kaggle datasets and normalize their directory layout.

    Parameters
    ----------
    base_dir : Path or str
        Root directory under which each dataset is downloaded (``base_dir/<name>``).
    datasets : dict, optional
        Mapping of dataset name to Kaggle slug. Defaults to :data:`KAGGLE_DATASETS`
        (``neu``, ``xray``, ``dtd``).
    """
    base_dir = Path(base_dir)
    datasets = datasets or KAGGLE_DATASETS

    for name, slug in datasets.items():
        out_dir = base_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if any(out_dir.iterdir()):
            logger.info("%s already exists, skipping download.", name)
            continue
        logger.info("Downloading %s ...", name)
        _download_dataset(slug, out_dir)

    _remove_junk_files(base_dir)

    # Normalize DTD: base/dtd/dtd/images -> base/dtd/images
    move_if_exists(base_dir / "dtd" / "dtd" / "images", base_dir / "dtd" / "images")
    shutil.rmtree(base_dir / "dtd" / "dtd", ignore_errors=True)

    # Normalize chest X-ray: base/xray/chest_xray/chest_xray/{split} -> base/xray/{split}
    xray_root = base_dir / "xray" / "chest_xray" / "chest_xray"
    for split in ("train", "test", "val"):
        move_if_exists(xray_root / split, base_dir / "xray" / split)
    shutil.rmtree(base_dir / "xray" / "chest_xray", ignore_errors=True)

    # Normalize NEU: base/neu/NEU-DET/{split}/images -> base/neu/{split}
    neu_root = base_dir / "neu" / "NEU-DET"
    for split in ("train", "validation"):
        move_if_exists(neu_root / split / "images", base_dir / "neu" / split)
    shutil.rmtree(neu_root, ignore_errors=True)

    logger.info("All datasets normalized and ready under %s", base_dir)


def print_directory_tree(path: Path | str, max_depth: int = 3) -> None:
    """Print a directory tree (directories only) rooted at ``path``.

    Parameters
    ----------
    path : Path or str
        Root of the directory structure to display.
    max_depth : int, optional
        Maximum depth of subdirectories to descend into (default 3).
    """

    def _walk(dir_path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        subdirs = sorted(p for p in dir_path.iterdir() if p.is_dir())
        for i, sub in enumerate(subdirs):
            connector = "└── " if i == len(subdirs) - 1 else "├── "
            print(f"{prefix}{connector}{sub.name}")
            extension = "    " if i == len(subdirs) - 1 else "│   "
            _walk(sub, prefix + extension, depth + 1)

    root = Path(path)
    print(root)
    _walk(root, "", 1)
