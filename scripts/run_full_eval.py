"""Run CNN + ROCKET on every dataset and dump a structured results summary.

Not part of the installed package; a one-off driver for producing the
comprehensive results reported in docs/paper.tex.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rocket2d.config import Config
from rocket2d.training import run_cnn, run_rocket

DATASETS = ["neu", "xray", "dtd"]
RESULTS_DIR = Path("results")
SUMMARY_PATH = RESULTS_DIR / "summary.json"


def main() -> None:
    config = Config(seed=42, base_data_dir="data")
    summary: dict[str, dict[str, dict]] = {"cnn": {}, "rocket": {}}

    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())
        summary.setdefault("cnn", {})
        summary.setdefault("rocket", {})

    for name in DATASETS:
        if name in summary["cnn"]:
            print(f"[skip] cnn/{name} already in summary")
        else:
            print(f"\n{'=' * 30}\nRUN CNN: {name.upper()}\n{'=' * 30}")
            save_dir = RESULTS_DIR / "cnn" / name
            t0 = time.perf_counter()
            metrics = run_cnn(
                name,
                config.dataset_dirs[name],
                img_size=128,
                batch_size=32,
                epochs=10,
                seed=config.seed,
                device=config.device,
                show_plots=False,
                save_dir=str(save_dir),
            )
            elapsed = time.perf_counter() - t0
            summary["cnn"][name] = {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "seconds": elapsed,
            }
            SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    for name in DATASETS:
        if name in summary["rocket"]:
            print(f"[skip] rocket/{name} already in summary")
        else:
            print(f"\n{'=' * 30}\nRUN ROCKET: {name.upper()}\n{'=' * 30}")
            save_dir = RESULTS_DIR / "rocket" / name
            t0 = time.perf_counter()
            metrics = run_rocket(
                name,
                config.dataset_dirs[name],
                img_size=128,
                seed=config.seed,
                n_kernels=5000,
                device=config.device,
                show_plots=False,
                save_dir=str(save_dir),
            )
            elapsed = time.perf_counter() - t0
            summary["rocket"][name] = {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "seconds": elapsed,
            }
            SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print("\nDone. Summary written to", SUMMARY_PATH)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
