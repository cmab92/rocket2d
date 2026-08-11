"""10-seed linear-SVM baseline sweep on NEU/X-Ray/DTD, added retroactively to
match the CNN/2D-ROCKET multi-seed protocol in Sec. 5.4/5.5 -- the SVM
baseline used throughout the MNIST/CIFAR-10/KTH-TIPS study was missing from
the real-world validation entirely. Not part of the installed package.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

DATASETS = ["neu", "xray", "dtd"]
SEEDS = list(range(42, 52))
N_WORKERS = 6  # SVM jobs are single-threaded sklearn, low memory -> safe to parallelize more

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "svm_multiseed"


def _run_job(args: tuple[str, int]) -> None:
    dataset, seed = args
    out_path = OUT_DIR / f"svm_{dataset}_seed{seed}.json"
    if out_path.exists():
        return

    print(f"[start] svm/{dataset}/seed={seed}", flush=True)

    from rocket2d.config import Config
    from rocket2d.training import run_svm

    config = Config(seed=seed, base_data_dir="data")
    t0 = time.perf_counter()
    metrics = run_svm(
        dataset, config.dataset_dirs[dataset], img_size=128, seed=seed,
        show_plots=False, save_dir=None,
    )
    elapsed = time.perf_counter() - t0

    result = {
        "model": "svm", "dataset": dataset, "seed": seed,
        "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"], "seconds": elapsed,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[done] svm/{dataset}/seed={seed}: acc={metrics['accuracy']:.4f} ({elapsed:.1f}s)", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(dataset, seed) for dataset in DATASETS for seed in SEEDS]
    cost_rank = {"neu": 0, "dtd": 1, "xray": 2}
    jobs.sort(key=lambda j: cost_rank[j[0]])
    print(f"Launching {len(jobs)} jobs across {N_WORKERS} workers...", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=N_WORKERS) as pool:
        for _ in pool.imap_unordered(_run_job, jobs):
            pass
    print("All jobs complete.", flush=True)


if __name__ == "__main__":
    main()
