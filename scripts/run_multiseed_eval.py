"""Multi-seed CNN vs 2D-ROCKET evaluation on NEU/X-Ray/DTD for variance estimation.

Each (model, dataset, seed) job is a fresh spawned process with a capped
thread count, so several jobs can run concurrently on a multi-core CPU
without the severe thread-contention overhead observed at full (32-thread)
default parallelism within a single process. Not part of the installed
package; a one-off driver for the significance-testing follow-up.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

DATASETS = ["neu", "xray", "dtd"]
MODELS = ["cnn", "rocket"]
NEW_SEEDS = list(range(43, 52))  # seed 42 reused from results/summary.json
THREADS_PER_WORKER = 4
N_WORKERS = 2

RESULTS_DIR = Path("results")
MULTISEED_DIR = RESULTS_DIR / "multiseed"


def _run_job(args: tuple[str, str, int]) -> None:
    model, dataset, seed = args
    out_path = MULTISEED_DIR / f"{model}_{dataset}_seed{seed}.json"
    if out_path.exists():
        return

    print(f"[start] {model}/{dataset}/seed={seed}", flush=True)

    import os

    os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_WORKER)
    os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_WORKER)

    import torch

    torch.set_num_threads(THREADS_PER_WORKER)

    from rocket2d.config import Config
    from rocket2d.training import run_cnn, run_rocket

    config = Config(seed=seed, base_data_dir="data")
    t0 = time.perf_counter()
    if model == "cnn":
        metrics = run_cnn(
            dataset, config.dataset_dirs[dataset], img_size=128, batch_size=32,
            epochs=30, seed=seed, device="cpu", show_plots=False, save_dir=None,
        )
    else:
        metrics = run_rocket(
            dataset, config.dataset_dirs[dataset], img_size=128, seed=seed,
            n_kernels=5000, device="cpu", show_plots=False, save_dir=None,
        )
    elapsed = time.perf_counter() - t0

    result = {
        "model": model, "dataset": dataset, "seed": seed,
        "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"], "seconds": elapsed,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[done] {model}/{dataset}/seed={seed}: acc={metrics['accuracy']:.4f} ({elapsed:.1f}s)", flush=True)


def main() -> None:
    MULTISEED_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        (model, dataset, seed)
        for model in MODELS
        for dataset in DATASETS
        for seed in NEW_SEEDS
    ]
    # Cheapest jobs first so partial progress is maximally useful if interrupted.
    cost_rank = {"neu": 0, "dtd": 1, "xray": 2}
    jobs.sort(key=lambda j: (0 if j[0] == "cnn" else 1, cost_rank[j[1]]))

    print(f"Launching {len(jobs)} jobs across {N_WORKERS} workers "
          f"({THREADS_PER_WORKER} threads each)...", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=N_WORKERS) as pool:
        for _ in pool.imap_unordered(_run_job, jobs):
            pass
    print("All jobs complete.", flush=True)


if __name__ == "__main__":
    main()
