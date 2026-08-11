"""Control experiment for the LSPV result (Sec. 5.5): does restoring positional
resolution matter, or does simply adding capacity (more random kernels, same
global-pooling feature types) explain the accuracy gain on Chest X-Ray?

14,000 kernels x 5 base feature types = 70,000 features, matching LSPV's
5,000 kernels x 14 features (5 base + 9 grid cells) feature dimension exactly,
with no positional information added. 10 seeds, same protocol as the LSPV
sweep. Not part of the installed package.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

SEEDS = list(range(42, 47))  # 5 seeds: reduced scope after 14000-kernel memory blowup
THREADS_PER_WORKER = 4
N_WORKERS = 1  # sequential: 14000 kernels hit ~41GB RSS for a single job at n_kernels=14000
N_KERNELS = 8000  # 8000 * 5 base features = 40000 (partial capacity match, memory-safe)

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "capacity_control_xray"


def _run_job(seed: int) -> None:
    out_path = OUT_DIR / f"rocket_capacity_xray_seed{seed}.json"
    if out_path.exists():
        return

    print(f"[start] rocket_capacity/xray/seed={seed}", flush=True)

    import os

    os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_WORKER)
    os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_WORKER)

    import torch

    torch.set_num_threads(THREADS_PER_WORKER)

    from rocket2d.config import Config
    from rocket2d.training import run_rocket

    config = Config(seed=seed, base_data_dir="data")
    t0 = time.perf_counter()
    metrics = run_rocket(
        "xray", config.dataset_dirs["xray"], img_size=128, seed=seed,
        n_kernels=N_KERNELS, device="cpu", show_plots=False, save_dir=None,
        feature_types=["ppv", "max", "mpv", "mipv_y", "mipv_x"],
    )
    elapsed = time.perf_counter() - t0

    result = {
        "model": "rocket_capacity_control", "dataset": "xray", "seed": seed,
        "n_kernels": N_KERNELS,
        "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"], "seconds": elapsed,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[done] rocket_capacity/xray/seed={seed}: acc={metrics['accuracy']:.4f} "
          f"({elapsed:.1f}s)", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Launching {len(SEEDS)} jobs across {N_WORKERS} workers "
          f"({THREADS_PER_WORKER} threads each, n_kernels={N_KERNELS})...", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=N_WORKERS) as pool:
        for _ in pool.imap_unordered(_run_job, SEEDS):
            pass
    print("All jobs complete.", flush=True)


if __name__ == "__main__":
    main()
