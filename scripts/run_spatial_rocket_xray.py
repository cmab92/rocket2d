"""10-seed sweep for 2D-ROCKET with an added LSPV (regional PPV) feature on
X-Ray, testing whether restoring coarse positional resolution closes the gap
to the CNN on this non-texture, localized-pathology task. Not part of the
installed package.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

SEEDS = list(range(42, 52))
THREADS_PER_WORKER = 4
N_WORKERS = 2

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "spatial_rocket_xray"


def _run_job(seed: int) -> None:
    out_path = OUT_DIR / f"rocket_spatial_xray_seed{seed}.json"
    if out_path.exists():
        return

    print(f"[start] rocket_spatial/xray/seed={seed}", flush=True)

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
        n_kernels=5000, device="cpu", show_plots=False, save_dir=None,
        feature_types=["ppv", "max", "mpv", "mipv_y", "mipv_x", "lspv"], lspv_grid=3,
    )
    elapsed = time.perf_counter() - t0

    result = {
        "model": "rocket_spatial", "dataset": "xray", "seed": seed,
        "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"], "seconds": elapsed,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[done] rocket_spatial/xray/seed={seed}: acc={metrics['accuracy']:.4f} ({elapsed:.1f}s)", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Launching {len(SEEDS)} jobs across {N_WORKERS} workers "
          f"({THREADS_PER_WORKER} threads each)...", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=N_WORKERS) as pool:
        for _ in pool.imap_unordered(_run_job, SEEDS):
            pass
    print("All jobs complete.", flush=True)


if __name__ == "__main__":
    main()
