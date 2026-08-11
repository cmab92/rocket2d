"""Aggregate the multi-seed sweep into mean/std and a paired significance test
(CNN vs 2D-ROCKET accuracy, matched by seed) per dataset.

Reuses the existing seed-42 single run from results/summary.json as one of
the ten samples. Not part of the installed package.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from scipy import stats

DATASETS = ["neu", "xray", "dtd"]
MODELS = ["cnn", "rocket"]
ALL_SEEDS = list(range(42, 52))  # 10 seeds: 42 (reused) + 43..51 (new)

RESULTS_DIR = Path("results")
MULTISEED_DIR = RESULTS_DIR / "multiseed"
SUMMARY_PATH = RESULTS_DIR / "summary.json"


def load_all() -> dict:
    base_summary = json.loads(SUMMARY_PATH.read_text())
    data: dict[str, dict[str, dict[int, dict]]] = {m: {d: {} for d in DATASETS} for m in MODELS}

    for model in MODELS:
        for dataset in DATASETS:
            entry = dict(base_summary[model][dataset])
            entry["seed"] = 42
            data[model][dataset][42] = entry

    for f in MULTISEED_DIR.glob("*.json"):
        r = json.loads(f.read_text())
        data[r["model"]][r["dataset"]][r["seed"]] = r

    return data


def main() -> None:
    data = load_all()
    report: dict = {}

    for dataset in DATASETS:
        report[dataset] = {}
        per_model_acc = {}
        for model in MODELS:
            seeds_present = sorted(data[model][dataset].keys())
            accs = [data[model][dataset][s]["accuracy"] for s in seeds_present]
            f1s = [data[model][dataset][s]["macro_f1"] for s in seeds_present]
            secs = [data[model][dataset][s]["seconds"] for s in seeds_present]
            per_model_acc[model] = (seeds_present, accs)
            report[dataset][model] = {
                "n_seeds": len(seeds_present),
                "seeds": seeds_present,
                "accuracy_mean": statistics.mean(accs),
                "accuracy_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                "accuracy_min": min(accs),
                "accuracy_max": max(accs),
                "macro_f1_mean": statistics.mean(f1s),
                "macro_f1_std": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
                "seconds_mean": statistics.mean(secs),
            }

        seeds_cnn, acc_cnn = per_model_acc["cnn"]
        seeds_rocket, acc_rocket = per_model_acc["rocket"]
        common = sorted(set(seeds_cnn) & set(seeds_rocket))
        if len(common) >= 2:
            paired_cnn = [data["cnn"][dataset][s]["accuracy"] for s in common]
            paired_rocket = [data["rocket"][dataset][s]["accuracy"] for s in common]
            diffs = [c - r for c, r in zip(paired_cnn, paired_rocket)]
            t_stat, p_value = stats.ttest_rel(paired_cnn, paired_rocket)
            report[dataset]["paired_test"] = {
                "n": len(common),
                "mean_diff_cnn_minus_rocket": statistics.mean(diffs),
                "std_diff": statistics.pstdev(diffs) if len(diffs) > 1 else 0.0,
                "t_stat": float(t_stat),
                "p_value": float(p_value),
            }

    out_path = RESULTS_DIR / "multiseed_summary.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
