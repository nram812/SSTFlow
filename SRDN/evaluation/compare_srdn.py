"""Compare matched SRDCNN and ResAFNO test evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure SRDN parent directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from srdn_metrics import paired_bootstrap_delta, write_json


def _load(run: Path):
    metrics_path = run / "evaluation" / "metrics.json"
    daily_path = run / "evaluation" / "daily_metrics.npz"
    if not metrics_path.exists() or not daily_path.exists():
        raise FileNotFoundError(f"missing evaluation outputs under {run}")
    with metrics_path.open() as handle:
        metrics = json.load(handle)
    daily = np.load(daily_path)
    return metrics, daily


def compare(resafno_run: Path, srdcnn_run: Path, output: Path | None = None):
    resafno, afno_daily = _load(resafno_run)
    srdcnn, cnn_daily = _load(srdcnn_run)
    if len(afno_daily["model_mse"]) != len(cnn_daily["model_mse"]):
        raise ValueError("SRDCNN and ResAFNO evaluations do not cover equal days")
    afno_mse = np.asarray(afno_daily["model_mse"], dtype=np.float64)
    cnn_mse = np.asarray(cnn_daily["model_mse"], dtype=np.float64)
    result = {
        "resafno": resafno,
        "srdcnn": srdcnn,
        "resafno_vs_srdcnn": paired_bootstrap_delta(afno_mse, cnn_mse),
        "decision_rule": (
            "ResAFNO is better only when ci95_high_c2 < 0 for the paired "
            "per-day MSE difference."
        ),
        "resafno_better_than_srdcnn": bool(
            paired_bootstrap_delta(afno_mse, cnn_mse)["win_at_95pct"]
        ),
    }
    if output is not None:
        write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resafno-run", required=True, type=Path)
    parser.add_argument("--srdcnn-run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    compare(args.resafno_run, args.srdcnn_run, args.output)


if __name__ == "__main__":
    main()
