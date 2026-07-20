"""
Run clean vs poisoned-client TA-FedX-CPS experiments.

Use in Colab:
    !python src/run_poisoning_comparison.py

This script does not modify the original UNSW-NB15 CSV files. It runs the
detector twice:
    1. clean federated clients
    2. one controlled label-poisoned federated client
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DETECTOR = ROOT / "src" / "ta_fedx_cps_detector.py"
CLEAN_DIR = ROOT / "fedx_had_outputs_clean"
POISONED_DIR = ROOT / "fedx_had_outputs_poisoned"
SUMMARY = ROOT / "poisoning_comparison_summary.csv"


def run_detector(label: str, args: list[str]) -> None:
    print(f"\n[COMPARE] Running {label} experiment")
    cmd = [sys.executable, str(DETECTOR), *args]
    print("[RUN]", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def load_metrics(label: str, folder: Path) -> pd.DataFrame:
    metrics_path = folder / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    df = pd.read_csv(metrics_path)
    df.insert(0, "Experiment", label)
    return df


def main() -> None:
    run_detector("clean", ["--no-poison", "--output-dir", str(CLEAN_DIR)])
    run_detector(
        "poisoned_client",
        ["--poison", "--poison-client", "1", "--poison-fraction", "0.35", "--output-dir", str(POISONED_DIR)],
    )

    combined = pd.concat(
        [
            load_metrics("clean", CLEAN_DIR),
            load_metrics("poisoned_client", POISONED_DIR),
        ],
        ignore_index=True,
    )
    combined.to_csv(SUMMARY, index=False)

    focus = combined[
        combined["Method"].isin(["FedAvg IDS", "Robust IDS", "Proposed TA-FedX-IDS", "Adaptive CDAW"])
    ]
    print("\n================ POISONING COMPARISON ================")
    print(focus.to_string(index=False))
    print("======================================================")
    print(f"\n[DONE] Combined comparison saved to: {SUMMARY}")
    print(f"[DONE] Clean figures saved in: {CLEAN_DIR}")
    print(f"[DONE] Poisoned figures saved in: {POISONED_DIR}")


if __name__ == "__main__":
    main()
