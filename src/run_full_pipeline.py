"""
Run the full Fed-LLM CPS pipeline.

Step 1: run TA-FedX-CPS detector.
Step 2: pass the exported detector event into the Fed-LLM incident agent.

Use:
    python src/run_full_pipeline.py

Note:
    The detector requires the UNSW-NB15 CSV files for real results.
    If they are missing, it runs in synthetic demo mode.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DETECTOR = ROOT / "src" / "ta_fedx_cps_detector.py"
AGENT = ROOT / "src" / "fed_llm_cps_agent.py"
EVENT = ROOT / "fedx_had_outputs" / "incident_event_for_agent.json"
REPORT = ROOT / "fed_llm_incident_report.md"


def run(cmd):
    print("\n[RUN]", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():
    run([sys.executable, str(DETECTOR)])
    if not EVENT.exists():
        raise FileNotFoundError(f"Detector did not create expected event file: {EVENT}")
    run([sys.executable, str(AGENT), "--event", str(EVENT), "--out", str(REPORT)])
    print(f"\n[DONE] Full pipeline report: {REPORT}")


if __name__ == "__main__":
    main()
