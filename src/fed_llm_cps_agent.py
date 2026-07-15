"""
Fed-LLM CPS Incident Commander

No-key Agentic RAG-style copilot for Cyber-Physical System incident response.
It consumes model outputs from a detector such as TA-FedX-CPS and retrieves
local runbooks to produce severity, diagnosis, remediation steps, and a
stakeholder update.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK_DIR = ROOT / "runbooks"


DEFAULT_EVENT = {
    "asset": "PLC-7 / Cooling Pump Line",
    "prediction": "cyber_attack",
    "confidence": 0.91,
    "adaptive_cdaw_score": 0.78,
    "client_trust": 0.42,
    "top_shap_features": ["ct_dst_src_ltm", "tcprtt", "dload", "service", "spkts"],
    "notes": "Spike in network timing and repeated destination-source communication.",
}


@dataclass
class Runbook:
    name: str
    path: Path
    text: str


def load_runbooks(runbook_dir: Path = RUNBOOK_DIR) -> List[Runbook]:
    runbooks = []
    for path in sorted(runbook_dir.glob("*.md")):
        runbooks.append(Runbook(name=path.stem, path=path, text=path.read_text(encoding="utf-8")))
    if not runbooks:
        raise FileNotFoundError(f"No runbooks found in {runbook_dir}")
    return runbooks


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", text.lower())


def event_to_query(event: Dict) -> str:
    parts = [
        str(event.get("prediction", "")),
        str(event.get("asset", "")),
        str(event.get("notes", "")),
        " ".join(map(str, event.get("top_shap_features", []))),
    ]
    score = float(event.get("adaptive_cdaw_score", 0.0))
    trust = float(event.get("client_trust", 1.0))
    if score >= 0.65:
        parts.append("high anomaly uncertainty suspicious abnormal")
    if trust <= 0.5:
        parts.append("low trust unreliable client poisoned abnormal update")
    return " ".join(parts)


def retrieve_runbooks(event: Dict, runbooks: List[Runbook], top_k: int = 2) -> List[Tuple[Runbook, float]]:
    query_terms = set(tokenize(event_to_query(event)))
    scored = []
    for runbook in runbooks:
        rb_terms = set(tokenize(runbook.text))
        overlap = len(query_terms & rb_terms)
        prediction = str(event.get("prediction", "")).lower()
        bonus = 0
        if "cyber" in prediction and "cyber" in runbook.name:
            bonus += 5
        if "fault" in prediction and "fault" in runbook.name:
            bonus += 5
        if float(event.get("adaptive_cdaw_score", 0.0)) >= 0.65 and "uncertain" in runbook.name:
            bonus += 3
        if float(event.get("client_trust", 1.0)) <= 0.5 and "uncertain" in runbook.name:
            bonus += 2
        scored.append((runbook, float(overlap + bonus)))
    return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


def classify_severity(event: Dict) -> str:
    confidence = float(event.get("confidence", 0.0))
    cdaw = float(event.get("adaptive_cdaw_score", 0.0))
    trust = float(event.get("client_trust", 1.0))
    prediction = str(event.get("prediction", "")).lower()

    if ("cyber" in prediction and confidence >= 0.85) or (cdaw >= 0.80 and trust <= 0.50):
        return "SEV-1"
    if confidence >= 0.70 or cdaw >= 0.60 or trust <= 0.55:
        return "SEV-2"
    return "SEV-3"


def infer_response_team(event: Dict) -> str:
    prediction = str(event.get("prediction", "")).lower()
    cdaw = float(event.get("adaptive_cdaw_score", 0.0))
    trust = float(event.get("client_trust", 1.0))

    if "cyber" in prediction and (cdaw >= 0.65 or trust <= 0.5):
        return "Cybersecurity team + control systems engineer"
    if "fault" in prediction:
        return "Maintenance/reliability team + control systems engineer"
    return "Maintenance and cybersecurity teams jointly"


def summarize_features(features: Iterable[str]) -> str:
    features = list(features)
    if not features:
        return "No SHAP feature explanation was provided."
    cyber_terms = {"proto", "service", "tcprtt", "dload", "spkts", "ct_dst_src_ltm", "ct_srv_src"}
    cyber_hits = [f for f in features if any(term in f for term in cyber_terms)]
    if cyber_hits:
        return "Top SHAP features include network/traffic indicators: " + ", ".join(features[:6]) + "."
    return "Top SHAP features include process/sensor indicators: " + ", ".join(features[:6]) + "."


def extract_actions(runbook_text: str) -> str:
    lines = []
    capture = False
    for line in runbook_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("## immediate actions"):
            capture = True
            continue
        if capture and stripped.startswith("## "):
            break
        if capture and re.match(r"^\d+\.", stripped):
            lines.append(stripped)
    if not lines:
        return "1. Review model output, validate physical state, and escalate to the responsible team."
    return "\n".join(lines)


def build_report(event: Dict, retrieved: List[Tuple[Runbook, float]]) -> str:
    severity = classify_severity(event)
    team = infer_response_team(event)
    prediction = str(event.get("prediction", "unknown"))
    confidence = float(event.get("confidence", 0.0))
    cdaw = float(event.get("adaptive_cdaw_score", 0.0))
    trust = float(event.get("client_trust", 1.0))
    feature_summary = summarize_features(event.get("top_shap_features", []))
    first_runbook_text = retrieved[0][0].text if retrieved else ""
    action_lines = extract_actions(first_runbook_text)
    citations = "\n".join(
        f"- {rb.path.as_posix()} (retrieval score: {score:.1f})"
        for rb, score in retrieved
    )

    return f"""
# Fed-LLM CPS Incident Report

## Diagnosis

- Asset: {event.get("asset", "unknown")}
- Model prediction: {prediction}
- Confidence: {confidence:.2f}
- Adaptive CDAW score: {cdaw:.2f}
- Client trust score: {trust:.2f}
- Severity: {severity}
- Recommended response owner: {team}

## Reasoning

The detector classified the event as `{prediction}`. {feature_summary}
The Adaptive CDAW score and client trust score are used to decide whether this is routine, suspicious, or uncertain.

## Recommended Actions

{action_lines}

## Stakeholder Update Draft

We are investigating an abnormal Cyber-Physical System event on {event.get("asset", "the affected asset")}. Current model output suggests `{prediction}` with confidence {confidence:.2f}. The event is classified as {severity}. Recommended owner: {team}. The team is following the retrieved runbook steps and will provide an update after validation checks are complete.

## Retrieved Runbook Citations

{citations}
""".strip()


def load_event(path: str | None) -> Dict:
    if path is None:
        return DEFAULT_EVENT
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fed-LLM CPS incident response copilot")
    parser.add_argument("--event", help="Path to event JSON. Uses a demo event if omitted.")
    parser.add_argument("--out", default="incident_report.md", help="Output markdown report path.")
    args = parser.parse_args()

    event = load_event(args.event)
    runbooks = load_runbooks()
    retrieved = retrieve_runbooks(event, runbooks)
    report = build_report(event, retrieved)

    out_path = Path(args.out)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\n[OK] Report saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
