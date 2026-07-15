# Fed-LLM CPS Incident Commander

**Fed-LLM CPS Incident Commander** is an Agentic RAG-style incident-response copilot for Cyber-Physical Systems (CPS).

It is designed to sit on top of a federated fault/cyber-attack detector such as TA-FedX-CPS. The detector produces model outputs such as prediction, confidence, Adaptive CDAW score, client trust, and SHAP features. This copilot retrieves relevant maintenance/security runbooks and drafts an incident response plan.

## What It Does

- Reads CPS incident context from JSON
- Uses model outputs such as:
  - prediction
  - confidence
  - Adaptive Consensus-Driven Anomaly Weighting score
  - client trust score
  - SHAP feature explanations
- Retrieves relevant runbooks using a local RAG-style search
- Classifies severity such as `SEV-1`, `SEV-2`, or `SEV-3`
- Recommends whether maintenance, cybersecurity, or both teams should respond
- Drafts a stakeholder update with runbook citations

## Why This Is Separate From TA-FedX-CPS

TA-FedX-CPS is the detection engine.

Fed-LLM is the response and reasoning layer.

```text
CPS sensor/network data
        |
TA-FedX-CPS detector
        |
prediction + confidence + CDAW + SHAP
        |
Fed-LLM Agentic RAG copilot
        |
severity + diagnosis + runbook-backed action plan
```

This keeps the machine learning project clean while showing a modern LLM/RAG incident-response extension.

## Project Structure

```text
.
|-- src/
|   `-- fed_llm_cps_agent.py
|-- runbooks/
|   |-- cyber_attack_playbook.md
|   |-- machinery_fault_playbook.md
|   `-- uncertain_case_playbook.md
|-- examples/
|   `-- sample_cps_event.json
|-- requirements.txt
|-- README.md
|-- LICENSE
`-- .gitignore
```

## Run Locally

```bash
python src/fed_llm_cps_agent.py --event examples/sample_cps_event.json --out incident_report.md
```

The script works without an LLM API key. It uses deterministic retrieval and report generation so the demo is reproducible.

## Future LLM Upgrade

The current implementation is LLM-ready. A future version can replace the final report generator with:

- OpenAI API
- LangChain agent
- local open-source LLM
- vector database retrieval

## Resume Summary

Built Fed-LLM, an Agentic RAG-style CPS incident-response copilot that consumes federated model outputs, Adaptive CDAW anomaly scores, and SHAP explanations to retrieve maintenance/security runbooks, classify severity, recommend remediation, and draft stakeholder updates.

