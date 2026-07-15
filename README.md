# Fed-LLM CPS Incident Commander

**Fed-LLM CPS Incident Commander** is an end-to-end Cyber-Physical System (CPS) project that combines:

1. **TA-FedX-CPS detector**: trust-aware federated learning for fault/cyber-attack detection.
2. **Fed-LLM response agent**: Agentic RAG-style runbook retrieval and incident response generation.

The detector produces outputs such as prediction, confidence, Adaptive CDAW score, client trust, and SHAP features. The response agent retrieves relevant maintenance/security runbooks and drafts an incident response plan.

## What It Does

- Reads CPS incident context from JSON
- Runs a complete TA-FedX-CPS detector pipeline
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

## End-to-End Architecture

TA-FedX-CPS is the detection engine. Fed-LLM is the response and reasoning layer.

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
|   |-- ta_fedx_cps_detector.py
|   |-- fed_llm_cps_agent.py
|   |-- run_full_pipeline.py
|   `-- show_figures_colab.py
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

## Run the Agent Demo Only

```bash
python src/fed_llm_cps_agent.py --event examples/sample_cps_event.json --out incident_report.md
```

The script works without an LLM API key. It uses deterministic retrieval and report generation so the demo is reproducible.

## Run the Full Detector + Agent Pipeline

Place the UNSW-NB15 files in the project root if you want real results:

```text
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv
```

Then run:

```bash
python src/run_full_pipeline.py
```

This will:

1. Train/evaluate the TA-FedX-CPS detector.
2. Export `fedx_had_outputs/incident_event_for_agent.json`.
3. Run the Fed-LLM incident agent.
4. Generate `fed_llm_incident_report.md`.

If the dataset is missing, the detector runs in synthetic demo mode.

## Show Figures in Google Colab

When you run the full pipeline with `!python`, Colab may save figures but show only text like `Figure(550x450)`.

After the pipeline finishes, run this in a normal Colab code cell:

```python
%run src/show_figures_colab.py
```

It displays:

- TA-FedX-IDS confusion matrix
- Adaptive CDAW confusion matrix
- ROC comparison
- CDAW score distribution
- SHAP explanation plot
- metrics tables

## Future LLM Upgrade

The current implementation is LLM-ready. A future version can replace the final report generator with:

- OpenAI API
- LangChain agent
- local open-source LLM
- vector database retrieval

## Resume Summary

Built Fed-LLM, an end-to-end CPS incident diagnosis system combining trust-aware federated fault/cyber-attack detection with an Agentic RAG-style response copilot that retrieves maintenance/security runbooks, classifies severity, recommends remediation, and drafts stakeholder updates.
