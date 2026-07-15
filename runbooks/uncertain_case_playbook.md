# Uncertain Fault-Vs-Cyber Case Playbook

Use this playbook when model confidence is low, Adaptive CDAW is high, or SHAP features show both cyber and physical indicators.

## Signals

- Model prediction and anomaly score disagree
- SHAP explanation contains both network and process/sensor features
- Similar symptoms could be caused by either real machinery fault or cyber manipulation
- Client trust score is low or the client update appears abnormal

## Immediate Actions

1. Do not assume one cause too early.
2. Run parallel maintenance and cybersecurity checks.
3. Verify sensor readings with independent measurements.
4. Review command history, network logs, and recent configuration changes.
5. Keep the system in a safe operating mode until root cause is confirmed.

## Escalation

Escalate to both maintenance and cybersecurity teams when uncertainty remains high.

