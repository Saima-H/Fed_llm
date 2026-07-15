# Cyber-Attack Response Playbook

Use this playbook when the detector predicts cyber attack, network features dominate the SHAP explanation, or the anomaly score is high with signs of command or traffic manipulation.

## Signals

- Unusual connection counts or repeated source-destination communication
- Abnormal protocol, service, packet, or TCP timing features
- Sudden command changes without matching physical process behavior
- Suspicious traffic spikes, failed authentication, or unexpected remote access

## Immediate Actions

1. Isolate the affected controller, endpoint, or network segment if safety allows.
2. Preserve logs from the controller, firewall, switch, historian, and endpoint.
3. Check recent command history and remote access sessions.
4. Compare sensor readings with physical inspection or redundant sensors.
5. Notify the cybersecurity or security operations team.

## Escalation

Treat as high severity if production safety, controller integrity, or critical process availability may be affected.

