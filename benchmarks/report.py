"""Human-readable and machine-readable M5.5 comparison reports."""

from __future__ import annotations

import json


def render_report(raw: dict[str, object], contextcanon: dict[str, object]) -> str:
    metrics = (
        ("Current Accuracy", "current_accuracy"),
        ("Target Accuracy", "target_accuracy"),
        ("Status Accuracy", "status_accuracy"),
        ("Traceability Rate", "traceability_rate"),
        ("Conflict Detection", "conflict_detection_rate"),
        ("Stability Rate", "stability_rate"),
    )
    lines = [
        "Metric                  Raw Context    ContextCanon",
        "---------------------------------------------------",
    ]
    for label, key in metrics:
        lines.append(
            f"{label:<24} {raw.get(key, 0):>10.1%} {contextcanon.get(key, 0):>14.1%}"
        )
    lines.append(
        f"Overall                  {raw.get('overall_points', 0)}/{raw.get('overall_possible', 0):<8}"
        f" {contextcanon.get('overall_points', 0)}/{contextcanon.get('overall_possible', 0)}"
    )
    return "\n".join(lines)


def report_json(raw: dict[str, object], contextcanon: dict[str, object]) -> str:
    return json.dumps(
        {"raw": raw, "contextcanon": contextcanon},
        indent=2,
        sort_keys=True,
    )

