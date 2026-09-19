"""Deterministic scoring for saved M5.5 predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .build_prompts import STABILITY_CASE_IDS, BenchmarkCase, load_cases


STATUSES = {"RESOLVED", "DIVERGED", "AMBIGUOUS", "UNVERIFIED"}
VALUES = {"JWT", "OAuth2", None}


def normalize_source(value: object) -> str:
    return str(value).strip().replace("\\", "/").removeprefix("./")


def validate_prediction(prediction: object) -> dict[str, object]:
    if not isinstance(prediction, dict):
        raise ValueError("prediction must be an object")
    for key in ("current", "target", "status", "evidence"):
        if key not in prediction:
            raise ValueError(f"prediction missing {key}")
    if prediction["current"] not in VALUES or prediction["target"] not in VALUES:
        raise ValueError("current and target must be JWT, OAuth2, or null")
    if prediction["status"] not in STATUSES:
        raise ValueError("invalid prediction status")
    evidence = prediction["evidence"]
    if not isinstance(evidence, dict):
        raise ValueError("prediction evidence must be an object")
    if any(not isinstance(evidence.get(key, []), list) for key in ("current", "target", "conflict")):
        raise ValueError("prediction evidence fields must be arrays")
    normalized = {
        key: [normalize_source(path) for path in evidence.get(key, [])]
        for key in ("current", "target", "conflict")
    }
    return {
        "current": prediction["current"],
        "target": prediction["target"],
        "status": prediction["status"],
        "evidence": normalized,
    }


def _sources(ground_truth: dict[str, object], key: str) -> set[str]:
    values = ground_truth.get(key, [])
    return {normalize_source(value) for value in values} if isinstance(values, list) else set()


def score_case(case: BenchmarkCase, prediction: object) -> dict[str, object]:
    answer = validate_prediction(prediction)
    truth = case.ground_truth
    current = answer["current"] == truth.get("current")
    target = answer["target"] == truth.get("target")
    status = answer["status"] == truth.get("status")
    evidence = answer["evidence"]
    traceable = True
    if current and truth.get("current") is not None:
        traceable &= bool(set(evidence["current"]) & _sources(truth, "valid_current_sources"))
    elif truth.get("current") is not None:
        traceable = False
    if target and truth.get("target") is not None:
        traceable &= bool(set(evidence["target"]) & _sources(truth, "valid_target_sources"))
    elif truth.get("target") is not None:
        traceable = False
    required_conflicts = _sources(truth, "required_conflict_sources")
    traceable &= required_conflicts <= set(evidence["conflict"])
    conflict = truth.get("status") in {"DIVERGED", "AMBIGUOUS"} and status
    return {
        "case_id": case.id,
        "current": int(current),
        "target": int(target),
        "status": int(status),
        "traceability": int(traceable),
        "conflict_detection": int(conflict),
        "is_conflict": int(truth.get("status") in {"DIVERGED", "AMBIGUOUS"}),
        "overall": int(current) + int(target) + int(status) + int(traceable),
    }


def load_predictions(root: Path, condition: str) -> dict[str, object]:
    predictions = {}
    for path in sorted((root / condition).glob("*.json")):
        predictions[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return predictions


def summarize(
    scores: Iterable[dict[str, object]],
    total_cases: int = 15,
    stability: float = 0.0,
) -> dict[str, object]:
    rows = list(scores)
    conflict_cases = sum(row["is_conflict"] for row in rows)
    return {
        "cases_scored": len(rows),
        "current_accuracy": sum(row["current"] for row in rows) / total_cases,
        "target_accuracy": sum(row["target"] for row in rows) / total_cases,
        "status_accuracy": sum(row["status"] for row in rows) / total_cases,
        "traceability_rate": sum(row["traceability"] for row in rows) / total_cases,
        "conflict_detection_rate": sum(row["conflict_detection"] for row in rows) / max(conflict_cases, 1),
        "stability_rate": stability,
        "overall_points": sum(row["overall"] for row in rows),
        "overall_possible": total_cases * 4,
    }


def stability_rate(variant_predictions: dict[str, list[object]]) -> float:
    """Return the fraction of cases stable across their supplied variants."""

    if not variant_predictions:
        return 0.0
    stable = 0
    for predictions in variant_predictions.values():
        tuples = {
            (
                validate_prediction(prediction)["current"],
                validate_prediction(prediction)["target"],
                validate_prediction(prediction)["status"],
            )
            for prediction in predictions
        }
        stable += int(len(tuples) == 1 and len(predictions) >= 3)
    return stable / len(variant_predictions)


def stability_variants(case: BenchmarkCase) -> list[tuple[str, str]]:
    return [(case.id, ordering) for ordering in ("canonical", "reverse", "permutation")]


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Score saved M5.5 predictions")
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    cases = load_cases()
    report = {}
    for condition in ("raw", "contextcanon"):
        predictions = load_predictions(args.results, condition)
        scores = [
            score_case(case, predictions[f"{case.id}__canonical"])
            for case in cases
            if f"{case.id}__canonical" in predictions
        ]
        variants: dict[str, list[object]] = {}
        for case in cases:
            if case.id not in STABILITY_CASE_IDS:
                continue
            variant_predictions = [
                predictions[key]
                for key in (
                    f"{case.id}__canonical",
                    f"{case.id}__reverse",
                    f"{case.id}__permutation",
                )
                if key in predictions
            ]
            if variant_predictions:
                variants[case.id] = variant_predictions
        report[condition] = {
            "cases": scores,
            "summary": summarize(scores, stability=stability_rate(variants)),
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
