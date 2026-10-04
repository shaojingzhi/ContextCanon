"""Opt-in Direct LLM baseline for the frozen ContextCanon V2.1 benchmark."""

from __future__ import annotations

import argparse
from math import ceil
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
from time import monotonic
from typing import Any

if __package__ in {None, ""}:  # Support direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextcanon.semantic import OpenAICompatibleExtractionClient, SemanticInferenceConfig
from experiments.v2_real_benchmark import BENCHMARK, BenchmarkCase, load_cases


ALLOWED_RELATIONS = {"SUPPORTING", "COMPATIBLE", "CONFLICTING", "DIVERGENT", "UNKNOWN"}
ALLOWED_SUMMARIES = {"INCOMPLETE", "UNRESOLVED", "CLEAR"}
CONTEXTCANON_RESULT = Path("/tmp/contextcanon_v2_1_36.jsonl")


def build_prompt(case: BenchmarkCase) -> str:
    evidence = [
        {
            "ref": item.evidence_id,
            "source": item.source_id,
            "location": item.location,
            "content": item.content,
        }
        for item in case.evidence
    ]
    fact_needs = [
        {
            "subject": item.subject,
            "dimension": item.dimension,
            "scope_constraint": item.scope_constraint,
            "value_type": item.value_type,
        }
        for item in case.fact_needs
    ]
    return (
        "Decide governance directly from the supplied Evidence and FactNeeds. "
        "Do not create Claims and do not provide reasoning.\n"
        "Allowed relations: SUPPORTING, COMPATIBLE, CONFLICTING, DIVERGENT, UNKNOWN.\n"
        "Frozen V2 rules for the same semantic subject and predicate:\n"
        "- Definitely non-overlapping scope: COMPATIBLE.\n"
        "- Unknown scope overlap + equal typed value: COMPATIBLE.\n"
        "- Unknown scope overlap + unequal typed value: UNKNOWN.\n"
        "- Overlapping scope + equal typed value: SUPPORTING.\n"
        "- Different values: MULTI + MULTI is COMPATIBLE; UNKNOWN + anything is UNKNOWN; "
        "SINGLE + MULTI or MULTI + SINGLE is UNKNOWN.\n"
        "- Overlapping SINGLE + SINGLE unequal values: OBSERVED versus INTENDED or REQUIRED "
        "is DIVERGENT; if DOCUMENTED participates and its stronger modality cannot be "
        "established, use UNKNOWN; otherwise use CONFLICTING.\n"
        "Modality meanings: OBSERVED is actual/runtime/current state; DOCUMENTED is stated "
        "without clear runtime, requirement, or intent semantics; INTENDED is a planned or "
        "future target; REQUIRED is a requirement, policy, or must constraint.\n"
        "Do not compare different unrelated semantic facts.\n"
        "Summary states:\n"
        "- INCOMPLETE: a required FactNeed has no usable evidence or fact.\n"
        "- UNRESOLVED: a relevant CONFLICTING, DIVERGENT, or UNKNOWN relation exists.\n"
        "- CLEAR: usable facts exist and no unresolved relation exists.\n"
        "Return only JSON with exactly this shape:\n"
        '{"relations":[{"left_evidence_refs":["E1"],'
        '"right_evidence_refs":["E2"],"relation":"CONFLICTING"}],'
        '"summary_state":"UNRESOLVED"}\n'
        "Use only supplied Evidence refs. No reasoning, confidence, Claims, or extra fields.\n"
        f"Evidence: {json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))}\n"
        f"FactNeeds: {json.dumps(fact_needs, ensure_ascii=False, separators=(',', ':'))}"
    )


def _endpoint(left: list[str] | tuple[str, ...], right: list[str] | tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    left_key, right_key = tuple(sorted(left)), tuple(sorted(right))
    return (left_key, right_key) if left_key <= right_key else (right_key, left_key)


def _relation_signatures(relations: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    signatures = []
    for relation in relations:
        left, right = _endpoint(
            relation["left_evidence_refs"], relation["right_evidence_refs"]
        )
        signatures.append((left, right, relation["relation"]))
    return sorted(signatures)


def parse_response(raw_response: str, evidence_refs: set[str]) -> dict[str, Any]:
    try:
        payload = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("response is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"relations", "summary_state"}:
        raise ValueError("response must contain exactly relations and summary_state")
    if payload["summary_state"] not in ALLOWED_SUMMARIES:
        raise ValueError("invalid summary_state")
    if not isinstance(payload["relations"], list):
        raise ValueError("relations must be a list")
    for relation in payload["relations"]:
        if not isinstance(relation, dict) or set(relation) != {
            "left_evidence_refs", "right_evidence_refs", "relation"
        }:
            raise ValueError("invalid relation object")
        left, right = relation["left_evidence_refs"], relation["right_evidence_refs"]
        if not all(isinstance(refs, list) and refs for refs in (left, right)):
            raise ValueError("relation endpoints must be non-empty lists")
        if not all(isinstance(ref, str) for ref in left + right):
            raise ValueError("evidence refs must be strings")
        if not set(left + right) <= evidence_refs:
            raise ValueError("relation references unknown Evidence")
        if relation["relation"] not in ALLOWED_RELATIONS:
            raise ValueError("invalid relation")
    return payload


def _gold_relations(case: BenchmarkCase) -> list[dict[str, Any]]:
    return [
        {
            "left_evidence_refs": list(item.left_evidence_ids),
            "right_evidence_refs": list(item.right_evidence_ids),
            "relation": item.relation,
        }
        for item in case.gold_relations
    ]


def validate_cases(cases: tuple[BenchmarkCase, ...], *, expected_count: int = 36) -> None:
    if len(cases) != expected_count:
        raise ValueError(f"baseline requires exactly {expected_count} cases")
    for case in cases:
        refs = {item.evidence_id for item in case.evidence}
        if case.gold_summary not in ALLOWED_SUMMARIES:
            raise ValueError(f"{case.case_id}: invalid Gold summary")
        for relation in _gold_relations(case):
            if relation["relation"] not in ALLOWED_RELATIONS:
                raise ValueError(f"{case.case_id}: invalid Gold relation")
            if not set(relation["left_evidence_refs"] + relation["right_evidence_refs"]) <= refs:
                raise ValueError(f"{case.case_id}: invalid Gold Evidence ref")


def evaluate_response(
    case: BenchmarkCase,
    raw_response: str | None,
    latency_ms: float | None,
    *,
    request_error: str | None = None,
) -> dict[str, Any]:
    schema_error = request_error
    parsed: dict[str, Any] | None = None
    if schema_error is None:
        try:
            parsed = parse_response(
                raw_response or "", {item.evidence_id for item in case.evidence}
            )
        except ValueError as error:
            schema_error = str(error)
    predicted_relations = parsed["relations"] if parsed else []
    predicted_summary = parsed["summary_state"] if parsed else None
    gold_relations = _gold_relations(case)
    relation_match = schema_error is None and (
        _relation_signatures(predicted_relations) == _relation_signatures(gold_relations)
    )
    summary_match = schema_error is None and predicted_summary == case.gold_summary
    return {
        "case_id": case.case_id,
        "domain": case.domain,
        "raw_model_response": raw_response,
        "predicted_relations": predicted_relations,
        "gold_relations": gold_relations,
        "predicted_summary": predicted_summary,
        "gold_summary": case.gold_summary,
        "relation_exact_match": relation_match,
        "summary_match": summary_match,
        "decision_exact_match": relation_match and summary_match,
        "schema_error": schema_error,
        "latency_ms": latency_ms,
    }


def _rate(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "total": total, "rate": count / total if total else 1.0}


def _latency_metrics(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None}
    ordered = sorted(values)
    return {
        "mean_ms": mean(ordered),
        "p50_ms": median(ordered),
        "p95_ms": ordered[ceil(0.95 * len(ordered)) - 1],
    }


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    latencies = [item["latency_ms"] for item in records if item["latency_ms"] is not None]
    return {
        "case_count": total,
        "relation_accuracy": _rate(sum(item["relation_exact_match"] for item in records), total),
        "summary_accuracy": _rate(sum(item["summary_match"] for item in records), total),
        "decision_exact_accuracy": _rate(sum(item["decision_exact_match"] for item in records), total),
        "schema_success_rate": _rate(sum(item["schema_error"] is None for item in records), total),
        "latency": _latency_metrics(latencies),
    }


def contextcanon_comparison(path: Path = CONTEXTCANON_RESULT) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [row for row in rows if "case_id" in row]
    if not cases:
        return None
    total = len(cases)
    latencies = [item["elapsed_model_call_ms"] for item in cases if item["elapsed_model_call_ms"] is not None]
    return {
        "case_count": total,
        "relation_accuracy": _rate(sum(item["relation_exact_match"] for item in cases), total),
        "summary_accuracy": _rate(sum(item["summary_match"] for item in cases), total),
        "decision_exact_accuracy": _rate(sum(item["governance_exact_match"] for item in cases), total),
        "latency": _latency_metrics(latencies),
    }


def run_live(cases: tuple[BenchmarkCase, ...], client: Any, model: str) -> list[dict[str, Any]]:
    records = []
    for case in cases:
        started = monotonic()
        raw_response = None
        request_error = None
        try:
            raw_response = client.complete(build_prompt(case), model=model).content
        except Exception as error:  # One failed request is one failed baseline case.
            request_error = f"provider request failed: {type(error).__name__}"
        latency_ms = round((monotonic() - started) * 1000, 3)
        record = evaluate_response(
            case, raw_response, latency_ms, request_error=request_error
        )
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))
        records.append(record)
    print(json.dumps({
        "aggregate_metrics": aggregate_metrics(records),
        "contextcanon_comparison": contextcanon_comparison(),
    }, ensure_ascii=False, sort_keys=True))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--expected-count", type=int, default=36)
    args = parser.parse_args(argv)

    cases = load_cases(args.benchmark, expected_count=args.expected_count)
    validate_cases(cases, expected_count=args.expected_count)
    if not args.run_live:
        print(json.dumps({
            "case_count": len(cases),
            "live_execution": False,
            "message": "live execution disabled; zero API calls made",
        }))
        return 0

    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY or OPENAI_API_KEY before --run-live.")
    client = OpenAICompatibleExtractionClient(
        api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
        timeout=60,
        inference=SemanticInferenceConfig(max_output_tokens=1024, thinking=False),
    )
    run_live(cases, client, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
