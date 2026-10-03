"""Opt-in 36-case real-model evaluation for ContextCanon V2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import permutations
import json
import os
from pathlib import Path
import sys
from time import monotonic
from typing import Any

import yaml

if __package__ in {None, ""}:  # Support ``python experiments/v2_real_benchmark.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextcanon.semantic import OpenAICompatibleExtractionClient, SemanticInferenceConfig
from contextcanon_v2 import (
    Cardinality,
    Claim,
    Evidence,
    FactNeed,
    GovernanceResult,
    Modality,
    SemanticCompilationError,
    SemanticCompiler,
    govern,
)


BENCHMARK = Path(__file__).parents[1] / "benchmarks" / "v2_design_cases.yaml"
CLAIM_FIELDS = (
    "subject",
    "predicate",
    "value",
    "value_type",
    "scope",
    "modality",
    "cardinality",
    "provenance",
)
CLASSIFICATIONS = (
    "PASS",
    "SEMANTIC_COMPILATION_ERROR",
    "CLAIM_SEMANTICS_ERROR",
    "DETERMINISTIC_GOVERNANCE_ERROR",
    "CLAIM_WRONG_GOVERNANCE_RIGHT",
)


@dataclass(frozen=True, slots=True)
class GoldRelation:
    left_evidence_ids: tuple[str, ...]
    right_evidence_ids: tuple[str, ...]
    relation: str


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    domain: str
    evidence: tuple[Evidence, ...]
    fact_needs: tuple[FactNeed, ...]
    gold_claims: tuple[Claim, ...]
    gold_relations: tuple[GoldRelation, ...]
    gold_summary: str


def load_cases(path: Path = BENCHMARK) -> tuple[BenchmarkCase, ...]:
    raw_cases = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise ValueError("benchmark must contain a case list")
    cases: list[BenchmarkCase] = []
    for raw in raw_cases:
        evidence = tuple(
            Evidence(
                item["evidence_id"],
                item["source_id"],
                item["location"],
                str(item["content"]),
                item.get("observed_at"),
                item.get("source_revision"),
            )
            for item in raw["evidence"]
        )
        needs = tuple(
            FactNeed(
                item["subject"],
                item["dimension"],
                item.get("scope_constraint"),
                item.get("value_type"),
            )
            for item in raw["fact_needs"]
        )
        claims = tuple(
            Claim(
                item["subject"],
                item["predicate"],
                item["value"],
                item["value_type"],
                dict(item.get("scope") or {}),
                Modality(item["modality"]),
                Cardinality(item["cardinality"]),
                tuple(item["evidence_ids"]),
            )
            for item in raw["gold_claims"]
        )
        relations = tuple(
            GoldRelation(
                tuple(item["left_evidence_ids"]),
                tuple(item["right_evidence_ids"]),
                item["relation"],
            )
            for item in raw["gold_relations"]
        )
        evidence_ids = {item.evidence_id for item in evidence}
        referenced = {
            evidence_id
            for claim in claims
            for evidence_id in claim.evidence_ids
        } | {
            evidence_id
            for relation in relations
            for evidence_id in relation.left_evidence_ids + relation.right_evidence_ids
        }
        if not referenced <= evidence_ids:
            raise ValueError(f"{raw['case_id']}: Gold provenance references missing Evidence")
        cases.append(BenchmarkCase(
            raw["case_id"], raw["domain"], evidence, needs, claims, relations, raw["gold_summary"]
        ))
    if len(cases) != 36 or len({case.case_id for case in cases}) != 36:
        raise ValueError("benchmark must contain 36 unique cases")
    return tuple(cases)


def _claim_record(claim: Claim | None) -> dict[str, Any] | None:
    if claim is None:
        return None
    return {
        "subject": claim.subject,
        "predicate": claim.predicate,
        "value": claim.value,
        "value_type": claim.value_type,
        "scope": claim.scope,
        "modality": claim.modality.value,
        "cardinality": claim.cardinality.value,
        "evidence_ids": list(claim.evidence_ids),
    }


def _claim_field_matches(gold: Claim | None, predicted: Claim | None) -> dict[str, bool]:
    if gold is None or predicted is None:
        return {field: False for field in CLAIM_FIELDS}
    return {
        "subject": predicted.subject == gold.subject,
        "predicate": predicted.predicate == gold.predicate,
        "value": predicted.value == gold.value,
        "value_type": predicted.value_type == gold.value_type,
        "scope": predicted.scope == gold.scope,
        "modality": predicted.modality == gold.modality,
        "cardinality": predicted.cardinality == gold.cardinality,
        "provenance": sorted(predicted.evidence_ids) == sorted(gold.evidence_ids),
    }


def _claim_diffs(gold_claims: tuple[Claim, ...], predicted_claims: tuple[Claim, ...]) -> list[dict[str, Any]]:
    size = max(len(gold_claims), len(predicted_claims))
    if size == 0:
        return []
    padded_gold: tuple[Claim | None, ...] = gold_claims + (None,) * (size - len(gold_claims))
    padded_predicted: tuple[Claim | None, ...] = predicted_claims + (None,) * (size - len(predicted_claims))
    best: tuple[Claim | None, ...] | None = None
    best_score = -1
    for candidate in permutations(padded_predicted):
        score = sum(
            sum(_claim_field_matches(gold, predicted).values())
            for gold, predicted in zip(padded_gold, candidate, strict=True)
        )
        if score > best_score:
            best, best_score = candidate, score
    assert best is not None
    diffs: list[dict[str, Any]] = []
    for gold, predicted in zip(padded_gold, best, strict=True):
        matches = _claim_field_matches(gold, predicted)
        mismatches = [field for field, matches_field in matches.items() if not matches_field]
        diffs.append({
            "gold_claim": _claim_record(gold),
            "predicted_claim": _claim_record(predicted),
            "claim_exact_match": not mismatches,
            **{f"{field}_match": matches[field] for field in CLAIM_FIELDS},
            "mismatched_fields": mismatches,
        })
    return diffs


def _endpoint(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    left, right = tuple(sorted(left)), tuple(sorted(right))
    return (left, right) if left <= right else (right, left)


def _predicted_relations(result: GovernanceResult) -> list[dict[str, Any]]:
    return [
        {
            "left_evidence_ids": list(record.left_claim.evidence_ids),
            "right_evidence_ids": list(record.right_claim.evidence_ids),
            "relation": record.relation.value,
        }
        for record in result.relations
    ]


def _relation_signatures(relations: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    values = []
    for relation in relations:
        left, right = _endpoint(
            tuple(relation["left_evidence_ids"]), tuple(relation["right_evidence_ids"])
        )
        values.append((left, right, relation["relation"]))
    return sorted(values)


def _gold_relation_records(case: BenchmarkCase) -> list[dict[str, Any]]:
    return [
        {
            "left_evidence_ids": list(relation.left_evidence_ids),
            "right_evidence_ids": list(relation.right_evidence_ids),
            "relation": relation.relation,
        }
        for relation in case.gold_relations
    ]


def evaluate_case(
    case: BenchmarkCase,
    predicted_claims: tuple[Claim, ...],
    *,
    governance_result: GovernanceResult | None = None,
    raw_model_response: str | None = None,
    semantic_compilation_error: str | None = None,
    elapsed_model_call_ms: float | None = None,
) -> dict[str, Any]:
    result = governance_result or govern(
        predicted_claims,
        case.fact_needs,
        compilation_error=semantic_compilation_error is not None,
    )
    diffs = _claim_diffs(case.gold_claims, predicted_claims)
    claim_exact = len(case.gold_claims) == len(predicted_claims) and all(
        diff["claim_exact_match"] for diff in diffs
    )
    field_matches = {
        field: all(diff[f"{field}_match"] for diff in diffs)
        for field in CLAIM_FIELDS
    }
    predicted_relations = _predicted_relations(result)
    gold_relations = _gold_relation_records(case)
    relation_exact = _relation_signatures(predicted_relations) == _relation_signatures(gold_relations)
    summary_match = result.summary_state.value == case.gold_summary
    governance_exact = relation_exact and summary_match

    if semantic_compilation_error is not None or (case.gold_claims and not predicted_claims):
        classification = "SEMANTIC_COMPILATION_ERROR"
    elif not claim_exact:
        classification = "CLAIM_WRONG_GOVERNANCE_RIGHT" if governance_exact else "CLAIM_SEMANTICS_ERROR"
    elif not governance_exact:
        classification = "DETERMINISTIC_GOVERNANCE_ERROR"
    else:
        classification = "PASS"

    return {
        "case_id": case.case_id,
        "domain": case.domain,
        "Evidence": [
            {
                "evidence_id": item.evidence_id,
                "source_id": item.source_id,
                "location": item.location,
                "content": item.content,
            }
            for item in case.evidence
        ],
        "FactNeeds": [
            {
                "subject": need.subject,
                "dimension": need.dimension,
                "scope_constraint": need.scope_constraint,
                "value_type": need.value_type,
            }
            for need in case.fact_needs
        ],
        "raw_model_response": raw_model_response,
        "predicted_claims": [_claim_record(claim) for claim in predicted_claims],
        "gold_claims": [_claim_record(claim) for claim in case.gold_claims],
        "claim_diffs": diffs,
        "claim_exact_match": claim_exact,
        **{f"{field}_match": field_matches[field] for field in CLAIM_FIELDS},
        "gold_claim_count": len(case.gold_claims),
        "predicted_claim_count": len(predicted_claims),
        "predicted_relations": predicted_relations,
        "gold_relations": gold_relations,
        "relation_exact_match": relation_exact,
        "predicted_summary": result.summary_state.value,
        "gold_summary": case.gold_summary,
        "summary_match": summary_match,
        "governance_exact_match": governance_exact,
        "failure_classification": classification,
        "semantic_compilation_error": semantic_compilation_error,
        "elapsed_model_call_ms": elapsed_model_call_ms,
    }


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)

    def metric(numerator: int, denominator: int) -> dict[str, Any]:
        return {
            "count": numerator,
            "total": denominator,
            "rate": numerator / denominator if denominator else 1.0,
        }

    claim_diffs = [diff for record in records for diff in record["claim_diffs"]]
    metrics = {
        "case_count": count,
        "case_claim_exact_accuracy": metric(sum(record["claim_exact_match"] for record in records), count),
        "relation_accuracy": metric(sum(record["relation_exact_match"] for record in records), count),
        "summary_accuracy": metric(sum(record["summary_match"] for record in records), count),
        "full_end_to_end_exact_accuracy": metric(
            sum(record["claim_exact_match"] and record["governance_exact_match"] for record in records), count
        ),
        "classification_counts": {
            classification: sum(record["failure_classification"] == classification for record in records)
            for classification in CLASSIFICATIONS
        },
    }
    for field in CLAIM_FIELDS:
        metrics[f"{field}_accuracy"] = metric(
            sum(diff[f"{field}_match"] for diff in claim_diffs), len(claim_diffs)
        )
    return metrics


def _run_live(cases: tuple[BenchmarkCase, ...], client: Any, model: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        raw_response: str | None = None
        elapsed_ms: float | None = None

        def complete(prompt: str) -> str:
            nonlocal raw_response, elapsed_ms
            started = monotonic()
            try:
                completion = client.complete(prompt, model=model)
                raw_response = completion.content
                return raw_response
            finally:
                elapsed_ms = round((monotonic() - started) * 1000, 3)

        error: str | None = None
        try:
            claims = SemanticCompiler(complete).compile(case.evidence, case.fact_needs)
        except SemanticCompilationError as compilation_error:
            claims = ()
            error = str(compilation_error)
        record = evaluate_case(
            case,
            claims,
            raw_model_response=raw_response,
            semantic_compilation_error=error,
            elapsed_model_call_ms=elapsed_ms,
        )
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))
        records.append(record)
    print(json.dumps({"aggregate_metrics": aggregate_metrics(records)}, ensure_ascii=False, sort_keys=True))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument("--model", default="deepseek-v4-pro")
    args = parser.parse_args(argv)

    cases = load_cases()
    self_check = [evaluate_case(case, case.gold_claims) for case in cases]
    pass_count = sum(record["failure_classification"] == "PASS" for record in self_check)
    if pass_count != 36:
        raise SystemExit(f"Gold self-check failed: {pass_count}/36")
    if not args.run_live:
        print(json.dumps({
            "case_count": len(cases),
            "gold_self_check_passes": pass_count,
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
    _run_live(cases, client, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
