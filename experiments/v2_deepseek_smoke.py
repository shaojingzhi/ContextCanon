"""Opt-in five-case DeepSeek smoke harness for the V2 semantic compiler."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from time import monotonic
from typing import Any

if __package__ in {None, ""}:  # Support ``python experiments/v2_deepseek_smoke.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextcanon.semantic import OpenAICompatibleExtractionClient, SemanticInferenceConfig
from contextcanon_v2 import (
    Evidence,
    FactNeed,
    SemanticCompilationError,
    SemanticCompiler,
    govern,
)


@dataclass(frozen=True, slots=True)
class SmokeCase:
    name: str
    evidence: tuple[Evidence, ...]
    fact_needs: tuple[FactNeed, ...]


def _evidence(evidence_id: str, source_id: str, location: str, content: str) -> Evidence:
    return Evidence(evidence_id, source_id, location, content)


def build_cases() -> tuple[SmokeCase, ...]:
    """Build the five small, offline-validatable smoke inputs."""

    return (
        SmokeCase(
            "scalar_current_conflict",
            (
                _evidence("A1", "runtime-config", "auth.protocol", "current runtime protocol is JWT"),
                _evidence("A2", "runtime-config-secondary", "auth.protocol", "current runtime protocol is OAuth2"),
            ),
            (FactNeed("auth", "protocol", value_type="enum"),),
        ),
        SmokeCase(
            "current_future_compatible",
            (
                _evidence("B1", "runtime", "auth.protocol", "current protocol is JWT"),
                _evidence("B2", "adr", "auth.protocol", "future intended protocol is OAuth2"),
            ),
            (FactNeed("auth", "protocol", value_type="enum"),),
        ),
        SmokeCase(
            "observed_required_divergence",
            (
                _evidence("C1", "runtime", "auth.protocol", "current protocol is JWT"),
                _evidence("C2", "policy", "auth.protocol", "required current protocol is OAuth2"),
            ),
            (FactNeed("auth", "protocol", value_type="enum"),),
        ),
        SmokeCase(
            "multi_supported_regions",
            (
                _evidence("D1", "deployment-a", "supported_regions", "US supported"),
                _evidence("D2", "deployment-b", "supported_regions", "EU supported"),
            ),
            (FactNeed("product", "supported_region", value_type="enum"),),
        ),
        SmokeCase(
            "semantic_normalization",
            (
                _evidence("E1", "config", "auth.method", "The authentication method is JWT"),
            ),
            (FactNeed("auth", "protocol", value_type="enum"),),
        ),
    )


def _claim_record(claim: Any) -> dict[str, Any]:
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


def _case_record(case: SmokeCase) -> dict[str, Any]:
    return {
        "case_name": case.name,
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
                "subject": item.subject,
                "dimension": item.dimension,
                "scope_constraint": item.scope_constraint,
                "value_type": item.value_type,
            }
            for item in case.fact_needs
        ],
    }


def _run_live(cases: tuple[SmokeCase, ...], client: Any, model: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        raw_response: str | None = None
        elapsed_model_call_ms: float | None = None

        def complete(prompt: str) -> str:
            nonlocal raw_response, elapsed_model_call_ms
            started = monotonic()
            try:
                completion = client.complete(prompt, model=model)
                raw_response = completion.content
                return raw_response
            finally:
                elapsed_model_call_ms = round((monotonic() - started) * 1000, 3)

        record = _case_record(case)
        record.update({
            "raw_model_response": None,
            "parsed_claims": [],
            "governance_relations": [],
            "summary_state": None,
            "semantic_compilation_error": None,
            "elapsed_model_call_ms": None,
        })
        try:
            claims = SemanticCompiler(complete).compile(case.evidence, case.fact_needs)
            result = govern(claims, case.fact_needs)
            record["parsed_claims"] = [_claim_record(claim) for claim in claims]
            record["governance_relations"] = [
                {
                    "left_evidence_ids": list(relation.left_claim.evidence_ids),
                    "right_evidence_ids": list(relation.right_claim.evidence_ids),
                    "relation": relation.relation.value,
                }
                for relation in result.relations
            ]
            record["summary_state"] = result.summary_state.value
        except SemanticCompilationError as error:
            record["semantic_compilation_error"] = str(error)
            failed_result = govern((), case.fact_needs, compilation_error=True)
            record["summary_state"] = failed_result.summary_state.value
        record["raw_model_response"] = raw_response
        record["elapsed_model_call_ms"] = elapsed_model_call_ms
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))
        records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true", help="allow exactly one provider call per smoke case")
    parser.add_argument("--model", default="deepseek-v4-pro")
    args = parser.parse_args(argv)

    cases = build_cases()
    if len(cases) != 5 or any(not case.evidence or not case.fact_needs for case in cases):
        raise SystemExit("invalid smoke case configuration")
    if not args.run_live:
        print(json.dumps({"case_count": len(cases), "live_execution": False, "message": "live execution disabled; zero API calls made"}))
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
