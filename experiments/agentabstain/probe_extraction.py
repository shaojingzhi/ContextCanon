"""Run the bounded semantic extractor against persistent runtime observations."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from contextcanon.semantic import (
    LLMStructuredExtractor,
    OpenAICompatibleExtractionClient,
    SemanticInferenceConfig,
)
from contextcanon.semantic_budget import BudgetedSemanticClient, SemanticBudget


FIXTURE = Path(__file__).with_name("fixtures") / "preview_008_extraction_observations.json"


def _claim_payload(claim: object) -> dict[str, object]:
    return {
        "entity": claim.entity,
        "property": claim.property,
        "value": claim.value,
        "value_type": claim.value_type,
        "role": claim.role,
        "confidence": claim.confidence,
        "temporal_scope": claim.temporal_scope,
    }


def main() -> int:
    observations = json.loads(FIXTURE.read_text(encoding="utf-8"))
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("No DEEPSEEK_API_KEY or OPENAI_API_KEY is configured.")
    budget = SemanticBudget(
        max_requests=2,
        max_total_seconds=15,
        per_request_deadline_seconds=8,
    )
    client = OpenAICompatibleExtractionClient(
        api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
        timeout=8,
        inference=SemanticInferenceConfig(max_output_tokens=1024, thinking=False),
    )
    extractor = LLMStructuredExtractor(
        BudgetedSemanticClient(client, budget, "extraction"),
        model="deepseek-v4-pro",
    )
    for raw in observations:
        observation = SimpleNamespace(**raw)
        started = time.monotonic()
        claims = extractor.extract(observation)
        metadata = extractor.last_response_metadata
        diagnostic = extractor.last_diagnostic or {}
        usage = metadata.get("usage") or {}
        print(json.dumps({
            "tool_name": observation.tool_name,
            "call_index": observation.call_index,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "finish_reason": metadata.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "content_length": metadata.get("content_length"),
            "response_content_present": metadata.get("response_content_present"),
            "failure_category": diagnostic.get("category"),
            "exception": diagnostic.get("exception"),
            "detail": diagnostic.get("detail"),
            "response_preview": diagnostic.get("response_preview", metadata.get("response_preview")),
            "accepted_claims": [_claim_payload(claim) for claim in claims],
        }, ensure_ascii=False, sort_keys=True))
    print(json.dumps({"semantic_budget": budget.diagnostics()}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
