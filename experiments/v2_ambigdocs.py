"""Frozen AmbigDocs adapter, scorer, and opt-in live validation runner."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
from statistics import mean, median
import sys
from time import monotonic
from collections.abc import Mapping, Sequence
from typing import Any

if __package__ in {None, ""}:  # Support ``python experiments/v2_ambigdocs.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextcanon.semantic import OpenAICompatibleExtractionClient, SemanticInferenceConfig
from contextcanon_v2 import Evidence, FactNeed, SemanticCompiler


DEFAULT_DATASET = Path("/tmp/ambigdocs/test.json")
MANIFEST = Path(__file__).parents[1] / "benchmarks" / "ambigdocs_test_sample_100.json"
RESULTS_DIR = Path(__file__).parents[1] / "benchmarks" / "results"
CONTEXTCANON_RESULTS = RESULTS_DIR / "ambigdocs_contextcanon_100.jsonl"
DIRECT_RESULTS = RESULTS_DIR / "ambigdocs_direct_100.jsonl"
AGGREGATE_RESULTS = RESULTS_DIR / "ambigdocs_100_aggregate.json"
MODEL = "deepseek-flash"
TIMEOUT_SECONDS = 60.0
MAX_OUTPUT_TOKENS = 2048


@dataclass(frozen=True, slots=True)
class GoldAssociation:
    entity: str
    answer: str
    evidence_id: str
    pid: str


@dataclass(frozen=True, slots=True)
class AmbigCase:
    qid: int
    ambiguous_entity: str
    question: str
    evidence: tuple[Evidence, ...]
    fact_needs: tuple[FactNeed, ...]
    gold: tuple[GoldAssociation, ...]


def _validate_row(row: Mapping[str, Any]) -> None:
    if set(row) != {"qid", "ambiguous_entity", "question", "documents"}:
        raise ValueError("unexpected AmbigDocs row fields")
    if not isinstance(row["qid"], int) or not isinstance(row["ambiguous_entity"], str):
        raise ValueError("invalid AmbigDocs row identity")
    documents = row["documents"]
    if not isinstance(documents, list) or not documents:
        raise ValueError("AmbigDocs documents must be a non-empty list")
    for document in documents:
        if set(document) != {"pid", "title", "text", "answer"}:
            raise ValueError("unexpected AmbigDocs document fields")
        if any(not isinstance(document[field], str) for field in ("pid", "title", "text", "answer")):
            raise ValueError("AmbigDocs document fields must be strings")


def map_row(row: Mapping[str, Any]) -> AmbigCase:
    """Map one public row without adding unsupported modality/scope Gold."""
    _validate_row(row)
    evidence: list[Evidence] = []
    needs: list[FactNeed] = []
    gold: list[GoldAssociation] = []
    for index, document in enumerate(row["documents"], start=1):
        evidence_id = f"E{index}"
        title = document["title"]
        evidence.append(
            Evidence(
                evidence_id=evidence_id,
                source_id=f"ambigdocs:{row['qid']}:{document['pid']}",
                location=title,
                content=document["text"],
            )
        )
        needs.append(FactNeed(subject=title, dimension=row["question"], value_type="string"))
        gold.append(
            GoldAssociation(
                entity=title,
                answer=document["answer"],
                evidence_id=evidence_id,
                pid=document["pid"],
            )
        )
    return AmbigCase(
        qid=row["qid"],
        ambiguous_entity=row["ambiguous_entity"],
        question=row["question"],
        evidence=tuple(evidence),
        fact_needs=tuple(needs),
        gold=tuple(gold),
    )


def load_cases(path: Path = DEFAULT_DATASET) -> tuple[AmbigCase, ...]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("AmbigDocs split must be a JSON list")
    cases = tuple(map_row(row) for row in rows)
    if len({case.qid for case in cases}) != len(cases):
        raise ValueError("AmbigDocs qids must be unique")
    return cases


def sample_qids(qids: Sequence[int], *, size: int = 100, seed: int = 42) -> tuple[int, ...]:
    ordered = sorted(qids)
    if size > len(ordered):
        raise ValueError("sample size exceeds the split")
    return tuple(sorted(random.Random(seed).sample(ordered, size)))


def load_sample_manifest(path: Path = MANIFEST) -> tuple[int, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "test" or payload.get("seed") != 42:
        raise ValueError("unexpected AmbigDocs sample manifest")
    qids = tuple(payload["qids"])
    if len(qids) != 100 or len(set(qids)) != 100:
        raise ValueError("AmbigDocs manifest must contain 100 unique qids")
    return qids


def direct_prompt(case: AmbigCase) -> str:
    """Build the future one-call Direct baseline input; no Gold answers included."""
    evidence = [
        {"ref": item.evidence_id, "source": item.source_id, "location": item.location, "content": item.content}
        for item in case.evidence
    ]
    targets = [
        {"subject": need.subject, "dimension": need.dimension, "value_type": need.value_type}
        for need in case.fact_needs
    ]
    return "\n".join(
        (
            "Associate each supplied disambiguated entity with its answer and supporting evidence refs.",
            "Return only JSON matching this schema:",
            '{"associations":[{"entity":"...","answer":"...","evidence_ref":"E1"}]}',
            f"Question: {json.dumps(case.question, ensure_ascii=False)}",
            f"FactNeeds: {json.dumps(targets, ensure_ascii=False, sort_keys=True)}",
            f"Evidence: {json.dumps(evidence, ensure_ascii=False, sort_keys=True)}",
        )
    )


def normalize_answer(value: str) -> str:
    """Apply fixed AmbigDocs-style surface normalization before scoring."""
    value = re.sub(r"[^\w\s]", " ", value.casefold(), flags=re.UNICODE)
    tokens = [token for token in value.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def _token_scores(predicted: str, gold: str) -> tuple[float, float]:
    predicted_tokens = set(normalize_answer(predicted).split())
    gold_tokens = set(normalize_answer(gold).split())
    overlap = len(predicted_tokens & gold_tokens)
    recall = overlap / len(gold_tokens) if gold_tokens else float(not predicted_tokens)
    precision = overlap / len(predicted_tokens) if predicted_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return recall, f1


def _pid_for_ref(case: AmbigCase, evidence_ref: str) -> str:
    for item in case.evidence:
        if item.evidence_id == evidence_ref:
            return item.source_id.rsplit(":", 1)[-1]
    raise ValueError(f"unknown Evidence ref: {evidence_ref}")


def project_contextcanon_claims(case: AmbigCase, claims: Sequence[Any]) -> tuple[dict[str, str], ...]:
    """Project Claim objects into canonical entity/answer/pid associations."""
    projected: list[dict[str, str]] = []
    for claim in claims:
        for evidence_id in claim.evidence_ids:
            projected.append(
                {
                    "entity": claim.subject,
                    "answer": str(claim.value),
                    "pid": _pid_for_ref(case, evidence_id),
                }
            )
    return tuple(projected)


def project_direct_associations(case: AmbigCase, associations: Sequence[Mapping[str, str]]) -> tuple[dict[str, str], ...]:
    """Resolve Direct's compact Evidence refs into canonical pid associations."""
    return tuple(
        {
            "entity": item["entity"],
            "answer": item["answer"],
            "pid": _pid_for_ref(case, item["evidence_ref"]),
        }
        for item in associations
    )


def parse_direct_response(raw_response: str, case: AmbigCase) -> tuple[dict[str, str], ...]:
    payload = json.loads(raw_response)
    if not isinstance(payload, dict) or set(payload) != {"associations"}:
        raise ValueError("Direct response must contain only associations")
    associations = payload["associations"]
    if not isinstance(associations, list):
        raise ValueError("associations must be a list")
    for item in associations:
        if not isinstance(item, dict) or set(item) != {"entity", "answer", "evidence_ref"}:
            raise ValueError("association fields do not match the frozen schema")
    return project_direct_associations(case, associations)


def score_predictions(cases: Sequence[AmbigCase], predictions: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Score canonical associations with completed-only and end-to-end views."""
    if set(predictions) != {case.qid for case in cases}:
        raise ValueError("predictions must contain exactly the selected qids")
    def rate(numerator: int, denominator: int) -> dict[str, int | float]:
        return {"count": numerator, "total": denominator, "rate": numerator / denominator if denominator else 0.0}

    def association_metrics(selected_cases: Sequence[AmbigCase]) -> dict[str, Any]:
        gold_pairs = {(target.entity, target.pid) for case in selected_cases for target in case.gold}
        gold_triples = {
            (target.entity, normalize_answer(target.answer), target.pid)
            for case in selected_cases for target in case.gold
        }
        predicted_pairs: set[tuple[str, str]] = set()
        predicted_triples: set[tuple[str, str, str]] = set()
        token_recalls: list[float] = []
        token_f1s: list[float] = []
        for case in selected_cases:
            for association in predictions[case.qid].get("associations", ()):
                entity, answer, pid = association["entity"], association["answer"], association["pid"]
                predicted_pairs.add((entity, pid))
                predicted_triples.add((entity, normalize_answer(answer), pid))
            for target in case.gold:
                candidates = [
                    item["answer"]
                    for item in predictions[case.qid].get("associations", ())
                    if item["entity"] == target.entity and item["pid"] == target.pid
                ]
                recall, f1 = max((_token_scores(value, target.answer) for value in candidates), default=(0.0, 0.0))
                token_recalls.append(recall)
                token_f1s.append(f1)
        return {
            "case_count": len(selected_cases),
            "entity_label_binding_accuracy": rate(len(predicted_pairs & gold_pairs), len(gold_pairs)),
            "provenance_attribution_accuracy": rate(len(predicted_pairs & gold_pairs), len(predicted_pairs)),
            "normalized_exact_answer_association": rate(len(predicted_triples & gold_triples), len(gold_triples)),
            "answer_token_recall": rate(round(sum(token_recalls), 12), len(token_recalls)),
            "answer_token_f1": rate(round(sum(token_f1s), 12), len(token_f1s)),
        }

    completed = tuple(case for case in cases if predictions[case.qid].get("success"))
    return {
        "case_count": len(cases),
        "schema_success_rate": rate(len(completed), len(cases)),
        "completed_only_metrics": association_metrics(completed),
        "end_to_end_metrics": association_metrics(cases),
    }


def selected_cases(cases: Sequence[AmbigCase]) -> tuple[AmbigCase, ...]:
    """Select the frozen sample in manifest qid order."""
    by_qid = {case.qid: case for case in cases}
    qids = load_sample_manifest()
    if any(qid not in by_qid for qid in qids):
        raise ValueError("AmbigDocs manifest contains a qid absent from the dataset")
    return tuple(by_qid[qid] for qid in qids)


def build_client(api_key: str, base_url: str) -> OpenAICompatibleExtractionClient:
    """Build the frozen non-thinking DeepSeek client without making a request."""
    return OpenAICompatibleExtractionClient(
        api_key,
        base_url=base_url,
        timeout=TIMEOUT_SECONDS,
        inference=SemanticInferenceConfig(max_output_tokens=MAX_OUTPUT_TOKENS, thinking=False),
    )


def _error_record(error: Exception) -> tuple[str, str]:
    root = error.__cause__ or error
    return type(root).__name__, str(error)


def _contextcanon_attempt(case: AmbigCase, client: Any) -> dict[str, Any]:
    raw_response: str | None = None
    usage: Any = None
    started = monotonic()

    def complete(prompt: str) -> str:
        nonlocal raw_response, usage
        completion = client.complete(prompt, model=MODEL)
        raw_response = completion.content
        usage = completion.usage
        return raw_response

    try:
        claims = SemanticCompiler(complete).compile(case.evidence, case.fact_needs)
        associations = project_contextcanon_claims(case, claims)
        success = True
        error_category = None
        error = None
    except Exception as caught:
        associations = ()
        success = False
        error_category, error = _error_record(caught)
    return {
        "qid": case.qid,
        "system": "contextcanon",
        "model": MODEL,
        "success": success,
        "latency_ms": round((monotonic() - started) * 1000, 3),
        "usage": usage,
        "raw_model_response": raw_response,
        "associations": list(associations),
        "error_category": error_category,
        "error": error,
    }


def _direct_attempt(case: AmbigCase, client: Any) -> dict[str, Any]:
    raw_response: str | None = None
    usage: Any = None
    started = monotonic()
    try:
        completion = client.complete(direct_prompt(case), model=MODEL)
        raw_response = completion.content
        usage = completion.usage
        associations = parse_direct_response(raw_response, case)
        success = True
        error_category = None
        error = None
    except Exception as caught:
        associations = ()
        success = False
        error_category, error = _error_record(caught)
    return {
        "qid": case.qid,
        "system": "direct",
        "model": MODEL,
        "success": success,
        "latency_ms": round((monotonic() - started) * 1000, 3),
        "usage": usage,
        "raw_model_response": raw_response,
        "associations": list(associations),
        "error_category": error_category,
        "error": error,
    }


def _load_records(path: Path, *, system: str, allowed_qids: set[int]) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        qid = record.get("qid")
        if record.get("system") != system or record.get("model") != MODEL or qid not in allowed_qids:
            raise ValueError(f"unexpected record in {path}")
        if qid in records:
            raise ValueError(f"duplicate recorded attempt for qid {qid} in {path}")
        records[qid] = record
    return records


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int((percentile * len(ordered) + 0.999999999)) - 1))
    return ordered[index]


def _usage_totals(records: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    totals: dict[str, int | None] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    seen = {key: False for key in totals}
    for record in records:
        usage = record.get("usage")
        if not isinstance(usage, Mapping):
            continue
        values = {
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
            "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
            "total_tokens": usage.get("total_tokens"),
        }
        for key, value in values.items():
            if isinstance(value, int):
                totals[key] = int(totals[key] or 0) + value
                seen[key] = True
    return {key: value if seen[key] else None for key, value in totals.items()}


def _system_aggregate(cases: Sequence[AmbigCase], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    predictions = {
        record["qid"]: {
            "success": record["success"],
            "associations": record["associations"],
        }
        for record in records
    }
    metrics = score_predictions(cases, predictions)
    latencies = [float(record["latency_ms"]) for record in records]
    errors = Counter(record["error_category"] for record in records if record.get("error_category"))
    return {
        "attempted_cases": len(records),
        "successful_cases": sum(bool(record["success"]) for record in records),
        "failed_cases": sum(not bool(record["success"]) for record in records),
        "provider_calls": len(records),
        "metrics": metrics,
        "usage": _usage_totals(records),
        "latency_ms": {
            "mean": round(mean(latencies), 3) if latencies else 0.0,
            "p50": round(median(latencies), 3) if latencies else 0.0,
            "p95": round(_nearest_rank(latencies, 0.95), 3),
        },
        "error_category_counts": dict(sorted(errors.items())),
    }


def run_live(
    cases: Sequence[AmbigCase],
    client: Any,
    *,
    contextcanon_path: Path = CONTEXTCANON_RESULTS,
    direct_path: Path = DIRECT_RESULTS,
    aggregate_path: Path = AGGREGATE_RESULTS,
) -> dict[str, Any]:
    """Run or safely resume the frozen, interleaved, one-call evaluation."""
    cases = tuple(cases)
    allowed_qids = {case.qid for case in cases}
    if len(allowed_qids) != len(cases):
        raise ValueError("selected AmbigDocs qids must be unique")
    contextcanon = _load_records(contextcanon_path, system="contextcanon", allowed_qids=allowed_qids)
    direct = _load_records(direct_path, system="direct", allowed_qids=allowed_qids)
    if aggregate_path.exists() or (len(contextcanon) == len(cases) and len(direct) == len(cases)):
        raise FileExistsError("completed AmbigDocs official run already exists")

    paths = {"contextcanon": contextcanon_path, "direct": direct_path}
    records = {"contextcanon": contextcanon, "direct": direct}
    attempts = {"contextcanon": _contextcanon_attempt, "direct": _direct_attempt}
    for index, case in enumerate(cases):
        order = ("contextcanon", "direct") if index % 2 == 0 else ("direct", "contextcanon")
        for system in order:
            if case.qid in records[system]:
                continue
            record = attempts[system](case, client)
            _append_record(paths[system], record)
            records[system][case.qid] = record
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

    if any(len(records[system]) != len(cases) for system in records):
        raise RuntimeError("AmbigDocs run ended without all attempts recorded")
    ordered_records = {
        system: [records[system][case.qid] for case in cases]
        for system in ("contextcanon", "direct")
    }
    aggregate = {
        "dataset": "yoonsanglee/AmbigDocs",
        "split": "test",
        "sample_count": len(cases),
        "model": MODEL,
        "provider": "official DeepSeek OpenAI-compatible endpoint",
        "thinking": False,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "total_provider_calls": sum(len(items) for items in ordered_records.values()),
        "systems": {
            system: _system_aggregate(cases, ordered_records[system])
            for system in ("contextcanon", "direct")
        },
    }
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = aggregate_path.with_suffix(aggregate_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(aggregate_path)
    print(json.dumps({"aggregate": aggregate}, ensure_ascii=False, sort_keys=True), flush=True)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args(argv)

    cases = load_cases(args.dataset)
    sample = selected_cases(cases)
    if not args.run_live:
        print(json.dumps({
            "dataset_case_count": len(cases),
            "case_count": len(sample),
            "model": MODEL,
            "thinking": False,
            "live_execution": False,
            "api_calls": 0,
        }, sort_keys=True))
        return 0
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY before --run-live.")
    client = build_client(api_key, os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"))
    run_live(sample, client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
