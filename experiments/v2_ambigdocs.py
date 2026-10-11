"""Offline AmbigDocs adapter and scoring protocol for future V2 runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

if __package__ in {None, ""}:  # Support ``python experiments/v2_ambigdocs.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextcanon_v2 import Evidence, FactNeed


DEFAULT_DATASET = Path("/tmp/ambigdocs/test.json")
MANIFEST = Path(__file__).parents[1] / "benchmarks" / "ambigdocs_test_sample_100.json"


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
        needs.append(FactNeed(subject=title, dimension="answer", value_type="string"))
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
            '{"associations":[{"entity":"...","answer":"...","evidence_ref":"E1"}]}',
            f"Question: {json.dumps(case.question, ensure_ascii=False)}",
            f"Ambiguous entity: {json.dumps(case.ambiguous_entity, ensure_ascii=False)}",
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


if __name__ == "__main__":
    cases = load_cases()
    selected = [case for case in cases if case.qid in load_sample_manifest()]
    print(json.dumps({"case_count": len(cases), "sample_count": len(selected), "api_calls": 0}, indent=2))
