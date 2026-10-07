"""Minimal offline RAMDocs adapter for ContextCanon V2 evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # Support ``python experiments/v2_ramdocs.py``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextcanon_v2 import (
    Evidence,
    FactNeed,
    Relation,
    SemanticCompiler,
    SummaryState,
    govern,
)


DEFAULT_DATASET = Path("/tmp/RAMDocs_test.jsonl")
EXPECTED_SHA256 = "c67f699c97349f00cf1bd08d1dbf8ca1d0cc38c306715c93a10a4f961dcf28b7"
EXPECTED_CASES = 500
EXPECTED_DOCUMENTS = 2766
DOCUMENT_TYPES = {"correct", "misinfo", "noise"}
SUMMARY_VALUES = {state.value for state in SummaryState}


@dataclass(frozen=True, slots=True)
class RamDocsCase:
    row_index: int
    question: str
    evidence: tuple[Evidence, ...]
    fact_needs: tuple[FactNeed, ...]
    gold_relevant_evidence_ids: frozenset[str]
    gold_correct_evidence_ids: frozenset[str]
    gold_misinfo_evidence_ids: frozenset[str]
    gold_competing_evidence: bool
    gold_summary_proxy: SummaryState


@dataclass(frozen=True, slots=True)
class RamDocsPrediction:
    relevant_evidence_ids: frozenset[str]
    competing_evidence: bool
    summary_state: SummaryState


def _case_from_row(row_index: int, row: dict[str, Any]) -> RamDocsCase:
    if set(row) != {
        "question", "documents", "disambig_entity", "gold_answers", "wrong_answers"
    }:
        raise ValueError(f"RAMDocs row {row_index} has unexpected fields")
    question = row["question"]
    documents = row["documents"]
    if not isinstance(question, str) or not question.strip() or not isinstance(documents, list):
        raise ValueError(f"RAMDocs row {row_index} is malformed")

    evidence: list[Evidence] = []
    correct: set[str] = set()
    misinfo: set[str] = set()
    for document_index, document in enumerate(documents):
        if not isinstance(document, dict) or set(document) != {"text", "type", "answer"}:
            raise ValueError(f"RAMDocs row {row_index} document {document_index} is malformed")
        document_type = document["type"]
        content = document["text"]
        if document_type not in DOCUMENT_TYPES or not isinstance(content, str):
            raise ValueError(f"RAMDocs row {row_index} document {document_index} is malformed")
        evidence_id = f"E{document_index + 1}"
        evidence.append(Evidence(
            evidence_id=evidence_id,
            source_id=f"ramdocs/row-{row_index:04d}/document-{document_index:02d}",
            location=f"documents[{document_index}]",
            content=content,
        ))
        if document_type == "correct":
            correct.add(evidence_id)
        elif document_type == "misinfo":
            misinfo.add(evidence_id)

    relevant = correct | misinfo
    if not correct:
        summary = SummaryState.INCOMPLETE
    elif misinfo:
        summary = SummaryState.UNRESOLVED
    else:
        summary = SummaryState.CLEAR
    return RamDocsCase(
        row_index=row_index,
        question=question,
        evidence=tuple(evidence),
        fact_needs=(FactNeed(subject=question, dimension="answer"),),
        gold_relevant_evidence_ids=frozenset(relevant),
        gold_correct_evidence_ids=frozenset(correct),
        gold_misinfo_evidence_ids=frozenset(misinfo),
        gold_competing_evidence=bool(correct and misinfo),
        gold_summary_proxy=summary,
    )


def load_cases(path: Path = DEFAULT_DATASET) -> tuple[RamDocsCase, ...]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"RAMDocs cache checksum mismatch: {digest}")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != EXPECTED_CASES:
        raise ValueError(f"RAMDocs cache must contain exactly {EXPECTED_CASES} examples")
    cases = tuple(_case_from_row(index, row) for index, row in enumerate(rows))
    if sum(len(case.evidence) for case in cases) != EXPECTED_DOCUMENTS:
        raise ValueError(f"RAMDocs cache must contain exactly {EXPECTED_DOCUMENTS} documents")
    return cases


def contextcanon_prediction(
    case: RamDocsCase,
    complete: Callable[[str], str],
) -> RamDocsPrediction:
    """Make one semantic completion and derive RAMDocs targets from its provenance."""
    claims = SemanticCompiler(complete).compile(case.evidence, case.fact_needs)
    result = govern(claims, case.fact_needs)
    relevant_ids = frozenset(
        evidence_id
        for claim in result.claims
        for evidence_id in claim.evidence_ids
    )
    distinct_values = {
        json.dumps(claim.value, ensure_ascii=False, sort_keys=True, default=str)
        for claim in result.claims
    }
    unresolved = any(
        record.relation in {Relation.CONFLICTING, Relation.DIVERGENT, Relation.UNKNOWN}
        for record in result.relations
    )
    return RamDocsPrediction(
        relevant_evidence_ids=relevant_ids,
        competing_evidence=len(distinct_values) > 1 or unresolved,
        summary_state=result.summary_state,
    )


def direct_prompt(case: RamDocsCase) -> str:
    evidence = [
        {
            "ref": item.evidence_id,
            "source": item.source_id,
            "location": item.location,
            "content": item.content,
        }
        for item in case.evidence
    ]
    return (
        "Evaluate the supplied question and Evidence. Return only JSON; do not provide "
        "reasoning. relevant_evidence_refs must contain Evidence that is relevant to "
        "answering the question, including evidence that supports a competing answer. "
        "competing_evidence is true only when the relevant Evidence supports competing "
        "answers. summary_state is INCOMPLETE when no usable answer evidence exists, "
        "UNRESOLVED when competing answer evidence exists, otherwise CLEAR.\n"
        "Return exactly: "
        '{"relevant_evidence_refs":["E1"],"competing_evidence":false,'
        '"summary_state":"CLEAR"}\n'
        f"Question: {json.dumps(case.question, ensure_ascii=False)}\n"
        f"Evidence: {json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))}"
    )


def parse_direct_response(raw_response: str, case: RamDocsCase) -> RamDocsPrediction:
    try:
        payload = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("response is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {
        "relevant_evidence_refs", "competing_evidence", "summary_state"
    }:
        raise ValueError("response fields do not match the RAMDocs schema")
    refs = payload["relevant_evidence_refs"]
    if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
        raise ValueError("relevant_evidence_refs must be a string list")
    if len(refs) != len(set(refs)):
        raise ValueError("relevant_evidence_refs contains duplicates")
    allowed = {item.evidence_id for item in case.evidence}
    if not set(refs) <= allowed:
        raise ValueError("relevant_evidence_refs contains an unknown reference")
    competing = payload["competing_evidence"]
    if not isinstance(competing, bool):
        raise ValueError("competing_evidence must be boolean")
    summary = payload["summary_state"]
    if summary not in SUMMARY_VALUES:
        raise ValueError("summary_state is invalid")
    return RamDocsPrediction(frozenset(refs), competing, SummaryState(summary))


def direct_prediction(
    case: RamDocsCase,
    complete: Callable[[str], str],
) -> RamDocsPrediction:
    """Make exactly one direct-baseline completion for one RAMDocs case."""
    return parse_direct_response(complete(direct_prompt(case)), case)


def _metric(numerator: int, denominator: int) -> dict[str, int | float]:
    return {
        "count": numerator,
        "total": denominator,
        "rate": numerator / denominator if denominator else 0.0,
    }


def score_predictions(
    cases: Sequence[RamDocsCase],
    predictions: Mapping[int, RamDocsPrediction],
) -> dict[str, Any]:
    expected_rows = {case.row_index for case in cases}
    if set(predictions) != expected_rows:
        raise ValueError("predictions must contain exactly one result for every case")

    relevant_tp = relevant_fp = relevant_fn = 0
    correct_retained = correct_total = 0
    misinfo_retained = misinfo_total = 0
    noise_rejected = noise_total = 0
    competing_tp = competing_fp = competing_fn = competing_tn = 0
    summary_correct = 0
    summary_counts = {
        state.value: {"gold": 0, "predicted": 0, "correct": 0}
        for state in SummaryState
    }

    for case in cases:
        prediction = predictions[case.row_index]
        all_ids = {item.evidence_id for item in case.evidence}
        if not prediction.relevant_evidence_ids <= all_ids:
            raise ValueError(f"row {case.row_index} prediction references unknown Evidence")
        gold = case.gold_relevant_evidence_ids
        predicted = prediction.relevant_evidence_ids
        relevant_tp += len(gold & predicted)
        relevant_fp += len(predicted - gold)
        relevant_fn += len(gold - predicted)
        correct_retained += len(case.gold_correct_evidence_ids & predicted)
        correct_total += len(case.gold_correct_evidence_ids)
        misinfo_retained += len(case.gold_misinfo_evidence_ids & predicted)
        misinfo_total += len(case.gold_misinfo_evidence_ids)
        noise = all_ids - gold
        noise_rejected += len(noise - predicted)
        noise_total += len(noise)

        if case.gold_competing_evidence and prediction.competing_evidence:
            competing_tp += 1
        elif not case.gold_competing_evidence and prediction.competing_evidence:
            competing_fp += 1
        elif case.gold_competing_evidence:
            competing_fn += 1
        else:
            competing_tn += 1

        gold_summary = case.gold_summary_proxy.value
        predicted_summary = prediction.summary_state.value
        summary_counts[gold_summary]["gold"] += 1
        summary_counts[predicted_summary]["predicted"] += 1
        if predicted_summary == gold_summary:
            summary_correct += 1
            summary_counts[gold_summary]["correct"] += 1

    precision_denominator = relevant_tp + relevant_fp
    recall_denominator = relevant_tp + relevant_fn
    competing_precision_denominator = competing_tp + competing_fp
    competing_recall_denominator = competing_tp + competing_fn
    competing_precision = (
        competing_tp / competing_precision_denominator
        if competing_precision_denominator else 0.0
    )
    competing_recall = (
        competing_tp / competing_recall_denominator
        if competing_recall_denominator else 0.0
    )
    competing_f1 = (
        2 * competing_precision * competing_recall / (competing_precision + competing_recall)
        if competing_precision + competing_recall else 0.0
    )
    return {
        "case_count": len(cases),
        "evidence_relevance": {
            "relevant_precision": _metric(relevant_tp, precision_denominator),
            "relevant_recall": _metric(relevant_tp, recall_denominator),
            "correct_recall": _metric(correct_retained, correct_total),
            "misinfo_retention_rate": _metric(misinfo_retained, misinfo_total),
            "noise_rejection_rate": _metric(noise_rejected, noise_total),
        },
        "competing_evidence": {
            "accuracy": _metric(competing_tp + competing_tn, len(cases)),
            "precision": _metric(competing_tp, competing_precision_denominator),
            "recall": _metric(competing_tp, competing_recall_denominator),
            "f1": competing_f1,
            "counts": {
                "true_positive": competing_tp,
                "false_positive": competing_fp,
                "false_negative": competing_fn,
                "true_negative": competing_tn,
            },
        },
        "summary_proxy": {
            "accuracy": _metric(summary_correct, len(cases)),
            "per_class_counts": summary_counts,
            "note": (
                "RAMDocs dataset-derived proxy only; INCOMPLETE has three Gold cases and "
                "is not a robust estimate of INCOMPLETE detection."
            ),
        },
    }


def dataset_summary(cases: Sequence[RamDocsCase]) -> dict[str, Any]:
    type_counts = Counter()
    combination_counts = Counter()
    for case in cases:
        correct = len(case.gold_correct_evidence_ids)
        misinfo = len(case.gold_misinfo_evidence_ids)
        noise = len(case.evidence) - correct - misinfo
        type_counts.update({"correct": correct, "misinfo": misinfo, "noise": noise})
        combination_counts.update({
            "correct+misinfo": bool(correct and misinfo),
            "correct+noise": bool(correct and noise),
            "correct+misinfo+noise": bool(correct and misinfo and noise),
        })
    summaries = Counter(case.gold_summary_proxy.value for case in cases)
    return {
        "case_count": len(cases),
        "document_count": sum(len(case.evidence) for case in cases),
        "document_type_counts": dict(type_counts),
        "combination_counts": dict(combination_counts),
        "summary_proxy_counts": dict(summaries),
        "live_execution": False,
        "api_calls": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args(argv)
    print(json.dumps(dataset_summary(load_cases(args.dataset)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
