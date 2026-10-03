"""Deterministic governance over the frozen V2 IR."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import json
from collections.abc import Sequence
from typing import Any

from .models import (
    Cardinality,
    Claim,
    FactNeed,
    GovernanceResult,
    Modality,
    Relation,
    RelationRecord,
    SummaryState,
)


def _canonical(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def _value(value: Any, value_type: str) -> tuple[bool, Any]:
    kind = _canonical(value_type)
    if kind in {"string", "enum"}:
        if not isinstance(value, str):
            return False, None
        return True, _canonical(value)
    if kind == "boolean":
        if isinstance(value, bool):
            return True, value
        if isinstance(value, str):
            normalized = _canonical(value)
            if normalized == "true":
                return True, True
            if normalized == "false":
                return True, False
        return False, None
    if kind == "number":
        if isinstance(value, bool):
            return False, None
        try:
            return True, Decimal(str(value).strip())
        except (InvalidOperation, ValueError, AttributeError):
            return False, None
    if kind == "date":
        try:
            if isinstance(value, datetime):
                return True, value.date().isoformat()
            if isinstance(value, date):
                return True, value.isoformat()
            return True, date.fromisoformat(str(value).strip()).isoformat()
        except (TypeError, ValueError):
            return False, None
    return False, None


def _typed_equal(left: Claim, right: Claim) -> bool | None:
    if _canonical(left.value_type) != _canonical(right.value_type):
        return None
    left_ok, left_value = _value(left.value, left.value_type)
    right_ok, right_value = _value(right.value, right.value_type)
    if not left_ok or not right_ok:
        return None
    return left_value == right_value


def _unknown_scope(scope: dict[str, Any]) -> bool:
    return any(
        value is None
        or (isinstance(value, str) and _canonical(value) in {"unknown", "?"})
        for value in scope.values()
    )


def _scope_overlap(left: dict[str, Any], right: dict[str, Any]) -> str:
    for key in set(left) & set(right):
        left_unknown = left[key] is None or (
            isinstance(left[key], str) and _canonical(left[key]) in {"unknown", "?"}
        )
        right_unknown = right[key] is None or (
            isinstance(right[key], str) and _canonical(right[key]) in {"unknown", "?"}
        )
        if not left_unknown and not right_unknown and _canonical(str(left[key])) != _canonical(str(right[key])):
            return "NO"
    if _unknown_scope(left) or _unknown_scope(right) or set(left) != set(right):
        return "UNKNOWN"
    return "YES"


def _scope_matches(claim: Claim, need: FactNeed) -> bool:
    if need.scope_constraint is None:
        return True
    overlap = _scope_overlap(claim.scope, need.scope_constraint)
    return overlap == "YES"


def _same_dimension(left: Claim, right: Claim) -> bool:
    return (
        _canonical(left.subject) == _canonical(right.subject)
        and _canonical(left.predicate) == _canonical(right.predicate)
    )


def _claim_sort_key(claim: Claim) -> str:
    return json.dumps(
        {
            "subject": claim.subject,
            "predicate": claim.predicate,
            "value": claim.value,
            "value_type": claim.value_type,
            "scope": claim.scope,
            "modality": claim.modality.value,
            "cardinality": claim.cardinality.value,
            "evidence_ids": claim.evidence_ids,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _relation(left: Claim, right: Claim) -> Relation:
    overlap = _scope_overlap(left.scope, right.scope)
    equal = _typed_equal(left, right)
    if overlap == "NO":
        return Relation.COMPATIBLE
    if equal is True:
        return Relation.SUPPORTING if overlap == "YES" else Relation.COMPATIBLE
    if overlap == "UNKNOWN" or equal is None:
        return Relation.UNKNOWN
    if left.cardinality is Cardinality.MULTI and right.cardinality is Cardinality.MULTI:
        return Relation.COMPATIBLE
    if (
        Cardinality.UNKNOWN in {left.cardinality, right.cardinality}
        or left.cardinality is not right.cardinality
    ):
        return Relation.UNKNOWN
    target = {Modality.INTENDED, Modality.REQUIRED}
    if (
        (left.modality is Modality.OBSERVED and right.modality in target)
        or (right.modality is Modality.OBSERVED and left.modality in target)
    ):
        return Relation.DIVERGENT
    if Modality.DOCUMENTED in {left.modality, right.modality}:
        return Relation.UNKNOWN
    return Relation.CONFLICTING


def govern(
    claims: Sequence[Claim],
    needs: Sequence[FactNeed],
    *,
    compilation_error: bool = False,
) -> GovernanceResult:
    """Evaluate Claims without semantic inference or source prioritization."""
    ordered_claims = tuple(sorted(claims, key=_claim_sort_key))
    records: list[RelationRecord] = []
    for index, left in enumerate(ordered_claims):
        for right in ordered_claims[index + 1:]:
            if not _same_dimension(left, right):
                continue
            records.append(RelationRecord(left, right, _relation(left, right)))
    records = sorted(
        records,
        key=lambda record: (
            _claim_sort_key(record.left_claim),
            _claim_sort_key(record.right_claim),
            record.relation.value,
        ),
    )

    relevant_claims: list[Claim] = []
    incomplete = compilation_error
    for need in needs:
        matches = tuple(
            claim for claim in ordered_claims
            if _canonical(claim.subject) == _canonical(need.subject)
            and _canonical(claim.predicate) == _canonical(need.dimension)
            and (need.value_type is None or _canonical(claim.value_type) == _canonical(need.value_type))
            and _scope_matches(claim, need)
        )
        if not matches:
            incomplete = True
        for claim in matches:
            if claim not in relevant_claims:
                relevant_claims.append(claim)

    unresolved: list[Claim] = []
    for record in records:
        if record.relation in {Relation.CONFLICTING, Relation.DIVERGENT, Relation.UNKNOWN}:
            if record.left_claim in relevant_claims or record.right_claim in relevant_claims:
                for claim in (record.left_claim, record.right_claim):
                    if claim not in unresolved:
                        unresolved.append(claim)

    if incomplete:
        summary = SummaryState.INCOMPLETE
    elif unresolved:
        summary = SummaryState.UNRESOLVED
    else:
        summary = SummaryState.CLEAR
    return GovernanceResult(
        claims=ordered_claims,
        relations=tuple(records),
        summary_state=summary,
    )


__all__ = ["govern"]
