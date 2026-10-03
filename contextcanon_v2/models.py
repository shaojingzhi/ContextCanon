"""Frozen ContextCanon V2 governance IR models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Modality(StrEnum):
    OBSERVED = "OBSERVED"
    DOCUMENTED = "DOCUMENTED"
    INTENDED = "INTENDED"
    REQUIRED = "REQUIRED"


class Cardinality(StrEnum):
    SINGLE = "SINGLE"
    MULTI = "MULTI"
    UNKNOWN = "UNKNOWN"


class Relation(StrEnum):
    SUPPORTING = "SUPPORTING"
    COMPATIBLE = "COMPATIBLE"
    CONFLICTING = "CONFLICTING"
    DIVERGENT = "DIVERGENT"
    UNKNOWN = "UNKNOWN"


class SummaryState(StrEnum):
    CLEAR = "CLEAR"
    UNRESOLVED = "UNRESOLVED"
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    source_id: str
    location: str
    content: str
    observed_at: str | None = None
    source_revision: str | None = None


@dataclass(frozen=True, slots=True)
class Claim:
    subject: str
    predicate: str
    value: Any
    value_type: str
    scope: dict[str, Any]
    modality: Modality
    cardinality: Cardinality
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FactNeed:
    subject: str
    dimension: str
    scope_constraint: dict[str, Any] | None = None
    value_type: str | None = None


@dataclass(frozen=True, slots=True)
class RelationRecord:
    left_claim: Claim
    right_claim: Claim
    relation: Relation


@dataclass(frozen=True, slots=True)
class GovernanceResult:
    claims: tuple[Claim, ...]
    relations: tuple[RelationRecord, ...]
    summary_state: SummaryState

    def unresolved_claims(self) -> tuple[Claim, ...]:
        """Return unresolved endpoint Claims without introducing identities."""
        values: list[Claim] = []
        for record in self.relations:
            if record.relation not in {
                Relation.CONFLICTING,
                Relation.DIVERGENT,
                Relation.UNKNOWN,
            }:
                continue
            for claim in (record.left_claim, record.right_claim):
                if claim not in values:
                    values.append(claim)
        return tuple(values)


__all__ = [
    "Cardinality",
    "Claim",
    "Evidence",
    "FactNeed",
    "GovernanceResult",
    "Modality",
    "Relation",
    "RelationRecord",
    "SummaryState",
]
