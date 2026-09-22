"""Open-world evidence governance with deterministic state transitions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Protocol

from .core import (
    Evidence,
    EvidenceRole,
    EvidenceRelation,
    FactAlignment,
    GovernanceState,
    JSONValue,
    SourceType,
    TemporalScope,
)


def _semantic_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _value_key(value: JSONValue) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class FactDescriptor:
    """Open-world fact identity expressed without a business ontology."""

    subject: str
    semantic_dimension: str
    temporal_scope: TemporalScope | None = None

    def __post_init__(self) -> None:
        if not self.subject.strip():
            raise ValueError("subject must not be empty")
        if not self.semantic_dimension.strip():
            raise ValueError("semantic_dimension must not be empty")
        if self.temporal_scope is not None and not isinstance(
            self.temporal_scope, TemporalScope
        ):
            raise TypeError("temporal_scope must be a TemporalScope or None")


@dataclass(frozen=True, slots=True)
class FactNeed:
    """Ephemeral fact requirement shared by query and action consumers."""

    subject_hint: str
    semantic_dimension: str
    expected_value_type: str | None = None
    temporal_scope: TemporalScope | None = None
    required: bool = True


ActionDependency = FactNeed


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    relation: FactAlignment
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.relation, FactAlignment):
            raise TypeError("relation must be a FactAlignment")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    fact: FactDescriptor
    value: JSONValue
    evidence: Evidence


class EvidenceAligner(Protocol):
    def align(
        self,
        incoming: EvidenceCandidate,
        existing_fact: FactDescriptor,
    ) -> AlignmentResult:
        ...


class RelationClassifier(Protocol):
    def classify(
        self,
        incoming: EvidenceCandidate,
        existing: EvidenceCandidate,
    ) -> EvidenceRelation:
        ...


class ExactFactAligner:
    """Conservative default; semantic aligners can be injected later."""

    def align(
        self,
        incoming: EvidenceCandidate,
        existing_fact: FactDescriptor,
    ) -> AlignmentResult:
        same_subject = _semantic_text(incoming.fact.subject) == _semantic_text(
            existing_fact.subject
        )
        same_dimension = _semantic_text(
            incoming.fact.semantic_dimension
        ) == _semantic_text(existing_fact.semantic_dimension)
        if not same_subject:
            return AlignmentResult(FactAlignment.UNRELATED, 1.0)
        if not same_dimension:
            return AlignmentResult(FactAlignment.RELATED_BUT_DISTINCT, 1.0)
        incoming_scope = incoming.fact.temporal_scope
        existing_scope = existing_fact.temporal_scope
        if incoming_scope == existing_scope:
            return AlignmentResult(FactAlignment.SAME_FACT, 1.0)
        unknown = {None, TemporalScope.UNKNOWN}
        if incoming_scope in unknown or existing_scope in unknown:
            return AlignmentResult(FactAlignment.UNKNOWN, 0.0)
        return AlignmentResult(FactAlignment.RELATED_BUT_DISTINCT, 1.0)


class DeterministicRelationClassifier:
    """Compare normalized values only after evidence has aligned to one fact."""

    def classify(
        self,
        incoming: EvidenceCandidate,
        existing: EvidenceCandidate,
    ) -> EvidenceRelation:
        if _value_key(incoming.value) == _value_key(existing.value):
            return EvidenceRelation.SUPPORTING
        if incoming.fact.temporal_scope != existing.fact.temporal_scope:
            return EvidenceRelation.COMPATIBLE
        return EvidenceRelation.CONFLICTING


@dataclass(frozen=True, slots=True)
class EvidenceRelationRecord:
    existing_evidence_id: str
    relation: EvidenceRelation


@dataclass(slots=True)
class GovernanceRecord:
    candidate: EvidenceCandidate
    relations: tuple[EvidenceRelationRecord, ...] = ()
    superseded_by: str | None = None

    @property
    def evidence(self) -> Evidence:
        return self.candidate.evidence


@dataclass(slots=True)
class GovernedFact:
    fact: FactDescriptor
    state: GovernanceState = GovernanceState.UNVERIFIED
    records: list[GovernanceRecord] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class QueryGovernanceResult:
    fact: FactDescriptor
    state: GovernanceState
    candidate_values: tuple[JSONValue, ...]
    evidence_refs: tuple[str, ...]
    sources: tuple[str, ...]


class ActionGovernanceDecision(StrEnum):
    ALLOW = "ALLOW"
    REQUIRE_CLARIFICATION = "REQUIRE_CLARIFICATION"


@dataclass(frozen=True, slots=True)
class ActionGovernanceResult:
    decision: ActionGovernanceDecision
    unresolved: tuple[QueryGovernanceResult, ...] = ()
    missing_needs: tuple[FactNeed, ...] = ()


class GovernanceStore:
    """Small in-memory store that owns deterministic governance transitions."""

    def __init__(
        self,
        *,
        aligner: EvidenceAligner | None = None,
        relation_classifier: RelationClassifier | None = None,
        now: datetime | None = None,
    ) -> None:
        self.aligner = aligner or ExactFactAligner()
        self.relation_classifier = relation_classifier or DeterministicRelationClassifier()
        self.now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        self.facts: list[GovernedFact] = []

    def ingest(
        self,
        candidate: EvidenceCandidate,
        *,
        relation: EvidenceRelation | None = None,
    ) -> GovernedFact:
        governed = self._find_same_fact(candidate)
        if governed is None:
            governed = GovernedFact(candidate.fact)
            self.facts.append(governed)

        active_existing = [
            record for record in governed.records
            if record.superseded_by is None and not self._expired(record.evidence)
        ]
        relations = tuple(
            EvidenceRelationRecord(
                record.evidence.id,
                relation or self.relation_classifier.classify(candidate, record.candidate),
            )
            for record in active_existing
        )
        superseded_ids = {
            item.existing_evidence_id
            for item in relations
            if item.relation is EvidenceRelation.SUPERSEDING
        }
        if superseded_ids:
            for record in active_existing:
                if record.evidence.id not in superseded_ids:
                    continue
                record.superseded_by = candidate.evidence.id
        governed.records.append(GovernanceRecord(candidate, relations))
        self._recompute(governed)
        return governed

    def refresh(self) -> None:
        for governed in self.facts:
            self._recompute(governed)

    def query(self, need: FactNeed) -> QueryGovernanceResult | None:
        candidate = EvidenceCandidate(
            FactDescriptor(
                need.subject_hint,
                need.semantic_dimension,
                need.temporal_scope,
            ),
            None,
            Evidence(
                id="query-need",
                source_id="query",
                source_type=SourceType.OTHER,
                location="dynamic",
                role=EvidenceRole.DOCUMENTED,
            ),
        )
        matches = [
            governed for governed in self.facts
            if self.aligner.align(candidate, governed.fact).relation is FactAlignment.SAME_FACT
        ]
        if len(matches) != 1:
            return None
        governed = matches[0]
        self._recompute(governed)
        unsuperseded = [
            record for record in governed.records
            if record.superseded_by is None
        ]
        active = [
            record for record in unsuperseded
            if not self._expired(record.evidence)
        ]
        records = active or unsuperseded
        values_by_key = {
            _value_key(record.candidate.value): record.candidate.value
            for record in records
        }
        return QueryGovernanceResult(
            fact=governed.fact,
            state=governed.state,
            candidate_values=tuple(
                values_by_key[key] for key in sorted(values_by_key)
            ),
            evidence_refs=tuple(sorted(record.evidence.id for record in records)),
            sources=tuple(sorted({record.evidence.source_id for record in records})),
        )

    def evaluate_action(self, needs: list[FactNeed]) -> ActionGovernanceResult:
        unresolved: list[QueryGovernanceResult] = []
        missing: list[FactNeed] = []
        for need in needs:
            result = self.query(need)
            if result is None:
                if need.required:
                    missing.append(need)
                continue
            if result.state is not GovernanceState.RESOLVED and need.required:
                unresolved.append(result)
        if unresolved or missing:
            return ActionGovernanceResult(
                ActionGovernanceDecision.REQUIRE_CLARIFICATION,
                tuple(unresolved),
                tuple(missing),
            )
        return ActionGovernanceResult(ActionGovernanceDecision.ALLOW)

    def _find_same_fact(self, candidate: EvidenceCandidate) -> GovernedFact | None:
        matches = [
            governed for governed in self.facts
            if self.aligner.align(candidate, governed.fact).relation is FactAlignment.SAME_FACT
        ]
        return matches[0] if len(matches) == 1 else None

    def _expired(self, evidence: Evidence) -> bool:
        valid_until = _parse_time(evidence.valid_until)
        return valid_until is not None and valid_until < self.now

    def _recompute(self, governed: GovernedFact) -> None:
        unsuperseded = [
            record for record in governed.records if record.superseded_by is None
        ]
        active = [record for record in unsuperseded if not self._expired(record.evidence)]
        if not active:
            governed.state = (
                GovernanceState.STALE if unsuperseded else GovernanceState.SUPERSEDED
            )
            return
        relations = {
            item.relation for record in active for item in record.relations
        }
        if EvidenceRelation.UNKNOWN in relations:
            governed.state = GovernanceState.AMBIGUOUS
            return
        if EvidenceRelation.CONFLICTING in relations:
            governed.state = GovernanceState.DIVERGED
            return
        if any(record.evidence.verified is True for record in active):
            governed.state = GovernanceState.RESOLVED
            return
        governed.state = GovernanceState.UNVERIFIED

__all__ = [
    "ActionDependency",
    "ActionGovernanceDecision",
    "ActionGovernanceResult",
    "AlignmentResult",
    "DeterministicRelationClassifier",
    "EvidenceAligner",
    "EvidenceCandidate",
    "EvidenceRelationRecord",
    "ExactFactAligner",
    "FactDescriptor",
    "FactNeed",
    "GovernanceRecord",
    "GovernanceStore",
    "GovernedFact",
    "QueryGovernanceResult",
    "RelationClassifier",
]
