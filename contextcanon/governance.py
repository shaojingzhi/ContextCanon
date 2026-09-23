"""Open-world evidence governance with deterministic state transitions."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
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
from .semantic import (
    SemanticModelClient,
    parse_semantic_response,
    sanitize_runtime_value,
)


_MIN_SEMANTIC_CONFIDENCE = 0.7
_ALIGNMENT_CANDIDATE_LIMIT = 8
_RELATION_CANDIDATE_LIMIT = 8


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


class DependencyAssessment(StrEnum):
    HAS_DEPENDENCIES = "HAS_DEPENDENCIES"
    NO_DEPENDENCIES = "NO_DEPENDENCIES"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class FactNeedExtractionResult:
    assessment: DependencyAssessment
    needs: tuple[FactNeed, ...] = ()


@dataclass(frozen=True, slots=True)
class ActionRequest:
    tool_name: str
    arguments: JSONValue
    tool_description: str | None = None
    context: JSONValue = None


class FactNeedExtractor(Protocol):
    def extract_for_action(self, action: ActionRequest) -> FactNeedExtractionResult:
        ...


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


def _fact_payload(fact: FactDescriptor) -> dict[str, JSONValue]:
    return {
        "subject": fact.subject,
        "semantic_dimension": fact.semantic_dimension,
        "temporal_scope": fact.temporal_scope.value if fact.temporal_scope else None,
    }


def _evidence_payload(candidate: EvidenceCandidate) -> dict[str, JSONValue]:
    evidence = candidate.evidence
    return {
        "fact": _fact_payload(candidate.fact),
        "value": candidate.value,
        "source": evidence.source_id,
        "role": evidence.role.value,
        "observed_at": evidence.observed_at,
        "valid_from": evidence.valid_from,
        "valid_until": evidence.valid_until,
    }


class LLMEvidenceAligner:
    """Narrow semantic same-fact classifier with conservative failure behavior."""

    def __init__(
        self,
        client: SemanticModelClient,
        *,
        model: str = "deepseek-v4-pro",
        minimum_confidence: float = _MIN_SEMANTIC_CONFIDENCE,
    ) -> None:
        self.client = client
        self.model = model
        self.minimum_confidence = minimum_confidence
        self.diagnostics: list[str] = []
        self.fast_path_hits = 0
        self.alignment_candidates_considered = 0

    def align(
        self,
        incoming: EvidenceCandidate,
        existing_fact: FactDescriptor,
    ) -> AlignmentResult:
        deterministic = ExactFactAligner().align(incoming, existing_fact)
        if deterministic.relation is FactAlignment.SAME_FACT:
            self.fast_path_hits += 1
            return deterministic
        prompt = (
            "Decide only whether these descriptors refer to the same real-world fact in "
            "the same relevant temporal scope. Do not choose which source is true and do "
            "not make an action decision. Return JSON with relation and confidence. "
            "relation must be SAME_FACT, RELATED_BUT_DISTINCT, UNRELATED, or UNKNOWN.\n"
            + json.dumps(
                {
                    "incoming": _fact_payload(incoming.fact),
                    "existing": _fact_payload(existing_fact),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        try:
            payload = parse_semantic_response(
                self.client.complete(prompt, model=self.model)
            )
            if not isinstance(payload, Mapping):
                raise ValueError("alignment response must be an object")
            relation = FactAlignment(str(payload.get("relation", "UNKNOWN")).upper())
            confidence = float(payload.get("confidence", 0.0))
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("alignment confidence is out of range")
            if confidence < self.minimum_confidence:
                return AlignmentResult(FactAlignment.UNKNOWN, confidence)
            return AlignmentResult(relation, confidence)
        except Exception as exc:
            self.diagnostics.append(f"alignment_error:{type(exc).__name__}")
            return AlignmentResult(FactAlignment.UNKNOWN, 0.0)


class DeterministicRelationClassifier:
    """Safe fallback that recognizes support but never invents a conflict."""

    def classify(
        self,
        incoming: EvidenceCandidate,
        existing: EvidenceCandidate,
    ) -> EvidenceRelation:
        if _value_key(incoming.value) == _value_key(existing.value):
            return EvidenceRelation.SUPPORTING
        if incoming.fact.temporal_scope != existing.fact.temporal_scope:
            return EvidenceRelation.COMPATIBLE
        return EvidenceRelation.UNKNOWN


class LLMRelationClassifier:
    """Classify evidence semantics without selecting truth or governance state."""

    def __init__(
        self,
        client: SemanticModelClient,
        *,
        model: str = "deepseek-v4-pro",
        minimum_confidence: float = _MIN_SEMANTIC_CONFIDENCE,
    ) -> None:
        self.client = client
        self.model = model
        self.minimum_confidence = minimum_confidence
        self.diagnostics: list[str] = []
        self.fast_path_hits = 0

    def classify(
        self,
        incoming: EvidenceCandidate,
        existing: EvidenceCandidate,
    ) -> EvidenceRelation:
        if _value_key(incoming.value) == _value_key(existing.value):
            self.fast_path_hits += 1
            return EvidenceRelation.SUPPORTING
        known_scopes = {None, TemporalScope.UNKNOWN}
        if (
            incoming.fact.temporal_scope != existing.fact.temporal_scope
            and incoming.fact.temporal_scope not in known_scopes
            and existing.fact.temporal_scope not in known_scopes
        ):
            self.fast_path_hits += 1
            return EvidenceRelation.COMPATIBLE
        prompt = (
            "Classify only the semantic relation between two evidence items already aligned "
            "to one fact. Do not choose truth, set governance state, or authorize an action. "
            "Return JSON with relation and confidence. relation must be SUPPORTING, "
            "EQUIVALENT, CONFLICTING, SUPERSEDING, COMPATIBLE, or UNKNOWN.\n"
            + json.dumps(
                {
                    "existing": _evidence_payload(existing),
                    "incoming": _evidence_payload(incoming),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        try:
            payload = parse_semantic_response(
                self.client.complete(prompt, model=self.model)
            )
            if not isinstance(payload, Mapping):
                raise ValueError("relation response must be an object")
            relation = EvidenceRelation(str(payload.get("relation", "UNKNOWN")).upper())
            confidence = float(payload.get("confidence", 0.0))
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("relation confidence is out of range")
            if confidence < self.minimum_confidence:
                return EvidenceRelation.UNKNOWN
            return relation
        except Exception as exc:
            self.diagnostics.append(f"relation_error:{type(exc).__name__}")
            return EvidenceRelation.UNKNOWN


class LLMFactNeedExtractor:
    """Extract ephemeral FactNeed values from runtime-visible action details."""

    def __init__(
        self,
        client: SemanticModelClient,
        *,
        model: str = "deepseek-v4-pro",
    ) -> None:
        self.client = client
        self.model = model
        self.diagnostics: list[str] = []

    def extract_for_action(self, action: ActionRequest) -> FactNeedExtractionResult:
        payload = {
            "tool_name": action.tool_name,
            "tool_description": action.tool_description,
            "arguments": sanitize_runtime_value(action.arguments),
            "context": sanitize_runtime_value(action.context),
        }
        prompt = (
            "Assess and extract factual dependencies required to safely execute this "
            "proposed action. Return JSON with assessment and a needs array. assessment "
            "must be HAS_DEPENDENCIES, NO_DEPENDENCIES, or UNKNOWN. HAS_DEPENDENCIES means "
            "the action contains or relies on factual claims whose correctness matters. "
            "NO_DEPENDENCIES means the action can execute correctly without any factual "
            "claim from external runtime evidence. UNKNOWN means this is unclear. Be "
            "conservative. Each need contains subject_hint, "
            "semantic_dimension, expected_value_type or null, temporal_scope or null, and "
            "required. Do not output ALLOW, BLOCK, REQUIRE_CLARIFICATION, or any governance "
            "state. Examples: send_message containing an event date and submit_return whose "
            "eligibility depends on order state are HAS_DEPENDENCIES; a healthcheck ping "
            "with no external factual assertion is NO_DEPENDENCIES.\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )
        try:
            result = parse_semantic_response(
                self.client.complete(prompt, model=self.model)
            )
            if not isinstance(result, Mapping) or not isinstance(result.get("needs"), list):
                raise ValueError("fact need response must contain assessment and needs")
            assessment = DependencyAssessment(str(result.get("assessment", "UNKNOWN")).upper())
            needs: list[FactNeed] = []
            invalid_need = False
            for item in result["needs"]:
                if not isinstance(item, Mapping):
                    invalid_need = True
                    continue
                subject = item.get("subject_hint")
                dimension = item.get("semantic_dimension")
                if not isinstance(subject, str) or not subject.strip():
                    invalid_need = True
                    continue
                if not isinstance(dimension, str) or not dimension.strip():
                    invalid_need = True
                    continue
                raw_scope = item.get("temporal_scope")
                scope = None if raw_scope is None else TemporalScope(str(raw_scope).upper())
                expected_type = item.get("expected_value_type")
                if expected_type is not None and not isinstance(expected_type, str):
                    invalid_need = True
                    continue
                required = item.get("required", True)
                if not isinstance(required, bool):
                    invalid_need = True
                    continue
                needs.append(FactNeed(
                    subject.strip(),
                    re.sub(r"\s+", " ", dimension.strip()),
                    expected_type,
                    scope,
                    required,
                ))
            if invalid_need:
                self.diagnostics.append("fact_need_error:invalid_need")
                return FactNeedExtractionResult(DependencyAssessment.UNKNOWN)
            if assessment is DependencyAssessment.NO_DEPENDENCIES and needs:
                self.diagnostics.append("fact_need_error:inconsistent_assessment")
                return FactNeedExtractionResult(DependencyAssessment.UNKNOWN)
            return FactNeedExtractionResult(assessment, tuple(needs))
        except Exception as exc:
            self.diagnostics.append(f"fact_need_error:{type(exc).__name__}")
            return FactNeedExtractionResult(DependencyAssessment.UNKNOWN)


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
        clock: Callable[[], datetime] | None = None,
        alignment_candidate_limit: int = _ALIGNMENT_CANDIDATE_LIMIT,
        relation_candidate_limit: int = _RELATION_CANDIDATE_LIMIT,
    ) -> None:
        self.aligner = aligner or ExactFactAligner()
        self.relation_classifier = relation_classifier or DeterministicRelationClassifier()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        if alignment_candidate_limit < 1:
            raise ValueError("alignment_candidate_limit must be positive")
        if relation_candidate_limit < 1:
            raise ValueError("relation_candidate_limit must be positive")
        self.alignment_candidate_limit = alignment_candidate_limit
        self.relation_candidate_limit = relation_candidate_limit
        self.facts: list[GovernedFact] = []

    def ingest(
        self,
        candidate: EvidenceCandidate,
        *,
        relation_overrides: Mapping[str, EvidenceRelation] | None = None,
    ) -> GovernedFact:
        governed = self._find_same_fact(candidate)
        if governed is None:
            governed = GovernedFact(candidate.fact)
            self.facts.append(governed)

        active_existing = [
            record for record in governed.records
            if record.superseded_by is None and self._freshness(record.evidence) is True
        ][-self.relation_candidate_limit :]
        overrides = relation_overrides or {}
        relations = tuple(
            EvidenceRelationRecord(
                record.evidence.id,
                overrides.get(record.evidence.id)
                or self.relation_classifier.classify(candidate, record.candidate),
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
            governed for governed in self._alignment_candidates(candidate.fact)
            if self.aligner.align(candidate, governed.fact).relation is FactAlignment.SAME_FACT
        ]
        if len(matches) != 1:
            return None
        return self.result_for(matches[0])

    def result_for(self, governed: GovernedFact) -> QueryGovernanceResult:
        """Render one stored fact without another semantic alignment call."""

        self._recompute(governed)
        unsuperseded = [
            record for record in governed.records
            if record.superseded_by is None
        ]
        active = [
            record for record in unsuperseded
            if self._freshness(record.evidence) is True
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
        candidates = self._alignment_candidates(candidate.fact)
        if hasattr(self.aligner, "alignment_candidates_considered"):
            self.aligner.alignment_candidates_considered += len(candidates)  # type: ignore[attr-defined]
        matches = [
            governed for governed in candidates
            if self.aligner.align(candidate, governed.fact).relation is FactAlignment.SAME_FACT
        ]
        return matches[0] if len(matches) == 1 else None

    def _alignment_candidates(
        self,
        fact: FactDescriptor,
    ) -> list[GovernedFact]:
        subject_tokens = set(_semantic_text(fact.subject).split())
        lexical_candidates = [
            governed
            for governed in reversed(self.facts)
            if subject_tokens & set(_semantic_text(governed.fact.subject).split())
        ]
        if lexical_candidates:
            return lexical_candidates[: self.alignment_candidate_limit]
        return list(reversed(self.facts))[: self.alignment_candidate_limit]

    def _now(self) -> datetime:
        current = self.clock()
        if current.tzinfo is None:
            return current.replace(tzinfo=timezone.utc)
        return current.astimezone(timezone.utc)

    def _freshness(self, evidence: Evidence) -> bool | None:
        valid_from = _parse_time(evidence.valid_from)
        valid_until = _parse_time(evidence.valid_until)
        if evidence.valid_from is not None and valid_from is None:
            return None
        if evidence.valid_until is not None and valid_until is None:
            return None
        now = self._now()
        if valid_from is not None and now < valid_from:
            return False
        if valid_until is not None and now > valid_until:
            return False
        return True

    def _recompute(self, governed: GovernedFact) -> None:
        unsuperseded = [
            record for record in governed.records if record.superseded_by is None
        ]
        active = [
            record
            for record in unsuperseded
            if self._freshness(record.evidence) is True
        ]
        if not active:
            if not unsuperseded:
                governed.state = GovernanceState.SUPERSEDED
            elif any(self._freshness(record.evidence) is None for record in unsuperseded):
                governed.state = GovernanceState.AMBIGUOUS
            else:
                governed.state = GovernanceState.STALE
            return
        active_ids = {record.evidence.id for record in active}
        relations = {
            item.relation
            for record in active
            for item in record.relations
            if item.existing_evidence_id in active_ids
        }
        if EvidenceRelation.UNKNOWN in relations:
            governed.state = GovernanceState.AMBIGUOUS
            return
        if EvidenceRelation.CONFLICTING in relations:
            governed.state = GovernanceState.DIVERGED
            return
        governed.state = GovernanceState.RESOLVED

__all__ = [
    "ActionDependency",
    "ActionGovernanceDecision",
    "ActionGovernanceResult",
    "ActionRequest",
    "AlignmentResult",
    "DeterministicRelationClassifier",
    "DependencyAssessment",
    "EvidenceAligner",
    "EvidenceCandidate",
    "EvidenceRelationRecord",
    "ExactFactAligner",
    "FactDescriptor",
    "FactNeed",
    "FactNeedExtractionResult",
    "FactNeedExtractor",
    "GovernanceRecord",
    "GovernanceStore",
    "GovernedFact",
    "LLMEvidenceAligner",
    "LLMFactNeedExtractor",
    "LLMRelationClassifier",
    "QueryGovernanceResult",
    "RelationClassifier",
]
