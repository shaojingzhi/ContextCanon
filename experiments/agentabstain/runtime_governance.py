"""Generic open-world governance path for live AgentAbstain observations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from contextcanon.core import Evidence, EvidenceRole, GovernanceState, SourceType, TemporalScope
from contextcanon.governance import (
    ActionGovernanceDecision,
    ActionGovernanceResult,
    ActionRequest,
    DependencyAssessment,
    EvidenceCandidate,
    FactDescriptor,
    FactNeedExtractor,
    GovernanceStore,
)
from contextcanon.semantic import ClaimCandidate, SemanticExtractor, normalize_candidate
from contextcanon.semantic_budget import SemanticBudget

from .adapter import ProposedToolCall, RuntimeObservation


def _stable_evidence_id(candidate: ClaimCandidate) -> str:
    payload = json.dumps(
        [
            candidate.provenance.observation_id,
            candidate.provenance.tool_name,
            candidate.provenance.location,
            candidate.value,
        ],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"runtime-evidence-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _evidence_candidate(candidate: ClaimCandidate) -> EvidenceCandidate:
    scope = (
        TemporalScope(candidate.temporal_scope)
        if candidate.temporal_scope is not None
        else None
    )
    provenance = candidate.provenance
    evidence = Evidence(
        id=_stable_evidence_id(candidate),
        source_id=provenance.tool_name,
        source_type=SourceType.OTHER,
        location=provenance.location,
        role=EvidenceRole(candidate.role),
        content=candidate.value,
        verified=None,
        temporal_scope=scope,
        confidence=candidate.confidence,
        observed_at=candidate.observed_at,
        valid_from=candidate.valid_from,
        valid_until=candidate.valid_until,
        observation_id=provenance.observation_id,
        provenance={
            "tool_name": provenance.tool_name,
            "tool_arguments": provenance.tool_arguments,
            "location": provenance.location,
            "excerpt": provenance.excerpt,
        },
    )
    return EvidenceCandidate(
        FactDescriptor(candidate.entity, candidate.property, scope),
        candidate.value,
        evidence,
    )


@dataclass(slots=True)
class RuntimeGovernance:
    """Task-local ingestion, query, and action consumer over one GovernanceStore."""

    extractor: SemanticExtractor
    fact_need_extractor: FactNeedExtractor
    store: GovernanceStore
    action_context: object = None
    budget: SemanticBudget | None = None
    tool_semantics_resolver: object | None = None
    observations: list[RuntimeObservation] = field(default_factory=list)

    def observe(self, observation: RuntimeObservation) -> list[EvidenceCandidate]:
        self.observations.append(observation)
        ingested: list[EvidenceCandidate] = []
        for raw_candidate in self.extractor.extract(observation):
            candidate = normalize_candidate(raw_candidate)
            if candidate is None:
                continue
            evidence_candidate = _evidence_candidate(candidate)
            self.store.ingest(evidence_candidate)
            ingested.append(evidence_candidate)
        return ingested

    def evaluate_action(self, proposed: ProposedToolCall) -> ActionGovernanceResult:
        if self.budget is not None and self.budget.governance_incomplete:
            return ActionGovernanceResult(ActionGovernanceDecision.REQUIRE_CLARIFICATION)
        extraction = self.fact_need_extractor.extract_for_action(ActionRequest(
            proposed.tool_name,
            proposed.parameters,
            context=self.action_context,
        ))
        if (
            extraction.assessment is DependencyAssessment.NO_DEPENDENCIES
            and not extraction.needs
        ):
            return ActionGovernanceResult(ActionGovernanceDecision.ALLOW)
        if (
            extraction.assessment is DependencyAssessment.HAS_DEPENDENCIES
            and extraction.needs
        ):
            return self.store.evaluate_action(list(extraction.needs))
        return ActionGovernanceResult(
            ActionGovernanceDecision.REQUIRE_CLARIFICATION
        )

    @property
    def metrics(self) -> dict[str, object]:
        metrics: dict[str, object] = {}
        if self.budget is not None:
            metrics.update(self.budget.diagnostics())
        for name, component in (
            ("fast_path_alignment_hits", self.store.aligner),
            ("alignment_candidates_considered", self.store.aligner),
            ("fast_path_relation_hits", self.store.relation_classifier),
            ("tool_semantics_cache_hits", getattr(self, "tool_semantics_resolver", None)),
        ):
            if component is not None and hasattr(component, name):
                metrics[name] = getattr(component, name)
        return metrics

    def render(self) -> str:
        lines = ["Governed runtime evidence:"]
        facts = sorted(
            self.store.facts,
            key=lambda item: (
                item.fact.subject.casefold(),
                item.fact.semantic_dimension.casefold(),
                item.fact.temporal_scope.value if item.fact.temporal_scope else "",
            ),
        )
        for governed in facts:
            result = self.store.result_for(governed)
            lines.extend([
                "",
                f"Fact: {result.fact.subject} / {result.fact.semantic_dimension}",
                f"State: {result.state.value}",
                "Evidence:",
            ])
            for value in result.candidate_values:
                lines.append(f"- {value}")
            lines.append(f"Sources: {', '.join(result.sources)}")
        return "\n".join(lines)

    @property
    def conflicts_detected(self) -> int:
        self.store.refresh()
        return sum(
            fact.state is GovernanceState.DIVERGED
            for fact in self.store.facts
        )

    @property
    def diagnostics(self) -> tuple[str, ...]:
        messages: list[str] = []
        components = (
            self.extractor,
            self.store.aligner,
            self.store.relation_classifier,
            self.fact_need_extractor,
        )
        for component in components:
            messages.extend(str(item) for item in getattr(component, "diagnostics", ()))
        return tuple(messages)


__all__ = ["RuntimeGovernance"]
