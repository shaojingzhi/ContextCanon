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
from contextcanon.semantic import (
    ClaimCandidate,
    ExtractionContext,
    SemanticExtractor,
    normalize_candidate,
)
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
    _observation_fingerprints: set[str] = field(default_factory=set, init=False, repr=False)
    _materialized_keys: set[tuple[str, str]] = field(default_factory=set, init=False, repr=False)
    observation_failures: list[dict[str, object]] = field(default_factory=list)
    raw_observations_selected_for_materialization: int = 0
    raw_observations_skipped: int = 0
    lazy_extraction_requests: int = 0
    lazy_extraction_cache_hits: int = 0
    claims_materialized: int = 0
    fact_needs_count: int = 0

    @staticmethod
    def _observation_fingerprint(observation: RuntimeObservation) -> str:
        """Identify identical runtime observations without using call order."""
        payload = {
            "tool_name": observation.tool_name,
            "tool_kind": observation.tool_kind,
            "tool_parameters": observation.tool_parameters,
            "tool_result": observation.tool_result,
            "success": observation.success,
            "error": observation.error,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def capture_observation(self, observation: RuntimeObservation) -> tuple[str, bool]:
        """Preserve a raw observation before any semantic work is attempted."""
        self.observations.append(observation)
        if not observation.success:
            self.observation_failures.append({
                "tool_name": observation.tool_name,
                "call_index": observation.call_index,
                "error": observation.error,
            })
        fingerprint = self._observation_fingerprint(observation)
        if fingerprint in self._observation_fingerprints:
            return fingerprint, False
        self._observation_fingerprints.add(fingerprint)
        return fingerprint, True

    @staticmethod
    def _semantic_hint(needs: tuple[FactNeed, ...] | None) -> str | None:
        if not needs:
            return None
        return "; ".join(
            f"{need.subject_hint} / {need.semantic_dimension}"
            for need in needs
        )

    @staticmethod
    def _need_identity(needs: tuple[FactNeed, ...] | None) -> str:
        payload = [
            {
                "subject_hint": need.subject_hint,
                "semantic_dimension": need.semantic_dimension,
                "expected_value_type": need.expected_value_type,
                "temporal_scope": need.temporal_scope.value if need.temporal_scope else None,
                "required": need.required,
            }
            for need in (needs or ())
        ]
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _observation_text(observation: RuntimeObservation) -> str:
        return json.dumps(
            {
                "tool_name": observation.tool_name,
                "tool_kind": observation.tool_kind,
                "tool_parameters": observation.tool_parameters,
                "tool_result": observation.tool_result,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).casefold()

    @classmethod
    def _relevance_score(
        cls,
        observation: RuntimeObservation,
        needs: tuple[FactNeed, ...],
    ) -> int:
        text = cls._observation_text(observation)
        score = 0
        for need in needs:
            subject_tokens = set(need.subject_hint.casefold().split())
            dimension_tokens = set(need.semantic_dimension.casefold().split())
            score += 3 * sum(token in text for token in subject_tokens if len(token) > 2)
            score += sum(token in text for token in dimension_tokens if len(token) > 2)
            if observation.tool_kind == "verify":
                score += 1
        return score

    def materialize_observation(
        self,
        observation: RuntimeObservation,
        *,
        needs: tuple[FactNeed, ...] | None = None,
    ) -> list[EvidenceCandidate]:
        """Turn one preserved observation into governed evidence.

        This method is intentionally separate from capture so callers can run
        it in a task-local worker and keep the MCP read path responsive.
        """
        fingerprint = self._observation_fingerprint(observation)
        need_identity = self._need_identity(needs)
        cache_key = (fingerprint, need_identity)
        if cache_key in self._materialized_keys:
            self.lazy_extraction_cache_hits += 1
            return []
        self._materialized_keys.add(cache_key)
        self.lazy_extraction_requests += 1
        ingested: list[EvidenceCandidate] = []
        semantic_hint = self._semantic_hint(needs)
        if semantic_hint is None and isinstance(self.action_context, str):
            semantic_hint = self.action_context
        fact_needs = tuple(
            {
                "subject_hint": need.subject_hint,
                "semantic_dimension": need.semantic_dimension,
                "expected_value_type": need.expected_value_type,
                "temporal_scope": need.temporal_scope.value if need.temporal_scope else None,
                "required": need.required,
            }
            for need in (needs or ())
        )
        context = ExtractionContext(semantic_hint, fact_needs)
        for raw_candidate in self.extractor.extract(observation, context):
            candidate = normalize_candidate(raw_candidate)
            if candidate is None:
                continue
            evidence_candidate = _evidence_candidate(candidate)
            self.store.ingest(evidence_candidate)
            ingested.append(evidence_candidate)
        self.claims_materialized += len(ingested)
        return ingested

    def materialize_relevant_observations(
        self,
        needs: tuple[FactNeed, ...],
        *,
        candidate_limit: int = 8,
    ) -> list[EvidenceCandidate]:
        """Materialize only raw observations relevant to the current FactNeed."""
        self.fact_needs_count += len(needs)
        if not needs:
            return []
        candidates = [observation for observation in self.observations if observation.success]
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (-self._relevance_score(item[1], needs), item[1].call_index, item[0]),
        )
        positive = [item for item in ranked if self._relevance_score(item[1], needs) > 0]
        selected = (positive or ranked)[:candidate_limit]
        self.raw_observations_selected_for_materialization += len(selected)
        self.raw_observations_skipped += len(self.observations) - len(selected)
        ingested: list[EvidenceCandidate] = []
        for _index, observation in selected:
            ingested.extend(self.materialize_observation(observation, needs=needs))
        return ingested

    def observe(self, observation: RuntimeObservation) -> list[EvidenceCandidate]:
        """Compatibility helper for synchronous callers and unit tests."""
        _fingerprint, is_new = self.capture_observation(observation)
        if not is_new:
            return []
        return self.materialize_observation(observation)

    def extract_action_needs(self, proposed: ProposedToolCall) -> FactNeedExtractionResult:
        return self.fact_need_extractor.extract_for_action(ActionRequest(
            proposed.tool_name,
            proposed.parameters,
            context=self.action_context,
        ))

    def evaluate_needs(self, extraction: FactNeedExtractionResult) -> ActionGovernanceResult:
        if self.budget is not None and self.budget.governance_incomplete:
            return ActionGovernanceResult(ActionGovernanceDecision.REQUIRE_CLARIFICATION)
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
        return ActionGovernanceResult(ActionGovernanceDecision.REQUIRE_CLARIFICATION)

    def evaluate_action(self, proposed: ProposedToolCall) -> ActionGovernanceResult:
        if self.budget is not None and self.budget.governance_incomplete:
            return ActionGovernanceResult(ActionGovernanceDecision.REQUIRE_CLARIFICATION)
        return self.evaluate_needs(self.extract_action_needs(proposed))

    @property
    def metrics(self) -> dict[str, object]:
        metrics: dict[str, object] = {}
        if self.budget is not None:
            metrics.update(self.budget.diagnostics())
        if self.observation_failures:
            metrics["observation_failures"] = list(self.observation_failures)
        aligner = self.store.aligner
        relation = self.store.relation_classifier
        tool_semantics = self.tool_semantics_resolver
        core_requests = self.budget.requests_started if self.budget is not None else 0
        core_time_ms = self.budget.total_semantic_wall_ms if self.budget is not None else 0
        tool_budget = getattr(tool_semantics, "budget", None)
        tool_requests = tool_budget.requests_started if tool_budget is not None else 0
        tool_time_ms = tool_budget.total_semantic_wall_ms if tool_budget is not None else 0
        if tool_budget is not None:
            metrics["semantic_requests_total"] = core_requests + tool_requests
            metrics["semantic_requests_by_stage"] = {
                **(self.budget.requests_by_stage if self.budget is not None else {}),
                **tool_budget.requests_by_stage,
            }
            metrics["semantic_time_ms_total"] = core_time_ms + tool_time_ms
            metrics["semantic_time_ms_by_stage"] = {
                **(self.budget.time_by_stage_ms if self.budget is not None else {}),
                **tool_budget.time_by_stage_ms,
            }
            metrics["semantic_timeouts"] = (
                (self.budget.requests_timed_out if self.budget is not None else 0)
                + tool_budget.requests_timed_out
            )
            metrics["semantic_budget_exhausted"] = (
                (self.budget.requests_skipped_budget if self.budget is not None else 0)
                + tool_budget.requests_skipped_budget
            )
        metrics["core_governance_requests"] = core_requests
        metrics["tool_semantics_requests"] = tool_requests
        metrics["tool_semantics_budget_exhausted"] = getattr(
            tool_semantics, "budget_exhausted", 0
        )
        metrics["tool_semantics_prepare_model_requests"] = getattr(
            tool_semantics, "prepare_model_requests", 0
        )
        metrics.update({
            "raw_observations_total": len(self.observations),
            "raw_observations_selected_for_materialization": self.raw_observations_selected_for_materialization,
            "raw_observations_skipped": self.raw_observations_skipped,
            "lazy_extraction_requests": self.lazy_extraction_requests,
            "lazy_extraction_cache_hits": self.lazy_extraction_cache_hits,
            "claims_materialized": self.claims_materialized,
            "fact_needs_count": self.fact_needs_count,
            "fast_path_alignment_hits": getattr(aligner, "fast_path_hits", 0),
            "alignment_candidates_considered": getattr(
                aligner, "alignment_candidates_considered", 0
            ),
            "fast_path_relation_hits": getattr(relation, "fast_path_hits", 0),
            "tool_semantics_cache_hits": getattr(tool_semantics, "cache_hits", 0),
            "tool_semantics_cache_misses": getattr(tool_semantics, "cache_misses", 0),
            "semantic_cache_hits_by_stage": {
                "alignment": getattr(aligner, "cache_hits", 0),
                "relation": getattr(relation, "cache_hits", 0),
                "tool_semantics": getattr(tool_semantics, "cache_hits", 0),
            },
        })
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
