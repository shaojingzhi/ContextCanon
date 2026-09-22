from __future__ import annotations

import unittest
from datetime import datetime, timezone

from contextcanon.core import (
    Evidence,
    EvidenceRelation,
    EvidenceRole,
    FactAlignment,
    GovernanceState,
    SourceType,
    TemporalScope,
)
from contextcanon.governance import (
    ActionGovernanceDecision,
    ActionRequest,
    AlignmentResult,
    DependencyAssessment,
    EvidenceCandidate,
    FactDescriptor,
    FactNeed,
    GovernanceStore,
    LLMEvidenceAligner,
    LLMFactNeedExtractor,
    LLMRelationClassifier,
)


def _evidence(
    identifier: str,
    source: str,
    *,
    scope: TemporalScope = TemporalScope.CURRENT,
    verified: bool | None = True,
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> Evidence:
    return Evidence(
        id=identifier,
        source_id=source,
        source_type=SourceType.OTHER,
        location="result",
        role=EvidenceRole.OBSERVED,
        verified=verified,
        temporal_scope=scope,
        confidence=0.95,
        valid_from=valid_from,
        valid_until=valid_until,
    )


def _candidate(
    identifier: str,
    source: str,
    value: str,
    *,
    subject: str = "Spring Gala",
    dimension: str = "the date when the event takes place",
    scope: TemporalScope = TemporalScope.CURRENT,
    verified: bool | None = True,
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> EvidenceCandidate:
    fact = FactDescriptor(subject, dimension, scope)
    return EvidenceCandidate(
        fact,
        value,
        _evidence(
            identifier,
            source,
            scope=scope,
            verified=verified,
            valid_from=valid_from,
            valid_until=valid_until,
        ),
    )


class ConflictClassifier:
    def classify(self, incoming, existing):
        return EvidenceRelation.CONFLICTING


class CompatibleClassifier:
    def classify(self, incoming, existing):
        return EvidenceRelation.COMPATIBLE


class FakeSemanticClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, model: str):
        self.prompts.append(prompt)
        return next(self.responses)


class MutableClock:
    def __init__(self, current: datetime):
        self.current = current

    def __call__(self) -> datetime:
        return self.current


class OpenWorldGovernanceTests(unittest.TestCase):
    def test_llm_fact_need_extractor_returns_ephemeral_fact_need(self) -> None:
        client = FakeSemanticClient([{
            "assessment": "HAS_DEPENDENCIES",
            "needs": [{
            "subject_hint": "Spring Gala",
            "semantic_dimension": "the date when the event takes place",
            "expected_value_type": "date",
            "temporal_scope": "CURRENT",
            "required": True,
        }]}])
        extractor = LLMFactNeedExtractor(client)
        result = extractor.extract_for_action(ActionRequest(
            "messages.send",
            {
                "text": "Spring Gala will take place on March 22",
                "gold_answer": "must-not-leak",
            },
        ))
        self.assertEqual(result.assessment, DependencyAssessment.HAS_DEPENDENCIES)
        self.assertEqual(len(result.needs), 1)
        self.assertEqual(result.needs[0].subject_hint, "Spring Gala")
        self.assertNotIn("gold_answer", client.prompts[0])
        self.assertNotIn("must-not-leak", client.prompts[0])

    def test_fact_need_extractor_preserves_empty_has_dependencies(self) -> None:
        client = FakeSemanticClient([{
            "assessment": "HAS_DEPENDENCIES",
            "needs": [],
        }])
        result = LLMFactNeedExtractor(client).extract_for_action(
            ActionRequest("messages.send", {"text": "External fact"})
        )
        self.assertEqual(result.assessment, DependencyAssessment.HAS_DEPENDENCIES)
        self.assertEqual(result.needs, ())

    def test_fact_need_extractor_returns_unknown_for_malformed_output(self) -> None:
        result = LLMFactNeedExtractor(FakeSemanticClient([{"needs": "bad"}])).extract_for_action(
            ActionRequest("messages.send", {})
        )
        self.assertEqual(result.assessment, DependencyAssessment.UNKNOWN)

    def test_fact_need_extractor_accepts_explicit_no_dependencies(self) -> None:
        client = FakeSemanticClient([{
            "assessment": "NO_DEPENDENCIES",
            "needs": [],
        }])
        result = LLMFactNeedExtractor(client).extract_for_action(
            ActionRequest("healthcheck.ping", {})
        )
        self.assertEqual(result.assessment, DependencyAssessment.NO_DEPENDENCIES)
        self.assertEqual(result.needs, ())

    def test_fact_need_extractor_rejects_inconsistent_no_dependencies(self) -> None:
        client = FakeSemanticClient([{
            "assessment": "NO_DEPENDENCIES",
            "needs": [{
                "subject_hint": "Deployment",
                "semantic_dimension": "window",
                "expected_value_type": "time",
                "temporal_scope": "CURRENT",
                "required": True,
            }],
        }])
        result = LLMFactNeedExtractor(client).extract_for_action(
            ActionRequest("production.deploy", {})
        )
        self.assertEqual(result.assessment, DependencyAssessment.UNKNOWN)

    def test_llm_aligner_handles_semantic_paraphrases(self) -> None:
        client = FakeSemanticClient([
            {"relation": "SAME_FACT", "confidence": 0.94},
        ])
        aligner = LLMEvidenceAligner(client)
        incoming = _candidate(
            "e2", "calendar-b", "2026-03-23",
            dimension="the day on which the event takes place",
        )
        existing = FactDescriptor(
            "Spring Gala", "scheduled date", TemporalScope.CURRENT
        )
        result = aligner.align(incoming, existing)
        self.assertEqual(result.relation, FactAlignment.SAME_FACT)

    def test_llm_aligner_low_confidence_is_unknown(self) -> None:
        client = FakeSemanticClient([
            {"relation": "SAME_FACT", "confidence": 0.4},
        ])
        result = LLMEvidenceAligner(client).align(
            _candidate("e2", "source-b", "two"),
            FactDescriptor("Spring Gala", "scheduled date", TemporalScope.CURRENT),
        )
        self.assertEqual(result.relation, FactAlignment.UNKNOWN)

    def test_conflicting_evidence_diverges_without_business_schema(self) -> None:
        store = GovernanceStore(relation_classifier=ConflictClassifier())
        store.ingest(_candidate("e1", "calendar-a", "2026-03-22"))
        governed = store.ingest(_candidate("e2", "calendar-b", "2026-03-23"))

        self.assertEqual(governed.state, GovernanceState.DIVERGED)
        self.assertEqual(len(store.facts), 1)

    def test_current_runtime_and_future_intent_are_distinct(self) -> None:
        store = GovernanceStore()
        current = _candidate(
            "runtime",
            "production-config",
            "JWT",
            subject="production authentication system",
            dimension="authentication protocol",
            scope=TemporalScope.CURRENT,
        )
        future = _candidate(
            "intent",
            "accepted-adr",
            "OAuth2",
            subject="production authentication system",
            dimension="authentication protocol",
            scope=TemporalScope.FUTURE,
        )
        store.ingest(current)
        store.ingest(future)

        self.assertEqual(len(store.facts), 2)
        self.assertEqual(
            {fact.state for fact in store.facts},
            {GovernanceState.RESOLVED},
        )

    def test_different_values_are_not_deterministically_conflicting(self) -> None:
        store = GovernanceStore()
        first = _candidate(
            "python", "service-doc-a", "Python",
            subject="Example service",
            dimension="supported programming languages",
        )
        second = _candidate(
            "java", "service-doc-b", "Java",
            subject="Example service",
            dimension="supported programming languages",
        )
        store.ingest(first)
        governed = store.ingest(second)
        self.assertEqual(governed.state, GovernanceState.AMBIGUOUS)
        self.assertNotEqual(governed.state, GovernanceState.DIVERGED)

    def test_llm_relation_classifier_can_mark_values_compatible(self) -> None:
        client = FakeSemanticClient([
            {"relation": "COMPATIBLE", "confidence": 0.93},
        ])
        classifier = LLMRelationClassifier(client)
        relation = classifier.classify(
            _candidate("java", "service-doc-b", "Java"),
            _candidate("python", "service-doc-a", "Python"),
        )
        self.assertEqual(relation, EvidenceRelation.COMPATIBLE)

    def test_llm_relation_classifier_low_confidence_is_unknown(self) -> None:
        client = FakeSemanticClient([
            {"relation": "CONFLICTING", "confidence": 0.3},
        ])
        relation = LLMRelationClassifier(client).classify(
            _candidate("e2", "calendar-b", "2026-03-23"),
            _candidate("e1", "calendar-a", "2026-03-22"),
        )
        self.assertEqual(relation, EvidenceRelation.UNKNOWN)

    def test_llm_relation_classifier_recognizes_true_conflict(self) -> None:
        client = FakeSemanticClient([
            {"relation": "CONFLICTING", "confidence": 0.96},
        ])
        store = GovernanceStore(relation_classifier=LLMRelationClassifier(client))
        store.ingest(_candidate("e1", "calendar-a", "2026-03-22"))
        governed = store.ingest(_candidate("e2", "calendar-b", "2026-03-23"))
        self.assertEqual(governed.state, GovernanceState.DIVERGED)

    def test_new_evidence_can_supersede_old_evidence(self) -> None:
        store = GovernanceStore()
        old = _candidate(
            "old",
            "maintenance-notice-v1",
            "22:00",
            subject="Service maintenance",
            dimension="when the maintenance window ends",
        )
        new = _candidate(
            "new",
            "maintenance-notice-v2",
            "23:30",
            subject="Service maintenance",
            dimension="when the maintenance window ends",
        )
        governed = store.ingest(old)
        store.ingest(
            new,
            relation_overrides={"old": EvidenceRelation.SUPERSEDING},
        )

        self.assertEqual(governed.records[0].superseded_by, "new")
        self.assertEqual(governed.state, GovernanceState.RESOLVED)
        result = store.query(FactNeed(
            "Service maintenance",
            "when the maintenance window ends",
            temporal_scope=TemporalScope.CURRENT,
        ))
        self.assertIsNotNone(result)
        self.assertEqual(result.candidate_values, ("23:30",))

    def test_llm_relation_classifier_can_supersede_old_evidence(self) -> None:
        client = FakeSemanticClient([
            {"relation": "SUPERSEDING", "confidence": 0.97},
        ])
        store = GovernanceStore(relation_classifier=LLMRelationClassifier(client))
        old = _candidate(
            "old", "notice-v1", "22:00",
            subject="Service maintenance",
            dimension="when the maintenance window ends",
        )
        new = _candidate(
            "new", "notice-v2", "23:30",
            subject="Service maintenance",
            dimension="when the maintenance window ends",
        )
        governed = store.ingest(old)
        store.ingest(new)
        self.assertEqual(governed.records[0].superseded_by, "new")
        self.assertEqual(governed.state, GovernanceState.RESOLVED)

    def test_superseding_override_is_pairwise(self) -> None:
        store = GovernanceStore(relation_classifier=CompatibleClassifier())
        for identifier, value in (("a", "22:00"), ("b", "22:15"), ("c", "22:30")):
            store.ingest(_candidate(
                identifier,
                f"notice-{identifier}",
                value,
                subject="Service maintenance",
                dimension="when the maintenance window ends",
            ))
        store.ingest(
            _candidate(
                "new", "updated-notice", "23:30",
                subject="Service maintenance",
                dimension="when the maintenance window ends",
            ),
            relation_overrides={"a": EvidenceRelation.SUPERSEDING},
        )
        records = store.facts[0].records
        self.assertEqual(records[0].superseded_by, "new")
        self.assertIsNone(records[1].superseded_by)
        self.assertIsNone(records[2].superseded_by)

    def test_expired_evidence_becomes_stale_deterministically(self) -> None:
        store = GovernanceStore(
            clock=MutableClock(datetime(2026, 3, 23, tzinfo=timezone.utc))
        )
        governed = store.ingest(_candidate(
            "expired",
            "status-page",
            "available",
            subject="Example service",
            dimension="current availability",
            valid_until="2026-03-22T23:59:59Z",
        ))
        self.assertEqual(governed.state, GovernanceState.STALE)
        result = store.query(FactNeed(
            "Example service",
            "current availability",
            temporal_scope=TemporalScope.CURRENT,
        ))
        self.assertIsNotNone(result)
        self.assertEqual(result.state, GovernanceState.STALE)

    def test_clock_advancement_changes_active_evidence_to_stale(self) -> None:
        clock = MutableClock(datetime(2026, 3, 22, 12, tzinfo=timezone.utc))
        store = GovernanceStore(clock=clock)
        governed = store.ingest(_candidate(
            "window", "status-page", "available",
            subject="Example service",
            dimension="current availability",
            valid_until="2026-03-22T13:00:00Z",
        ))
        self.assertEqual(governed.state, GovernanceState.RESOLVED)
        clock.current = datetime(2026, 3, 22, 14, tzinfo=timezone.utc)
        result = store.query(FactNeed(
            "Example service", "current availability",
            temporal_scope=TemporalScope.CURRENT,
        ))
        self.assertEqual(result.state, GovernanceState.STALE)

    def test_future_valid_from_is_not_active_today(self) -> None:
        clock = MutableClock(datetime(2026, 3, 22, 12, tzinfo=timezone.utc))
        store = GovernanceStore(clock=clock)
        governed = store.ingest(_candidate(
            "future", "release-plan", "enabled",
            subject="Example feature",
            dimension="current availability",
            valid_from="2026-03-23T00:00:00Z",
        ))
        self.assertEqual(governed.state, GovernanceState.STALE)

    def test_verified_is_orthogonal_to_resolved_state(self) -> None:
        candidate = _candidate(
            "unverified", "runtime-observation", "available",
            subject="Example service",
            dimension="current availability",
            verified=None,
        )
        governed = GovernanceStore().ingest(candidate)
        self.assertEqual(governed.state, GovernanceState.RESOLVED)
        self.assertIsNone(governed.records[0].evidence.verified)

    def test_query_exposes_conflicting_values_and_sources(self) -> None:
        store = GovernanceStore(relation_classifier=ConflictClassifier())
        store.ingest(_candidate("e1", "calendar-a", "2026-03-22"))
        store.ingest(_candidate("e2", "calendar-b", "2026-03-23"))

        result = store.query(FactNeed(
            "Spring Gala",
            "the date when the event takes place",
            expected_value_type="date",
            temporal_scope=TemporalScope.CURRENT,
        ))
        self.assertIsNotNone(result)
        self.assertEqual(result.state, GovernanceState.DIVERGED)
        self.assertEqual(result.candidate_values, ("2026-03-22", "2026-03-23"))
        self.assertEqual(result.sources, ("calendar-a", "calendar-b"))

    def test_protected_action_uses_the_same_governance_state(self) -> None:
        store = GovernanceStore(relation_classifier=ConflictClassifier())
        store.ingest(_candidate("e1", "calendar-a", "2026-03-22"))
        store.ingest(_candidate("e2", "calendar-b", "2026-03-23"))
        need = FactNeed(
            "Spring Gala",
            "the date when the event takes place",
            "date",
            TemporalScope.CURRENT,
        )

        decision = store.evaluate_action([need])
        self.assertEqual(
            decision.decision,
            ActionGovernanceDecision.REQUIRE_CLARIFICATION,
        )
        self.assertEqual(decision.unresolved[0].state, GovernanceState.DIVERGED)

    def test_semantic_aligner_is_a_separate_injected_boundary(self) -> None:
        class ParaphraseAligner:
            def align(self, incoming, existing_fact):
                if incoming.fact.subject == existing_fact.subject:
                    return AlignmentResult(FactAlignment.SAME_FACT, 0.9)
                return AlignmentResult(FactAlignment.UNRELATED, 1.0)

        store = GovernanceStore(
            aligner=ParaphraseAligner(),
            relation_classifier=ConflictClassifier(),
        )
        first = _candidate(
            "e1", "calendar-a", "2026-03-22",
            dimension="scheduled date",
        )
        second = _candidate(
            "e2", "calendar-b", "2026-03-23",
            dimension="the day on which the event occurs",
        )
        store.ingest(first)
        governed = store.ingest(second)

        self.assertEqual(len(store.facts), 1)
        self.assertEqual(governed.state, GovernanceState.DIVERGED)

    def test_alignment_falls_back_to_bounded_recent_facts_without_token_overlap(self) -> None:
        class CountingAligner:
            def __init__(self):
                self.calls = 0

            def align(self, incoming, existing_fact):
                self.calls += 1
                if (
                    incoming.fact.subject == "production authentication"
                    and existing_fact.subject == "login service"
                ):
                    return AlignmentResult(FactAlignment.SAME_FACT, 0.95)
                return AlignmentResult(FactAlignment.UNRELATED, 0.95)

        aligner = CountingAligner()
        store = GovernanceStore(aligner=aligner, alignment_candidate_limit=2)
        store.ingest(_candidate(
            "login", "config", "JWT",
            subject="login service",
            dimension="authentication protocol",
        ))
        store.ingest(_candidate(
            "billing", "billing-config", "enabled",
            subject="billing service",
            dimension="availability",
        ))
        store.ingest(_candidate(
            "incoming", "runtime", "JWT",
            subject="production authentication",
            dimension="login mechanism",
        ))

        self.assertEqual(len(store.facts), 2)
        self.assertLessEqual(aligner.calls, 3)

    def test_unknown_alignment_does_not_aggressively_merge(self) -> None:
        class UnknownAligner:
            def align(self, incoming, existing_fact):
                return AlignmentResult(FactAlignment.UNKNOWN, 0.4)

        store = GovernanceStore(aligner=UnknownAligner())
        store.ingest(_candidate("e1", "source-a", "one"))
        store.ingest(_candidate("e2", "source-b", "two"))
        self.assertEqual(len(store.facts), 2)

    def test_unknown_relation_preserves_ambiguity(self) -> None:
        store = GovernanceStore()
        store.ingest(_candidate("e1", "source-a", "one"))
        governed = store.ingest(
            _candidate("e2", "source-b", "two"),
            relation_overrides={"e1": EvidenceRelation.UNKNOWN},
        )
        self.assertEqual(governed.state, GovernanceState.AMBIGUOUS)


if __name__ == "__main__":
    unittest.main()
