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
    AlignmentResult,
    EvidenceCandidate,
    FactDescriptor,
    FactNeed,
    GovernanceStore,
)


def _evidence(
    identifier: str,
    source: str,
    *,
    scope: TemporalScope = TemporalScope.CURRENT,
    verified: bool | None = True,
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
            valid_until=valid_until,
        ),
    )


class OpenWorldGovernanceTests(unittest.TestCase):
    def test_conflicting_evidence_diverges_without_business_schema(self) -> None:
        store = GovernanceStore()
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
        store.ingest(new, relation=EvidenceRelation.SUPERSEDING)

        self.assertEqual(governed.records[0].superseded_by, "new")
        self.assertEqual(governed.state, GovernanceState.RESOLVED)
        result = store.query(FactNeed(
            "Service maintenance",
            "when the maintenance window ends",
            temporal_scope=TemporalScope.CURRENT,
        ))
        self.assertIsNotNone(result)
        self.assertEqual(result.candidate_values, ("23:30",))

    def test_expired_evidence_becomes_stale_deterministically(self) -> None:
        store = GovernanceStore(now=datetime(2026, 3, 23, tzinfo=timezone.utc))
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

    def test_query_exposes_conflicting_values_and_sources(self) -> None:
        store = GovernanceStore()
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
        store = GovernanceStore()
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

        store = GovernanceStore(aligner=ParaphraseAligner())
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
            relation=EvidenceRelation.UNKNOWN,
        )
        self.assertEqual(governed.state, GovernanceState.AMBIGUOUS)


if __name__ == "__main__":
    unittest.main()
