from __future__ import annotations

import time
import unittest

from contextcanon.core import EvidenceRelation, FactAlignment, TemporalScope
from contextcanon.governance import (
    ActionGovernanceDecision,
    EvidenceCandidate,
    FactDescriptor,
    LLMEvidenceAligner,
    LLMRelationClassifier,
)
from contextcanon.semantic_budget import (
    BudgetedSemanticClient,
    SemanticBudget,
    SemanticDeadlineExceeded,
)
from contextcanon.tool_semantics import LLMToolSemanticsResolver, ToolSemantics
from experiments.agentabstain.adapter import ProposedToolCall
from experiments.agentabstain.runtime_governance import RuntimeGovernance

from test_governance import FakeSemanticClient, _candidate


class SemanticBudgetTests(unittest.TestCase):
    def test_identical_facts_use_alignment_fast_path(self) -> None:
        client = FakeSemanticClient([])
        aligner = LLMEvidenceAligner(client)
        result = aligner.align(
            _candidate("one", "a", "2026-03-22"),
            FactDescriptor("Spring Gala", "the date when the event takes place", TemporalScope.CURRENT),
        )
        self.assertEqual(result.relation, FactAlignment.SAME_FACT)
        self.assertEqual(client.prompts, [])
        self.assertEqual(aligner.fast_path_hits, 1)

    def test_same_value_uses_relation_fast_path(self) -> None:
        client = FakeSemanticClient([])
        classifier = LLMRelationClassifier(client)
        relation = classifier.classify(
            _candidate("one", "a", "2026-03-22"),
            _candidate("two", "b", "2026-03-22"),
        )
        self.assertEqual(relation, EvidenceRelation.SUPPORTING)
        self.assertEqual(client.prompts, [])
        self.assertEqual(classifier.fast_path_hits, 1)

    def test_different_values_still_use_semantic_relation(self) -> None:
        client = FakeSemanticClient([{"relation": "COMPATIBLE", "confidence": 0.9}])
        relation = LLMRelationClassifier(client).classify(
            _candidate("one", "a", "Python"),
            _candidate("two", "b", "Java"),
        )
        self.assertEqual(relation, EvidenceRelation.COMPATIBLE)
        self.assertEqual(len(client.prompts), 1)

    def test_tool_semantics_cache_reuses_identical_metadata(self) -> None:
        client = FakeSemanticClient([{"semantics": "READ", "confidence": 0.95}])
        resolver = LLMToolSemanticsResolver(client)
        metadata = {"readOnlyHint": False}
        self.assertEqual(resolver.classify("filesystem.read", description="read", annotations=metadata), ToolSemantics.READ)
        self.assertEqual(resolver.classify("filesystem.read", description="read", annotations=metadata), ToolSemantics.READ)
        self.assertEqual(len(client.prompts), 1)
        self.assertEqual(resolver.cache_hits, 1)
        self.assertEqual(resolver.cache_misses, 1)

    def test_request_count_budget_skips_third_operation(self) -> None:
        budget = SemanticBudget(max_requests=2, max_total_seconds=10, per_request_deadline_seconds=1)
        client = FakeSemanticClient(["one", "two", "three"])
        wrapped = BudgetedSemanticClient(client, budget, "alignment")
        self.assertEqual(wrapped.complete("", model="m"), "one")
        self.assertEqual(wrapped.complete("", model="m"), "two")
        with self.assertRaises(SemanticDeadlineExceeded):
            wrapped.complete("", model="m")
        self.assertEqual(len(client.prompts), 2)
        self.assertTrue(budget.governance_incomplete)
        self.assertEqual(budget.requests_skipped_budget, 1)

    def test_semantic_stages_share_one_task_budget(self) -> None:
        budget = SemanticBudget(max_requests=2, max_total_seconds=10, per_request_deadline_seconds=1)
        extraction = BudgetedSemanticClient(FakeSemanticClient(["one"]), budget, "extraction")
        alignment = BudgetedSemanticClient(FakeSemanticClient(["two"]), budget, "alignment")
        self.assertIs(extraction.budget, alignment.budget)

    def test_total_budget_is_checked_before_new_request(self) -> None:
        now = [0.0]
        budget = SemanticBudget(
            max_requests=3,
            max_total_seconds=2,
            per_request_deadline_seconds=1,
            clock=lambda: now[0],
        )
        client = FakeSemanticClient(["one", "two"])
        calls: list[str] = []
        def complete(prompt: str, *, model: str):
            calls.append(prompt)
            now[0] += 2.0
            return "one"
        client.complete = complete  # type: ignore[method-assign]
        wrapped = BudgetedSemanticClient(client, budget, "extraction")
        self.assertEqual(wrapped.complete("", model="m"), "one")
        now[0] += 100.0
        with self.assertRaises(SemanticDeadlineExceeded):
            wrapped.complete("", model="m")
        self.assertEqual(len(calls), 1)
        self.assertTrue(budget.governance_incomplete)

    def test_idle_time_outside_semantic_operations_does_not_consume_budget(self) -> None:
        now = [0.0]
        budget = SemanticBudget(
            max_requests=3,
            max_total_seconds=10,
            per_request_deadline_seconds=5,
            clock=lambda: now[0],
        )
        client = FakeSemanticClient([])
        def complete(prompt: str, *, model: str):
            now[0] += 2.0
            return "ok"
        client.complete = complete  # type: ignore[method-assign]
        wrapped = BudgetedSemanticClient(client, budget, "alignment")
        self.assertEqual(wrapped.complete("", model="m"), "ok")
        now[0] += 100.0
        self.assertAlmostEqual(budget.remaining_time_budget, 8.0, places=2)
        self.assertEqual(wrapped.complete("", model="m"), "ok")
        self.assertAlmostEqual(budget.remaining_time_budget, 6.0, places=2)

    def test_cumulative_semantic_time_consumes_budget(self) -> None:
        now = [0.0]
        budget = SemanticBudget(
            max_requests=3,
            max_total_seconds=5,
            per_request_deadline_seconds=5,
            clock=lambda: now[0],
        )
        client = FakeSemanticClient([])
        def complete(prompt: str, *, model: str):
            now[0] += 2.0
            return "ok"
        client.complete = complete  # type: ignore[method-assign]
        wrapped = BudgetedSemanticClient(client, budget, "relation")
        wrapped.complete("", model="m")
        wrapped.complete("", model="m")
        self.assertAlmostEqual(budget.remaining_time_budget, 1.0, places=2)

    def test_operation_deadline_returns_control_without_worker(self) -> None:
        budget = SemanticBudget(max_requests=1, max_total_seconds=2, per_request_deadline_seconds=0.05)
        client = FakeSemanticClient([])

        def never_finishes(prompt: str, *, model: str):
            time.sleep(1)
            return "never"

        client.complete = never_finishes  # type: ignore[method-assign]
        started = time.monotonic()
        with self.assertRaises(SemanticDeadlineExceeded):
            BudgetedSemanticClient(client, budget, "alignment").complete("", model="m")
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(budget.requests_timed_out, 1)
        self.assertTrue(budget.governance_incomplete)

    def test_timeout_consumes_total_semantic_budget(self) -> None:
        budget = SemanticBudget(max_requests=2, max_total_seconds=10, per_request_deadline_seconds=0.05)
        client = FakeSemanticClient([])

        def never_finishes(prompt: str, *, model: str):
            time.sleep(1)
            return "never"

        client.complete = never_finishes  # type: ignore[method-assign]
        with self.assertRaises(SemanticDeadlineExceeded):
            BudgetedSemanticClient(client, budget, "extraction").complete("", model="m")
        self.assertGreaterEqual(budget.total_semantic_wall_ms, 40)
        self.assertLess(budget.remaining_time_budget, 9.99)
        self.assertEqual(budget.requests_timed_out, 1)

    def test_incomplete_governance_requires_clarification(self) -> None:
        budget = SemanticBudget(max_requests=0)
        governance = RuntimeGovernance(
            extractor=object(),  # not reached by the barrier
            fact_need_extractor=object(),
            store=object(),
            budget=budget,
        )
        budget.governance_incomplete = True
        result = governance.evaluate_action(ProposedToolCall("deploy", "commit", {}))
        self.assertEqual(result.decision, ActionGovernanceDecision.REQUIRE_CLARIFICATION)


if __name__ == "__main__":
    unittest.main()
