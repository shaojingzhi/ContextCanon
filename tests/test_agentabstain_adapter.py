from __future__ import annotations

import unittest

from experiments.agentabstain.adapter import (
    AgentAbstainAdapter,
    GuardDecision,
    ProposedToolCall,
    RuntimeObservation,
)


TASKS = {
    "preview_008": ("phone_and_messages.send_phone_message", {"date": "2026-03-22"}, {"date": "2026-03-23"}),
    "preview_013": ("retail_orders.manage_returns_and_exchanges", {"status": "delivered"}, {"status": "returned"}),
    "preview_015": ("disaster_relief_operations.publish_community_announcement", {"status": "closed"}, {"status": "open"}),
}


def _adapter(task: str, first: object, second: object, *, success: bool = True) -> AgentAbstainAdapter:
    adapter = AgentAbstainAdapter()
    names = {
        "preview_008": ("event_system.event_search", "event_system.event_verifier"),
        "preview_013": ("retail_orders.get_order", "verify_order_summary_document"),
        "preview_015": ("disaster_relief_operations.get_district_situation", "document_authoring_and_publication.lookup_external_records"),
    }
    first_name, second_name = names[task]
    adapter.observe_tool_result(RuntimeObservation(first_name, "lookup", {}, first, success, 0))
    adapter.observe_tool_result(RuntimeObservation(second_name, "verify", {}, second, True, 1))
    return adapter


class AgentAbstainAdapterTests(unittest.TestCase):
    def test_all_tasks_allow_consistent_and_require_clarification_on_conflict(self) -> None:
        for task, (commit, consistent, conflicting) in TASKS.items():
            with self.subTest(task=task):
                consistent_adapter = _adapter(task, consistent, consistent)
                self.assertEqual(
                    consistent_adapter.evaluate_proposed_commit(ProposedToolCall(commit, "commit")),
                    GuardDecision.ALLOW,
                )
                conflict_adapter = _adapter(task, consistent, conflicting)
                self.assertEqual(
                    conflict_adapter.evaluate_proposed_commit(ProposedToolCall(commit, "commit")),
                    GuardDecision.REQUIRE_CLARIFICATION,
                )

    def test_observation_provenance_is_preserved(self) -> None:
        adapter = _adapter("preview_008", {"date": "2026-03-22"}, {"date": "2026-03-23"})
        self.assertEqual(len(adapter.ledger.claims), 2)
        self.assertEqual(
            adapter.ledger.claims[0].evidence.source_id,
            "event_system.event_search",
        )
        self.assertEqual(adapter.ledger.claims[0].evidence.location, "call:0:date")

    def test_plain_text_event_dates_produce_a_conflict(self) -> None:
        adapter = AgentAbstainAdapter()
        adapter.observe_tool_result(
            RuntimeObservation(
                "filesystem.read",
                "lookup",
                {"path": "event-info.txt"},
                "Riverside Community Hall Spring Gala\nDate: March 22, 2026",
                True,
                0,
            )
        )
        adapter.observe_tool_result(
            RuntimeObservation(
                "event_system.event_search",
                "lookup",
                {},
                "Riverside Community Hall Spring Gala is scheduled for March 23, 2026.",
                True,
                1,
            )
        )
        self.assertEqual([claim.value for claim in adapter.ledger.claims], ["2026-03-22", "2026-03-23"])
        self.assertEqual(len(adapter.ledger.conflicts()), 1)

    def test_shelter_extraction_uses_the_named_entity(self) -> None:
        adapter = AgentAbstainAdapter()
        result = {
            "shelters": [
                {"name": "Harborview Middle School", "status": "open"},
                {"name": "Bayview Civic Center", "status": "open"},
                {"name": "Seaside Church Hall", "status": "closed"},
            ]
        }
        adapter.observe_tool_result(
            RuntimeObservation(
                "disaster_relief_operations.get_district_situation",
                "lookup",
                {},
                result,
                True,
                0,
            )
        )
        self.assertEqual(len(adapter.ledger.claims), 1)
        claim = adapter.ledger.claims[0]
        self.assertEqual((claim.subject, claim.predicate, claim.value), ("shelter/Seaside Church Hall", "status", "closed"))

    def test_failed_tool_call_is_not_evidence(self) -> None:
        adapter = _adapter("preview_013", {"status": "delivered"}, {"status": "returned"}, success=False)
        self.assertEqual([claim.value for claim in adapter.ledger.claims], ["returned"])
        self.assertEqual(adapter.ledger.conflicts(), [])

    def test_unrelated_commit_is_not_blocked(self) -> None:
        adapter = _adapter("preview_015", {"status": "closed"}, {"status": "open"})
        decision = adapter.evaluate_proposed_commit(ProposedToolCall("other.commit", "commit"))
        self.assertEqual(decision, GuardDecision.ALLOW)

    def test_no_majority_voting(self) -> None:
        adapter = AgentAbstainAdapter()
        for index, value in enumerate(("2026-03-22", "2026-03-22", "2026-03-23")):
            adapter.observe_tool_result(RuntimeObservation(f"event_system.source_{index}", "lookup", {}, {"date": value}, True, index))
        self.assertEqual(len(adapter.ledger.conflicts()), 1)

    def test_forbidden_benchmark_fields_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeObservation.from_mapping({
                "tool_name": "source",
                "tool_kind": "lookup",
                "tool_result": {"status": "open"},
                "call_index": 0,
                "task_type": "abstain",
            })

    def test_governance_does_not_accept_hidden_task_values(self) -> None:
        adapter = AgentAbstainAdapter()
        adapter.observe_tool_result(RuntimeObservation("retail_orders.get_order", "lookup", {}, {"status": "delivered"}, True, 0))
        self.assertEqual(adapter.ledger.conflicts(), [])
        self.assertNotIn("task_type", RuntimeObservation.__dataclass_fields__)
        self.assertNotIn("abstention_trigger", RuntimeObservation.__dataclass_fields__)

    def test_rendering_contains_evidence_not_benchmark_labels(self) -> None:
        adapter = _adapter("preview_015", {"status": "closed"}, {"status": "open"})
        rendered = adapter.render_governed_evidence()
        self.assertIn("Runtime evidence conflict", rendered)
        self.assertNotIn("abstain", rendered.lower())
        self.assertNotIn("gold", rendered.lower())


if __name__ == "__main__":
    unittest.main()

