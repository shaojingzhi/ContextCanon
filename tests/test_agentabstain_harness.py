from __future__ import annotations

import unittest

from contextcanon.core import EvidenceRelation, FactAlignment, TemporalScope
from contextcanon.governance import (
    AlignmentResult,
    FactNeed,
    GovernanceState,
    GovernanceStore,
)
from contextcanon.semantic import ClaimCandidate, Provenance
from experiments.agentabstain.adapter import (
    AgentAbstainAdapter,
    GuardDecision,
    RuntimeObservation,
)
from experiments.agentabstain.harness import RuntimeMCPBridge
from experiments.agentabstain.runtime_governance import RuntimeGovernance


class FakeMCP:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        return {"result": {"status": "ok"}}


class ErrorMCP(FakeMCP):
    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        return {"isError": True, "structuredContent": {"date": "2026-03-22"}}


class HarnessBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_generic_runtime_blocks_when_fact_need_extraction_fails(self) -> None:
        class EmptyExtractor:
            def extract(self, observation, context=None):
                return []

        class FailedNeedExtractor:
            last_failed = True

            def extract_for_action(self, action):
                return []

        fake = FakeMCP()
        governance = RuntimeGovernance(
            extractor=EmptyExtractor(),
            fact_need_extractor=FailedNeedExtractor(),
            store=GovernanceStore(),
        )
        bridge = RuntimeMCPBridge(
            fake.call_tool,
            {"message.send": "commit"},
            condition="guard",
            governance=governance,
        )
        result = await bridge.call_tool("message.send", {"text": "status update"})
        self.assertEqual(fake.calls, [])
        self.assertIn("Clarification is required", result["structuredContent"]["message"])

    async def test_generic_runtime_ingests_governs_and_blocks_action(self) -> None:
        class Extractor:
            def extract(self, observation, context=None):
                dimensions = (
                    "scheduled date",
                    "the day on which the workshop takes place",
                )
                return [ClaimCandidate(
                    entity="Community workshop",
                    property=dimensions[observation.call_index],
                    value=observation.tool_result["date"],
                    value_type="date",
                    role="OBSERVED",
                    confidence=0.95,
                    provenance=Provenance(
                        f"observation-{observation.call_index}",
                        observation.tool_name,
                        observation.tool_parameters,
                        "date",
                        observation.tool_result["date"],
                    ),
                    temporal_scope=TemporalScope.CURRENT.value,
                )]

        class Aligner:
            def align(self, incoming, existing_fact):
                return AlignmentResult(FactAlignment.SAME_FACT, 0.95)

        class Classifier:
            def classify(self, incoming, existing):
                return EvidenceRelation.CONFLICTING

        class NeedExtractor:
            def extract_for_action(self, action):
                return [FactNeed(
                    "Community workshop",
                    "scheduled date",
                    "date",
                    TemporalScope.CURRENT,
                )]

        responses = iter((
            {"result": {"date": "2026-04-10"}},
            {"result": {"date": "2026-04-11"}},
        ))

        async def call_tool(name: str, arguments: dict) -> dict:
            if name == "message.send":
                self.fail("blocked commit must not reach the runtime")
            return next(responses)

        governance = RuntimeGovernance(
            extractor=Extractor(),
            fact_need_extractor=NeedExtractor(),
            store=GovernanceStore(
                aligner=Aligner(),
                relation_classifier=Classifier(),
            ),
        )
        bridge = RuntimeMCPBridge(
            call_tool,
            {"source.a": "lookup", "source.b": "verify", "message.send": "commit"},
            condition="guard",
            governance=governance,
        )
        await bridge.call_tool("source.a", {})
        await bridge.call_tool("source.b", {})
        result = await bridge.call_tool("message.send", {"text": "Workshop is April 10"})

        self.assertEqual(governance.store.facts[0].state, GovernanceState.DIVERGED)
        self.assertEqual(bridge.diagnostics.conflicts_detected, 1)
        self.assertFalse(bridge.diagnostics.commit_dispatched)
        self.assertEqual(
            bridge.diagnostics.guard_decision,
            GuardDecision.REQUIRE_CLARIFICATION.value,
        )
        self.assertIn("Clarification is required", result["structuredContent"]["message"])
        rendered = bridge.governed_evidence()
        self.assertIn("State: DIVERGED", rendered)
        self.assertIn("2026-04-10", rendered)
        self.assertIn("source.a", rendered)

    async def test_lookup_and_verify_results_are_forwarded(self) -> None:
        fake = FakeMCP()
        bridge = RuntimeMCPBridge(
            fake.call_tool,
            {"event_system.event_search": "lookup"},
            condition="governed",
        )
        await bridge.call_tool(
            "event_system.event_search",
            {},
        )
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(bridge.diagnostics.observations_seen, 1)

    async def test_mcp_is_error_is_not_claim(self) -> None:
        bridge = RuntimeMCPBridge(
            ErrorMCP().call_tool,
            {"industrial_and_infrastructure_control.event_verifier": "verify"},
            condition="governed",
        )
        result = await bridge.call_tool("industrial_and_infrastructure_control.event_verifier", {})
        self.assertTrue(result["isError"])
        self.assertEqual(bridge.diagnostics.observations_seen, 1)
        self.assertEqual(bridge.adapter.ledger.claims, [])

    async def test_governed_result_contains_runtime_evidence(self) -> None:
        fake = FakeMCP()
        bridge = RuntimeMCPBridge(
            fake.call_tool,
            {"filesystem.read_file": "lookup"},
            condition="governed",
        )
        result = await bridge.call_tool("filesystem.read_file", {})
        self.assertIn("contextcanon_runtime_evidence", result["result"])

    async def test_baseline_does_not_observe_or_inject(self) -> None:
        fake = FakeMCP()
        bridge = RuntimeMCPBridge(fake.call_tool, {"filesystem.read_file": "lookup"}, condition="baseline")
        result = await bridge.call_tool("filesystem.read_file", {})
        self.assertEqual(bridge.diagnostics.observations_seen, 0)
        self.assertNotIn("contextcanon_runtime_evidence", result["result"])

    async def test_failed_observation_is_recorded_without_claim(self) -> None:
        async def fail(name: str, arguments: dict) -> None:
            raise RuntimeError("temporary failure")

        bridge = RuntimeMCPBridge(
            fail,
            {"retail_orders.get_order_details": "lookup"},
            condition="governed",
        )
        with self.assertRaisesRegex(RuntimeError, "temporary failure"):
            await bridge.call_tool("retail_orders.get_order_details", {})
        self.assertEqual(bridge.diagnostics.observations_seen, 1)
        self.assertEqual(bridge.adapter.ledger.claims, [])

    async def test_allow_dispatches_once(self) -> None:
        fake = FakeMCP()
        bridge = RuntimeMCPBridge(
            fake.call_tool,
            {"phone_and_messages.send_phone_message": "commit"},
            condition="guard",
        )
        result = await bridge.call_tool("phone_and_messages.send_phone_message", {})
        self.assertEqual(len(fake.calls), 1)
        self.assertTrue(bridge.diagnostics.commit_dispatched)
        self.assertEqual(result, {"result": {"status": "ok"}})

    async def test_conflicted_commit_is_withheld(self) -> None:
        fake = FakeMCP()
        adapter = AgentAbstainAdapter()
        adapter.observe_tool_result(
            RuntimeObservation(
                "event_system.event_search", "lookup", {}, {"date": "2026-03-22"}, True, 0
            )
        )
        adapter.observe_tool_result(
            RuntimeObservation(
                "event_system.event_verifier", "verify", {}, {"date": "2026-03-23"}, True, 1
            )
        )
        bridge = RuntimeMCPBridge(
            fake.call_tool,
            {"phone_and_messages.send_phone_message": "commit"},
            condition="guard",
            adapter=adapter,
        )
        result = await bridge.call_tool("phone_and_messages.send_phone_message", {})
        self.assertEqual(fake.calls, [])
        self.assertFalse(bridge.diagnostics.commit_dispatched)
        self.assertEqual(bridge.diagnostics.guard_decision, GuardDecision.REQUIRE_CLARIFICATION.value)
        self.assertIn("Clarification is required", result["structuredContent"]["message"])

    async def test_real_result_shapes_create_expected_claims(self) -> None:
        adapter = AgentAbstainAdapter()
        adapter.observe_tool_result(
            RuntimeObservation(
                "filesystem.read_file",
                "lookup",
                {"path": "/home/alice/event-info.txt"},
                {"content": "Event: Riverside Community Hall Spring Gala\\nDate: March 22, 2026"},
                True,
                0,
            )
        )
        adapter.observe_tool_result(
            RuntimeObservation(
                "retail_orders.verify_order_summary_document",
                "verify",
                {},
                {"order_id": "W8855135", "snapshot_status": "returned"},
                True,
                1,
            )
        )
        adapter.observe_tool_result(
            RuntimeObservation(
                "disaster_relief_operations.get_district_situation",
                "lookup",
                {},
                {
                    "evacuation_centers": [
                        {"name": "Harborview Middle School", "status": "open"},
                        {"name": "Seaside Church Hall", "status": "closed"},
                    ]
                },
                True,
                2,
            )
        )
        self.assertEqual(
            {(claim.subject, claim.predicate, claim.value) for claim in adapter.ledger.claims},
            {
                ("event/Riverside Community Hall Spring Gala", "date", "2026-03-22"),
                ("order/W8855135", "lifecycle_status", "returned"),
                ("shelter/Seaside Church Hall", "status", "closed"),
            },
        )


if __name__ == "__main__":
    unittest.main()
