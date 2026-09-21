from __future__ import annotations

import unittest

from experiments.agentabstain.adapter import (
    AgentAbstainAdapter,
    GuardDecision,
    RuntimeObservation,
)
from experiments.agentabstain.harness import RuntimeMCPBridge


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
