from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from contextcanon_v2 import SummaryState


LANGGRAPH_AVAILABLE = importlib.util.find_spec("langgraph") is not None


def _claim(value: str, evidence_ref: str, modality: str) -> dict[str, object]:
    return {
        "subject": "release",
        "predicate": "channel",
        "value": value,
        "value_type": "enum",
        "scope": {"environment": "production"},
        "modality": modality,
        "cardinality": "SINGLE",
        "evidence_refs": [evidence_ref],
    }


def _completion(runtime_channel: str):
    response = json.dumps(
        {
            "claims": [
                _claim("stable", "E1", "REQUIRED"),
                _claim(runtime_channel, "E3", "OBSERVED"),
            ]
        }
    )
    return lambda _prompt: response


class LangGraphDemoTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(LANGGRAPH_AVAILABLE, "install the langgraph-demo extra")
    async def test_conflict_blocks_side_effect_after_read_evidence(self) -> None:
        from experiments.v2_langgraph_demo import build_demo, initial_state

        graph, proxy, backend = build_demo(_completion("canary"))

        state = await graph.ainvoke(initial_state())

        self.assertEqual(
            state["attempted_tools"],
            ["read_policy", "read_runtime_state", "deploy_release"],
        )
        self.assertEqual(
            [name for name, _arguments in backend.calls],
            ["read_policy", "read_runtime_state"],
        )
        self.assertEqual(
            [item.evidence_id for item in proxy.evidence],
            ["static-policy", "runtime-1", "runtime-2"],
        )
        self.assertTrue(state["blocked"])
        self.assertEqual(
            state["governance_summary_state"], SummaryState.UNRESOLVED.value
        )

    @unittest.skipUnless(LANGGRAPH_AVAILABLE, "install the langgraph-demo extra")
    async def test_clear_allows_side_effect_dispatch(self) -> None:
        from experiments.v2_langgraph_demo import build_demo, initial_state

        graph, proxy, backend = build_demo(
            _completion("stable"), runtime_channel="stable"
        )

        state = await graph.ainvoke(initial_state())

        self.assertFalse(state["blocked"])
        self.assertEqual(len(proxy.evidence), 3)
        self.assertEqual(
            [name for name, _arguments in backend.calls],
            ["read_policy", "read_runtime_state", "deploy_release"],
        )
        self.assertEqual(state["tool_results"][-1]["deployed"], True)

    def test_demo_has_no_openai_agents_sdk_dependency(self) -> None:
        text = (
            Path(__file__).parents[1] / "experiments" / "v2_langgraph_demo.py"
        ).read_text()
        self.assertNotIn("from agents", text)
        self.assertNotIn("experiments.agentabstain", text)

    @unittest.skipUnless(LANGGRAPH_AVAILABLE, "install the langgraph-demo extra")
    async def test_live_path_uses_real_model_choices_without_retry(self) -> None:
        from experiments.v2_langgraph_live import run_live

        responses = iter(
            [
                '{"tool":"read_policy","arguments":{}}',
                '{"tool":"read_runtime_state","arguments":{}}',
                '{"tool":"deploy_release","arguments":{"channel":"stable"}}',
                json.dumps(
                    {
                        "claims": [
                            _claim("stable", "E1", "REQUIRED"),
                            _claim("canary", "E3", "OBSERVED"),
                        ]
                    }
                ),
            ]
        )

        class Client:
            def complete(self, _prompt: str, *, model: str):
                return SimpleNamespace(
                    content=next(responses), usage=None, model=model
                )

        record = await run_live(Client(), "test-model")

        self.assertEqual(record["provider_call_count"], 4)
        self.assertEqual(record["agent_model_call_count"], 3)
        self.assertEqual(record["evidence_count"], 3)
        self.assertEqual(record["governance_summary_state"], "UNRESOLVED")
        self.assertTrue(record["blocked"])
        self.assertFalse(record["side_effect_dispatched"])
        self.assertIsNone(record["error"])

    @unittest.skipUnless(LANGGRAPH_AVAILABLE, "install the langgraph-demo extra")
    async def test_invalid_live_choice_fails_without_retry(self) -> None:
        from experiments.v2_langgraph_live import run_live

        class Client:
            calls = 0

            def complete(self, _prompt: str, *, model: str):
                self.calls += 1
                return SimpleNamespace(content="not json", usage=None, model=model)

        client = Client()
        record = await run_live(client, "test-model")

        self.assertEqual(client.calls, 1)
        self.assertEqual(record["provider_call_count"], 1)
        self.assertIn("JSONDecodeError", record["error"])
