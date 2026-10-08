from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest import mock

from contextcanon_v2 import (
    Evidence,
    FactNeed,
    Relation,
    SemanticCompiler,
    SummaryState,
    GovernedToolProxy,
    ToolGovernanceBlocked,
    ToolPolicy,
)
from experiments.agentabstain.openai_runtime import build_contextcanon_server_class


def _claim(
    value: str,
    evidence_ref: str,
    *,
    modality: str = "OBSERVED",
) -> dict[str, object]:
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


def _compiler(*claims: dict[str, object]) -> SemanticCompiler:
    response = json.dumps({"claims": claims})
    return SemanticCompiler(lambda _prompt: response)


class Backend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_tool(self, name: str, arguments: dict) -> dict[str, object]:
        self.calls.append((name, arguments))
        if name == "release.read_runtime":
            return {"channel": "canary"}
        return {"dispatched": True}


class V2ToolProxyTests(unittest.IsolatedAsyncioTestCase):
    static = Evidence(
        "static-1",
        "release-policy.md",
        "channel",
        "production release channel must be stable",
    )
    need = FactNeed(
        "release",
        "channel",
        {"environment": "production"},
        "enum",
    )

    def proxy(
        self,
        compiler: SemanticCompiler,
        backend: Backend,
        *,
        initial_evidence: tuple[Evidence, ...] = (),
    ) -> GovernedToolProxy:
        def result_to_evidence(
            _name: str, _arguments: dict, result: dict
        ) -> tuple[Evidence, ...]:
            return (
                Evidence(
                    "runtime-1",
                    "release.read_runtime",
                    "result.channel",
                    f"production runtime channel is {result['channel']}",
                ),
            )

        return GovernedToolProxy(
            backend.call_tool,
            {
                "release.read_runtime": ToolPolicy.READ,
                "release.deploy": ToolPolicy.SIDE_EFFECT,
            },
            lambda _name, _arguments: (self.need,),
            compiler,
            initial_evidence=initial_evidence,
            result_to_evidence=result_to_evidence,
        )

    async def test_static_and_runtime_evidence_share_one_governance_decision(self) -> None:
        backend = Backend()
        proxy = self.proxy(
            _compiler(
                _claim("stable", "E1", modality="REQUIRED"),
                _claim("canary", "E2"),
            ),
            backend,
            initial_evidence=(self.static,),
        )

        self.assertEqual(proxy.evidence, (self.static,))
        await proxy.call_tool("release.read_runtime")
        self.assertEqual(
            [evidence.evidence_id for evidence in proxy.evidence],
            ["static-1", "runtime-1"],
        )

        with self.assertRaises(ToolGovernanceBlocked) as raised:
            await proxy.call_tool("release.deploy", {"channel": "canary"})

        decision = raised.exception.governance_result
        self.assertEqual(decision.summary_state, SummaryState.UNRESOLVED)
        self.assertEqual(
            [record.relation for record in decision.relations],
            [Relation.DIVERGENT],
        )
        self.assertEqual(
            [name for name, _arguments in backend.calls],
            ["release.read_runtime"],
        )

    async def test_clear_allows_side_effect_dispatch(self) -> None:
        backend = Backend()
        proxy = self.proxy(
            _compiler(_claim("stable", "E1", modality="REQUIRED")),
            backend,
            initial_evidence=(self.static,),
        )

        result = await proxy.call_tool("release.deploy", {"channel": "stable"})

        self.assertEqual(result, {"dispatched": True})
        self.assertEqual(backend.calls, [("release.deploy", {"channel": "stable"})])

    async def test_incomplete_blocks_before_dispatch(self) -> None:
        backend = Backend()
        proxy = self.proxy(_compiler(), backend)

        with self.assertRaises(ToolGovernanceBlocked) as raised:
            await proxy.call_tool("release.deploy")

        self.assertEqual(
            raised.exception.governance_result.summary_state,
            SummaryState.INCOMPLETE,
        )
        self.assertEqual(backend.calls, [])

    async def test_compilation_failure_is_incomplete_and_not_dispatched(self) -> None:
        backend = Backend()
        proxy = self.proxy(SemanticCompiler(lambda _prompt: "not json"), backend)

        with self.assertRaises(ToolGovernanceBlocked) as raised:
            await proxy.call_tool("release.deploy")

        self.assertEqual(
            raised.exception.governance_result.summary_state,
            SummaryState.INCOMPLETE,
        )
        self.assertEqual(backend.calls, [])

    def test_generic_layer_has_no_agent_framework_dependency(self) -> None:
        source = (
            Path(__file__).parents[1] / "contextcanon_v2" / "tool_proxy.py"
        ).read_text()
        for framework_name in ("AgentAbstain", "OpenAI", "LangGraph", "MCP"):
            with self.subTest(framework_name=framework_name):
                self.assertNotIn(framework_name, source)

        parameters = inspect.signature(GovernedToolProxy).parameters
        self.assertEqual(
            set(parameters),
            {
                "backend_call_tool",
                "tool_policies",
                "dependency_resolver",
                "compiler",
                "initial_evidence",
                "result_to_evidence",
            },
        )

    def test_existing_openai_mcp_demo_accepts_generic_proxy_factory(self) -> None:
        parameters = inspect.signature(build_contextcanon_server_class).parameters
        self.assertIn("v2_proxy_factory", parameters)

    async def test_existing_openai_mcp_demo_routes_through_generic_proxy(self) -> None:
        dispatched: list[str] = []

        class Result:
            pass

        class FakeNameSafeMCPServer:
            def __init__(self, *args, **kwargs) -> None:
                self._encoded_to_original = {}

            def _encode(self, name: str) -> str:
                return name

            async def call_tool(self, name, arguments, meta=None):
                dispatched.append(name)
                return Result()

        src = ModuleType("src")
        runtime = ModuleType("src.runtime")
        openaisdk = ModuleType("src.runtime.openaisdk")
        openaisdk._NameSafeMCPServer = FakeNameSafeMCPServer

        def proxy_factory(backend_call_tool):
            return GovernedToolProxy(
                backend_call_tool,
                {
                    "source.read": ToolPolicy.READ,
                    "action.write": ToolPolicy.SIDE_EFFECT,
                },
                lambda _name, _arguments: (self.need,),
                _compiler(_claim("stable", "E1", modality="REQUIRED")),
                initial_evidence=(self.static,),
            )

        with mock.patch.dict(
            sys.modules,
            {
                "src": src,
                "src.runtime": runtime,
                "src.runtime.openaisdk": openaisdk,
            },
        ):
            server_type = build_contextcanon_server_class(
                ".", v2_proxy_factory=proxy_factory
            )
            server = server_type(condition="guard")
            await server.call_tool("source.read", {})
            await server.call_tool("action.write", {})

        self.assertEqual(dispatched, ["source.read", "action.write"])
        self.assertIsInstance(server.contextcanon_v2_proxy, GovernedToolProxy)


if __name__ == "__main__":
    unittest.main()
