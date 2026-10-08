from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
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
from experiments.agentabstain.run_agent import (
    V2_EVENT_DATE_NEED,
    V2_SIDE_EFFECT_TOOL,
    V2_TOOL_POLICIES,
    _build_v2_proxy_factory,
    _build_runner_server_type,
    _drain_legacy_governance,
    _runtime_diagnostics,
)


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


class V2AgentRunnerWiringTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def diagnostics() -> dict[str, object]:
        return {
            "evidence_count": 0,
            "side_effect_attempted": False,
            "side_effect_dispatched": False,
            "governance_summary_state": None,
            "blocked": False,
        }

    @staticmethod
    def claim(value: str, evidence_ref: str) -> dict[str, object]:
        return {
            "subject": V2_EVENT_DATE_NEED.subject,
            "predicate": V2_EVENT_DATE_NEED.dimension,
            "value": value,
            "value_type": "date",
            "scope": {},
            "modality": "DOCUMENTED",
            "cardinality": "SINGLE",
            "evidence_refs": [evidence_ref],
        }

    async def test_v2_runner_factory_captures_full_read_results_as_evidence(self) -> None:
        calls: list[str] = []

        async def backend(name: str, _arguments: dict) -> dict[str, object]:
            calls.append(name)
            return {"structuredContent": {"event": {"date": "2026-03-22"}}}

        diagnostics = self.diagnostics()
        factory = _build_v2_proxy_factory(
            lambda _prompt: json.dumps({"claims": []}), diagnostics
        )
        proxy = factory(backend)

        await proxy.call_tool("filesystem.read_file", {"path": "event-info.txt"})

        self.assertEqual(calls, ["filesystem.read_file"])
        self.assertEqual(len(proxy.evidence), 1)
        evidence = proxy.evidence[0]
        self.assertEqual(evidence.source_id, "filesystem.read_file")
        self.assertEqual(evidence.location, "tool-result/1")
        self.assertEqual(
            json.loads(evidence.content),
            {"event": {"date": "2026-03-22"}},
        )

    async def test_v2_runner_conflict_blocks_send_before_backend_dispatch(self) -> None:
        calls: list[str] = []

        async def backend(name: str, _arguments: dict) -> dict[str, object]:
            calls.append(name)
            return {"date": "2026-03-22" if len(calls) == 1 else "2026-03-23"}

        response = json.dumps({
            "claims": [
                self.claim("2026-03-22", "E1"),
                self.claim("2026-03-23", "E2"),
            ]
        })
        completions: list[str] = []
        diagnostics = self.diagnostics()
        proxy = _build_v2_proxy_factory(
            lambda prompt: completions.append(prompt) or response,
            diagnostics,
        )(backend)
        await proxy.call_tool("filesystem.read_file", {})
        await proxy.call_tool(
            "industrial_and_infrastructure_control.event_verifier", {}
        )

        with self.assertRaises(ToolGovernanceBlocked):
            await proxy.call_tool(V2_SIDE_EFFECT_TOOL, {"message_text": "date"})

        self.assertEqual(len(completions), 1)
        self.assertEqual(
            calls,
            [
                "filesystem.read_file",
                "industrial_and_infrastructure_control.event_verifier",
            ],
        )
        self.assertEqual(diagnostics["governance_summary_state"], "UNRESOLVED")
        self.assertTrue(diagnostics["blocked"])
        self.assertFalse(diagnostics["side_effect_dispatched"])

    async def test_v2_runner_clear_dispatches_send(self) -> None:
        calls: list[str] = []

        async def backend(name: str, _arguments: dict) -> dict[str, object]:
            calls.append(name)
            return {"ok": True}

        diagnostics = self.diagnostics()
        response = json.dumps({"claims": [self.claim("2026-03-22", "E1")]})
        proxy = _build_v2_proxy_factory(
            lambda _prompt: response, diagnostics
        )(backend)
        await proxy.call_tool("filesystem.read_file", {})

        await proxy.call_tool(V2_SIDE_EFFECT_TOOL, {"message_text": "date"})

        self.assertEqual(calls[-1], V2_SIDE_EFFECT_TOOL)
        self.assertEqual(diagnostics["governance_summary_state"], "CLEAR")
        self.assertTrue(diagnostics["side_effect_dispatched"])
        self.assertFalse(diagnostics["blocked"])

    def test_v2_demo_uses_only_explicit_static_tool_policies(self) -> None:
        self.assertEqual(
            V2_TOOL_POLICIES,
            {
                "filesystem.read_file": ToolPolicy.READ,
                "industrial_and_infrastructure_control.event_search": ToolPolicy.READ,
                "industrial_and_infrastructure_control.event_verifier": ToolPolicy.READ,
                "phone_and_messages.verify_event_file": ToolPolicy.READ,
                V2_SIDE_EFFECT_TOOL: ToolPolicy.SIDE_EFFECT,
            },
        )

    def test_v2_mode_passes_proxy_factory_and_legacy_default_does_not(self) -> None:
        v2_args = SimpleNamespace(runtime="v2", agentabstain_repo="repo")
        legacy_args = SimpleNamespace(runtime="legacy", agentabstain_repo="repo")
        complete = lambda _prompt: json.dumps({"claims": []})
        diagnostics = self.diagnostics()

        with (
            mock.patch(
                "experiments.agentabstain.run_agent._v2_complete",
                return_value=complete,
            ) as v2_complete,
            mock.patch(
                "experiments.agentabstain.run_agent.build_contextcanon_server_class",
                return_value="server-type",
            ) as build_server,
        ):
            self.assertEqual(
                _build_runner_server_type(v2_args, diagnostics), "server-type"
            )
            self.assertIn("v2_proxy_factory", build_server.call_args.kwargs)
            v2_complete.assert_called_once_with(v2_args)

            build_server.reset_mock()
            v2_complete.reset_mock()
            self.assertEqual(
                _build_runner_server_type(legacy_args, diagnostics), "server-type"
            )
            build_server.assert_called_once_with("repo")
            v2_complete.assert_not_called()

    async def test_v2_finish_path_never_dereferences_legacy_bridge(self) -> None:
        class Proxy:
            evidence = (object(), object())

        class V2Server:
            contextcanon_v2_proxy = Proxy()

            @property
            def contextcanon_bridge(self):
                raise AssertionError("legacy bridge accessed")

        server = V2Server()
        diagnostics = self.diagnostics()

        await _drain_legacy_governance("v2", server)
        result = _runtime_diagnostics("v2", server, diagnostics)

        self.assertEqual(result["evidence_count"], 2)


if __name__ == "__main__":
    unittest.main()
