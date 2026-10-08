"""Offline LangGraph demonstration using the framework-neutral V2 tool proxy."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from itertools import count
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from contextcanon_v2 import (
    Evidence,
    FactNeed,
    GovernedToolProxy,
    SemanticCompiler,
    ToolGovernanceBlocked,
    ToolPolicy,
)

RELEASE_NEED = FactNeed(
    subject="release",
    dimension="channel",
    scope_constraint={"environment": "production"},
    value_type="enum",
)
STATIC_POLICY = Evidence(
    evidence_id="static-policy",
    source_id="release-policy.md",
    location="production.channel",
    content="production release channel must be stable",
)
TOOL_PLAN = ("read_policy", "read_runtime_state", "deploy_release")


class DemoState(TypedDict):
    attempted_tools: list[str]
    tool_results: list[Any]
    next_tool: str | None
    blocked: bool
    governance_summary_state: str | None


class ScriptedModel:
    """Offline stand-in that lets LangGraph own deterministic tool selection."""

    def choose_tool(self, state: DemoState) -> str | None:
        position = len(state["attempted_tools"])
        return TOOL_PLAN[position] if position < len(TOOL_PLAN) else None


class ReleaseBackend:
    def __init__(self, runtime_channel: str) -> None:
        self.runtime_channel = runtime_channel
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        if name == "read_policy":
            return {"environment": "production", "required_channel": "stable"}
        if name == "read_runtime_state":
            return {
                "environment": "production",
                "observed_channel": self.runtime_channel,
            }
        return {"deployed": True, "channel": arguments["channel"]}


def initial_state() -> DemoState:
    return {
        "attempted_tools": [],
        "tool_results": [],
        "next_tool": None,
        "blocked": False,
        "governance_summary_state": None,
    }


def build_demo(
    complete: Callable[[str], str],
    *,
    runtime_channel: str = "canary",
    initial_evidence: Sequence[Evidence] = (STATIC_POLICY,),
) -> tuple[Any, GovernedToolProxy, ReleaseBackend]:
    """Build a LangGraph loop whose tool node delegates to GovernedToolProxy."""
    backend = ReleaseBackend(runtime_channel)
    evidence_number = count(1)

    def result_to_evidence(
        name: str, _arguments: dict[str, Any], result: Any
    ) -> tuple[Evidence, ...]:
        number = next(evidence_number)
        return (
            Evidence(
                evidence_id=f"runtime-{number}",
                source_id=name,
                location=f"tool-result/{number}",
                content=json.dumps(result, sort_keys=True),
            ),
        )

    proxy = GovernedToolProxy(
        backend.call_tool,
        {
            "read_policy": ToolPolicy.READ,
            "read_runtime_state": ToolPolicy.READ,
            "deploy_release": ToolPolicy.SIDE_EFFECT,
        },
        lambda name, _arguments: (RELEASE_NEED,) if name == "deploy_release" else (),
        SemanticCompiler(complete),
        initial_evidence=initial_evidence,
        result_to_evidence=result_to_evidence,
    )
    model = ScriptedModel()

    async def agent_node(state: DemoState) -> dict[str, Any]:
        return {"next_tool": model.choose_tool(state)}

    async def tool_node(state: DemoState) -> dict[str, Any]:
        name = state["next_tool"]
        if name is None:
            return {}
        attempted = [*state["attempted_tools"], name]
        arguments = {"channel": "stable"} if name == "deploy_release" else {}
        try:
            result = await proxy.call_tool(name, arguments)
        except ToolGovernanceBlocked as error:
            return {
                "attempted_tools": attempted,
                "blocked": True,
                "governance_summary_state": error.governance_result.summary_state.value,
            }
        return {
            "attempted_tools": attempted,
            "tool_results": [*state["tool_results"], result],
        }

    graph = StateGraph(DemoState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", lambda state: "done" if state["next_tool"] is None else "tool",
        {"tool": "tool", "done": END},
    )
    graph.add_conditional_edges(
        "tool", lambda state: "done" if state["blocked"] else "agent",
        {"agent": "agent", "done": END},
    )
    return graph.compile(), proxy, backend
