"""Framework-neutral governance at an asynchronous tool boundary."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any

from .compiler import SemanticCompilationError, SemanticCompiler
from .governance import govern
from .models import Evidence, FactNeed, GovernanceResult, SummaryState


class ToolPolicy(StrEnum):
    READ = "READ"
    SIDE_EFFECT = "SIDE_EFFECT"


class ToolGovernanceBlocked(RuntimeError):
    """Raised before dispatch when required knowledge is not clear."""

    def __init__(self, tool_name: str, result: GovernanceResult) -> None:
        super().__init__(
            f"{tool_name} blocked by governance: {result.summary_state.value}"
        )
        self.tool_name = tool_name
        self.governance_result = result


class GovernedToolProxy:
    """Collect read evidence and gate side effects without an agent dependency."""

    def __init__(
        self,
        backend_call_tool: Callable[[str, dict[str, Any]], Awaitable[Any]],
        tool_policies: Mapping[str, ToolPolicy],
        dependency_resolver: Callable[[str, dict[str, Any]], Sequence[FactNeed]],
        compiler: SemanticCompiler,
        *,
        initial_evidence: Sequence[Evidence] = (),
        result_to_evidence: Callable[
            [str, dict[str, Any], Any], Sequence[Evidence]
        ]
        | None = None,
    ) -> None:
        self._backend_call_tool = backend_call_tool
        self._tool_policies = dict(tool_policies)
        self._dependency_resolver = dependency_resolver
        self._compiler = compiler
        self._evidence = list(initial_evidence)
        self._result_to_evidence = result_to_evidence

    @property
    def evidence(self) -> tuple[Evidence, ...]:
        return tuple(self._evidence)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        arguments = dict(arguments or {})
        try:
            policy = ToolPolicy(self._tool_policies[name])
        except (KeyError, ValueError) as error:
            raise ValueError(f"no valid policy for tool: {name}") from error

        if policy is ToolPolicy.READ:
            result = await self._backend_call_tool(name, arguments)
            if self._result_to_evidence is not None:
                self._evidence.extend(
                    self._result_to_evidence(name, arguments, result)
                )
            return result

        needs = tuple(self._dependency_resolver(name, arguments))
        try:
            claims = self._compiler.compile(self._evidence, needs)
            decision = govern(claims, needs)
        except SemanticCompilationError:
            decision = govern((), needs, compilation_error=True)
        if decision.summary_state is not SummaryState.CLEAR:
            raise ToolGovernanceBlocked(name, decision)
        return await self._backend_call_tool(name, arguments)


__all__ = ["GovernedToolProxy", "ToolGovernanceBlocked", "ToolPolicy"]
