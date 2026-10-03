"""Small runtime bridge for real AgentAbstain MCP call boundaries.

The bridge receives only runtime tool names, kinds, arguments, results, and
errors. It deliberately does not know task labels or evaluator metadata.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic
from typing import Any, Awaitable, Callable, Mapping

from contextcanon.governance import (
    ActionGovernanceDecision,
    ActionGovernanceResult,
    FactNeed,
)
from contextcanon.semantic import SemanticExtractor
from contextcanon.tool_semantics import (
    StaticToolSemanticsResolver,
    ToolSemantics,
    ToolSemanticsResolver,
)

from .adapter import (
    AgentAbstainAdapter,
    GuardDecision,
    ProposedToolCall,
    RuntimeObservation,
)
from .runtime_governance import RuntimeGovernance


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    return str(value)


def _result_payload(result: Any) -> Any:
    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = result
    if isinstance(structured, dict) and set(structured) == {"result"}:
        structured = structured["result"]
    return _json_safe(structured)


def _result_is_error(result: Any) -> bool:
    def is_explicit_true(value: Any) -> bool:
        return value is True or (
            isinstance(value, str) and value.strip().casefold() == "true"
        )

    if isinstance(result, dict):
        return is_explicit_true(result.get("isError", result.get("is_error", False)))
    value = getattr(result, "is_error", None)
    if value is None:
        value = getattr(result, "isError", False)
    return is_explicit_true(value)


def _inject_governed_evidence(result: Any, evidence: str) -> Any:
    block = f"[ContextCanon runtime evidence]\n\n{evidence}"
    if isinstance(result, dict):
        enriched = dict(result)
        structured = enriched.get("structuredContent", enriched.get("structured_content"))
        if structured is None and isinstance(enriched.get("result"), dict):
            nested = dict(enriched["result"])
            nested["contextcanon_runtime_evidence"] = block
            enriched["result"] = nested
            return enriched
        if isinstance(structured, dict):
            structured = dict(structured)
            structured["contextcanon_runtime_evidence"] = block
            if "structuredContent" in enriched:
                enriched["structuredContent"] = structured
            else:
                enriched["structured_content"] = structured
        else:
            enriched["contextcanon_runtime_evidence"] = block
        return enriched

    if hasattr(result, "model_copy"):
        try:
            from mcp.types import TextContent

            content = list(getattr(result, "content", []) or [])
            content.append(TextContent(type="text", text=block))
            return result.model_copy(update={"content": content})
        except Exception:
            return result
    return result


@dataclass(slots=True)
class BridgeDiagnostics:
    observations_seen: int = 0
    claims_created: int = 0
    conflicts_detected: int = 0
    commit_attempted: bool = False
    guard_decision: str | None = None
    commit_dispatched: bool = False
    raw_tool_latency_ms: int = 0
    governance_background_time_ms: int = 0
    governance_barrier_wait_ms: int = 0
    time_to_return_read_tool_ms: int = 0
    time_to_side_effect_decision_ms: int = 0
    pending_governance_count: int = 0
    background_governance_failures: int = 0
    background_governance_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations_seen": self.observations_seen,
            "claims_created": self.claims_created,
            "conflicts_detected": self.conflicts_detected,
            "commit_attempted": self.commit_attempted,
            "guard_decision": self.guard_decision,
            "commit_dispatched": self.commit_dispatched,
            "raw_tool_latency_ms": self.raw_tool_latency_ms,
            "governance_background_time_ms": self.governance_background_time_ms,
            "governance_barrier_wait_ms": self.governance_barrier_wait_ms,
            "time_to_return_read_tool_ms": self.time_to_return_read_tool_ms,
            "time_to_side_effect_decision_ms": self.time_to_side_effect_decision_ms,
            "pending_governance_count": self.pending_governance_count,
            "background_governance_failures": self.background_governance_failures,
            "background_governance_errors": list(self.background_governance_errors),
        }


class GovernanceWorkState(StrEnum):
    PENDING = "PENDING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(slots=True)
class GovernanceWork:
    observation: RuntimeObservation
    state: GovernanceWorkState = GovernanceWorkState.PENDING
    task: asyncio.Task[None] | None = None
    error: str | None = None


class RuntimeMCPBridge:
    """Observe lookup/verify calls and gate commit calls before dispatch."""

    def __init__(
        self,
        call_tool: Callable[..., Awaitable[Any]],
        tool_semantics_resolver: ToolSemanticsResolver | Mapping[str, str],
        *,
        condition: str,
        adapter: AgentAbstainAdapter | None = None,
        extractor: SemanticExtractor | None = None,
        governance: RuntimeGovernance | None = None,
        governance_barrier_timeout_seconds: float = 5.0,
    ) -> None:
        if condition not in {"baseline", "governed", "guard"}:
            raise ValueError("condition must be baseline, governed, or guard")
        self._call_tool = call_tool
        if isinstance(tool_semantics_resolver, Mapping):
            legacy_semantics = {
                name: {
                    "lookup": ToolSemantics.READ,
                    "verify": ToolSemantics.VERIFY,
                    "commit": ToolSemantics.SIDE_EFFECT,
                }.get(kind, ToolSemantics.UNKNOWN)
                for name, kind in tool_semantics_resolver.items()
            }
            tool_semantics_resolver = StaticToolSemanticsResolver(legacy_semantics)
        self.tool_semantics_resolver = tool_semantics_resolver
        self.condition = condition
        if governance is not None and (adapter is not None or extractor is not None):
            raise ValueError("governance cannot be combined with legacy adapter/extractor")
        self.governance = governance
        self.adapter = (
            None
            if governance is not None
            else adapter or AgentAbstainAdapter(extractor=extractor)
        )
        self.call_index = 0
        self.diagnostics = BridgeDiagnostics()
        if governance_barrier_timeout_seconds <= 0:
            raise ValueError("governance barrier timeout must be positive")
        self.governance_barrier_timeout_seconds = governance_barrier_timeout_seconds
        self._governance_lock = asyncio.Lock()
        self._governance_work: list[GovernanceWork] = []

    async def prepare_tool_semantics(
        self,
        metadata: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Prime stable tool classifications during discovery, not invocation."""
        if self.condition == "baseline":
            return
        prepare = getattr(self.tool_semantics_resolver, "prepare", None)
        if prepare is not None:
            safe_metadata = {
                name: {
                    "description": values.get("description"),
                    "input_schema": _json_safe(values.get("input_schema")),
                    "annotations": _json_safe(values.get("annotations")),
                }
                for name, values in metadata.items()
            }
            await asyncio.to_thread(prepare, safe_metadata)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
        tool_description: str | None = None,
        input_schema: Any = None,
        annotations: Any = None,
    ) -> Any:
        call_started = monotonic()
        arguments = arguments or {}
        if self.condition == "baseline":
            semantics = ToolSemantics.UNKNOWN
        else:
            semantics = await asyncio.to_thread(
                self.tool_semantics_resolver.classify,
                tool_name,
                description=tool_description,
                input_schema=_json_safe(input_schema),
                annotations=_json_safe(annotations),
            )
        kind = {
            ToolSemantics.READ: "lookup",
            ToolSemantics.VERIFY: "verify",
            ToolSemantics.SIDE_EFFECT: "commit",
        }.get(semantics, "")
        if semantics is ToolSemantics.UNKNOWN and self.condition == "guard":
            self.diagnostics.commit_attempted = True
            self.diagnostics.guard_decision = GuardDecision.REQUIRE_CLARIFICATION.value
            return {
                "isError": True,
                "structuredContent": {
                    "message": (
                        "The tool's operational semantics could not be established. "
                        "Clarification is required before it can be executed."
                    )
                },
            }
        if semantics is ToolSemantics.SIDE_EFFECT:
            decision_started = monotonic()
            self.diagnostics.commit_attempted = True
            proposed = ProposedToolCall(tool_name, kind, arguments)
            if self.governance is not None:
                action_result = await self._evaluate_governed_action(proposed)
                decision = (
                    GuardDecision.ALLOW
                    if action_result.decision is ActionGovernanceDecision.ALLOW
                    else GuardDecision.REQUIRE_CLARIFICATION
                )
            else:
                assert self.adapter is not None
                decision = self.adapter.evaluate_proposed_commit(proposed)
            self.diagnostics.time_to_side_effect_decision_ms += round(
                (monotonic() - decision_started) * 1000
            )
            self.diagnostics.guard_decision = decision.value
            if self.condition == "guard" and decision is GuardDecision.REQUIRE_CLARIFICATION:
                return {
                    "isError": True,
                    "structuredContent": {
                        "message": (
                            "The proposed action depends on unresolved governed evidence. "
                            "Clarification is required before this action can be executed."
                        )
                    },
                }

        dispatch_started = monotonic()
        try:
            if meta is None:
                result = await self._call_tool(tool_name, arguments)
            else:
                result = await self._call_tool(tool_name, arguments, meta=meta)
        except Exception as exc:
            if kind in {"lookup", "verify"} and self.condition != "baseline":
                self._capture_observation(
                    tool_name,
                    kind,
                    arguments,
                    None,
                    success=False,
                    error=str(exc),
                )
            raise
        finally:
            self.diagnostics.raw_tool_latency_ms += round(
                (monotonic() - dispatch_started) * 1000
            )

        result_is_error = _result_is_error(result)
        if kind in {"lookup", "verify"} and self.condition != "baseline":
            self._capture_observation(
                tool_name,
                kind,
                arguments,
                _result_payload(result),
                success=not result_is_error,
                error="MCP tool returned isError=true" if result_is_error else None,
            )
            if not result_is_error and self.condition in {"governed", "guard"}:
                result = _inject_governed_evidence(
                    result,
                    self.governed_evidence(),
                )
            self.diagnostics.time_to_return_read_tool_ms += round(
                (monotonic() - call_started) * 1000
            )
        if semantics is ToolSemantics.SIDE_EFFECT:
            self.diagnostics.commit_dispatched = True
        return result

    def _capture_observation(
        self,
        tool_name: str,
        tool_kind: str,
        arguments: dict[str, Any],
        result: Any,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        self.diagnostics.observations_seen += 1
        observation = RuntimeObservation(
            tool_name=tool_name,
            tool_kind=tool_kind,
            tool_parameters=arguments,
            tool_result=result,
            success=success,
            call_index=self.call_index,
            error=error,
        )
        if self.governance is not None:
            _fingerprint, _is_new = self.governance.capture_observation(observation)
            claims: list[Any] = []
            conflicts = self.governance.conflicts_detected
        else:
            assert self.adapter is not None
            claims = self.adapter.observe_tool_result(observation)
            conflicts = len(self.adapter.ledger.conflicts())
        self.diagnostics.claims_created += len(claims)
        self.diagnostics.conflicts_detected = conflicts
        self.call_index += 1

    async def _process_governance(self, work: GovernanceWork) -> None:
        started = monotonic()
        try:
            async with self._governance_lock:
                assert self.governance is not None
                claims = await asyncio.to_thread(
                    self.governance.materialize_observation,
                    work.observation,
                )
                self.diagnostics.claims_created += len(claims)
                self.diagnostics.conflicts_detected = (
                    self.governance.conflicts_detected
                )
            work.state = GovernanceWorkState.COMPLETE
        except Exception as exc:
            work.state = GovernanceWorkState.FAILED
            work.error = f"{type(exc).__name__}: {exc}"
            if self.governance is not None and self.governance.budget is not None:
                self.governance.budget.governance_incomplete = True
            self.diagnostics.background_governance_failures += 1
            self.diagnostics.background_governance_errors.append(work.error)
        finally:
            self.diagnostics.governance_background_time_ms += round(
                (monotonic() - started) * 1000
            )
            self.diagnostics.pending_governance_count = sum(
                item.state is GovernanceWorkState.PENDING
                for item in self._governance_work
            )

    async def _evaluate_governed_action(
        self,
        proposed: ProposedToolCall,
    ) -> ActionGovernanceResult:
        """Evaluate one action within a single strict barrier deadline."""
        assert self.governance is not None
        deadline = monotonic() + self.governance_barrier_timeout_seconds

        def remaining() -> float:
            return max(0.0, deadline - monotonic())

        try:
            extraction = await asyncio.wait_for(
                asyncio.to_thread(self.governance.extract_action_needs, proposed),
                timeout=remaining(),
            )
            materialize_started = monotonic()
            await asyncio.wait_for(
                asyncio.to_thread(
                    self.governance.materialize_relevant_observations,
                    extraction.needs,
                ),
                timeout=remaining(),
            )
            self.diagnostics.governance_barrier_wait_ms += round(
                (monotonic() - materialize_started) * 1000
            )
            self.diagnostics.claims_created = self.governance.claims_materialized
            self.diagnostics.conflicts_detected = self.governance.conflicts_detected
            barrier_ok = await self._await_governance_barrier(
                extraction.needs,
                timeout=remaining(),
            )
            if not barrier_ok or remaining() <= 0:
                raise TimeoutError
            return await asyncio.wait_for(
                asyncio.to_thread(self.governance.evaluate_needs, extraction),
                timeout=remaining(),
            )
        except TimeoutError:
            if self.governance.budget is not None:
                self.governance.budget.governance_incomplete = True
            return ActionGovernanceResult(
                ActionGovernanceDecision.REQUIRE_CLARIFICATION
            )
        except Exception as exc:
            if self.governance.budget is not None:
                self.governance.budget.governance_incomplete = True
            self.diagnostics.background_governance_failures += 1
            self.diagnostics.background_governance_errors.append(
                f"{type(exc).__name__}: {exc}"
            )
            return ActionGovernanceResult(
                ActionGovernanceDecision.REQUIRE_CLARIFICATION
            )

    async def _await_governance_barrier(
        self,
        needs: tuple[FactNeed, ...],
        *,
        timeout: float,
    ) -> bool:
        """Wait for possibly relevant task-local work, bounded and fail closed.

        Raw observations are intentionally untyped until extraction completes.
        Until a work item can be proven irrelevant, it is relevant to the
        barrier.  This conservative rule prevents a pending conflicting value
        from being skipped merely because its wording differs from a FactNeed.
        """
        if not needs:
            return not any(
                item.state is GovernanceWorkState.FAILED
                for item in self._governance_work
            )
        relevant = [
            item for item in self._governance_work
            if item.state is GovernanceWorkState.PENDING
        ]
        if not relevant:
            return not any(
                item.state is GovernanceWorkState.FAILED
                for item in self._governance_work
            )
        started = monotonic()
        tasks = [item.task for item in relevant if item.task is not None]
        try:
            if tasks:
                await asyncio.wait_for(
                    asyncio.shield(asyncio.gather(*tasks)),
                    timeout=timeout,
                )
        except TimeoutError:
            return False
        finally:
            self.diagnostics.governance_barrier_wait_ms += round(
                (monotonic() - started) * 1000
            )
        return not any(
            item.state in {GovernanceWorkState.PENDING, GovernanceWorkState.FAILED}
            for item in relevant
        )

    async def drain_governance(self) -> None:
        """Observe all background outcomes before task teardown."""
        tasks = [item.task for item in self._governance_work if item.task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def governed_evidence(self) -> str:
        if self.governance is not None:
            return self.governance.render()
        assert self.adapter is not None
        return self.adapter.render_governed_evidence()

    @property
    def semantic_diagnostics(self) -> tuple[str, ...]:
        messages = list(
            self.governance.diagnostics if self.governance is not None else ()
        )
        messages.extend(
            str(item)
            for item in getattr(self.tool_semantics_resolver, "diagnostics", ())
        )
        return tuple(messages)

    @property
    def governance_work(self) -> tuple[GovernanceWork, ...]:
        return tuple(self._governance_work)
