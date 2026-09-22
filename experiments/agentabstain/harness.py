"""Small runtime bridge for real AgentAbstain MCP call boundaries.

The bridge receives only runtime tool names, kinds, arguments, results, and
errors. It deliberately does not know task labels or evaluator metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from contextcanon.semantic import SemanticExtractor
from contextcanon.governance import ActionGovernanceDecision

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
    if isinstance(result, dict):
        return bool(result.get("isError", result.get("is_error", False)))
    return bool(
        getattr(result, "is_error", getattr(result, "isError", False))
    )


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations_seen": self.observations_seen,
            "claims_created": self.claims_created,
            "conflicts_detected": self.conflicts_detected,
            "commit_attempted": self.commit_attempted,
            "guard_decision": self.guard_decision,
            "commit_dispatched": self.commit_dispatched,
        }


class RuntimeMCPBridge:
    """Observe lookup/verify calls and gate commit calls before dispatch."""

    def __init__(
        self,
        call_tool: Callable[..., Awaitable[Any]],
        tool_kinds: Mapping[str, str],
        *,
        condition: str,
        adapter: AgentAbstainAdapter | None = None,
        extractor: SemanticExtractor | None = None,
        governance: RuntimeGovernance | None = None,
    ) -> None:
        if condition not in {"baseline", "governed", "guard"}:
            raise ValueError("condition must be baseline, governed, or guard")
        self._call_tool = call_tool
        self._tool_kinds = dict(tool_kinds)
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

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        arguments = arguments or {}
        kind = self._tool_kinds.get(tool_name, "")
        if kind == "commit":
            self.diagnostics.commit_attempted = True
            proposed = ProposedToolCall(tool_name, kind, arguments)
            if self.governance is not None:
                action_result = self.governance.evaluate_action(proposed)
                decision = (
                    GuardDecision.ALLOW
                    if action_result.decision is ActionGovernanceDecision.ALLOW
                    else GuardDecision.REQUIRE_CLARIFICATION
                )
            else:
                assert self.adapter is not None
                decision = self.adapter.evaluate_proposed_commit(proposed)
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

        try:
            if meta is None:
                result = await self._call_tool(tool_name, arguments)
            else:
                result = await self._call_tool(tool_name, arguments, meta=meta)
        except Exception as exc:
            if kind in {"lookup", "verify"} and self.condition != "baseline":
                self._record_observation(
                    tool_name,
                    kind,
                    arguments,
                    None,
                    success=False,
                    error=str(exc),
                )
            raise

        result_is_error = _result_is_error(result)
        if kind in {"lookup", "verify"} and self.condition != "baseline":
            self._record_observation(
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
        if kind == "commit":
            self.diagnostics.commit_dispatched = True
        return result

    def _record_observation(
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
            claims = self.governance.observe(observation)
            conflicts = self.governance.conflicts_detected
        else:
            assert self.adapter is not None
            claims = self.adapter.observe_tool_result(observation)
            conflicts = len(self.adapter.ledger.conflicts())
        self.diagnostics.claims_created += len(claims)
        self.diagnostics.conflicts_detected = conflicts
        self.call_index += 1

    def governed_evidence(self) -> str:
        if self.governance is not None:
            return self.governance.render()
        assert self.adapter is not None
        return self.adapter.render_governed_evidence()
