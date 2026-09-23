"""OpenAI Agents SDK bridge around AgentAbstain's official MCP server."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any

from contextcanon.semantic import SemanticExtractor
from contextcanon.tool_semantics import (
    StaticToolSemanticsResolver,
    ToolSemantics,
    ToolSemanticsResolver,
)

from .adapter import AgentAbstainAdapter
from .harness import RuntimeMCPBridge
from .runtime_governance import RuntimeGovernance


# The upstream MCP schema currently exposes no ``kind`` field to the MCP
# client. These are runtime tool names, not benchmark labels or critical
# action metadata. Keep this mapping limited to the six-task spike.
SPIKE_TOOL_KINDS = {
    "phone_and_messages.send_phone_message": "commit",
    "retail_orders.manage_returns_and_exchanges": "commit",
    "disaster_relief_operations.publish_community_announcement": "commit",
    "filesystem.read_file": "lookup",
    "industrial_and_infrastructure_control.event_search": "lookup",
    "industrial_and_infrastructure_control.event_verifier": "verify",
    "retail_orders.get_order_details": "lookup",
    "retail_orders.verify_order_summary_document": "verify",
    "disaster_relief_operations.get_district_situation": "lookup",
    "document_authoring_and_publication.lookup_external_records": "lookup",
    "phone_and_messages.verify_event_file": "verify",
    "retail_orders.find_customer_account": "lookup",
    "retail_orders.read_customer_profile": "lookup",
    "retail_orders.search_orders": "lookup",
    "retail_orders.get_product_details": "lookup",
    "retail_orders.verify_order_financials": "verify",
    "disaster_relief_operations.verify_shelter_status": "verify",
    "document_authoring_and_publication.verify_external_records": "verify",
}


class AgentAbstainStaticToolSemanticsResolver(StaticToolSemanticsResolver):
    """Compatibility-only semantics for the deterministic AgentAbstain spike."""

    def __init__(self) -> None:
        kind_to_semantics = {
            "lookup": ToolSemantics.READ,
            "verify": ToolSemantics.VERIFY,
            "commit": ToolSemantics.SIDE_EFFECT,
        }
        super().__init__({
            name: kind_to_semantics[kind]
            for name, kind in SPIKE_TOOL_KINDS.items()
        })


def canonical_tool_name(encoded_name: str, encoded_to_original: dict[str, str]) -> str:
    """Decode the OpenAI-safe name using the upstream server's mapping."""

    return encoded_to_original.get(encoded_name, encoded_name)


def official_server_env() -> dict[str, str]:
    """Pass only the official dataset path and import path to the subprocess."""

    return {
        key: os.environ[key]
        for key in ("AGENTABSTAIN_DATA", "PYTHONPATH")
        if os.environ.get(key)
    }


def build_contextcanon_server_class(agentabstain_repo: str | Path):
    """Create a subclass without importing upstream SDKs at module import time."""

    repo = str(Path(agentabstain_repo).expanduser().resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.runtime.openaisdk import _NameSafeMCPServer

    class ContextCanonOpenAIMCPServer(_NameSafeMCPServer):
        """Preserve upstream behavior while intercepting decoded MCP calls."""

        def __init__(self, *args: Any, adapter: AgentAbstainAdapter | None = None,
                     extractor: SemanticExtractor | None = None,
                     governance: RuntimeGovernance | None = None,
                     tool_semantics_resolver: ToolSemanticsResolver | None = None,
                     condition: str, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._contextcanon_tool_metadata: dict[str, dict[str, Any]] = {}
            self._contextcanon_bridge = RuntimeMCPBridge(
                self._dispatch_canonical,
                tool_semantics_resolver or AgentAbstainStaticToolSemanticsResolver(),
                condition=condition,
                adapter=adapter,
                extractor=extractor,
                governance=governance,
            )

        async def list_tools(self, run_context=None, agent=None):
            tools = await super().list_tools(run_context, agent)
            metadata: dict[str, dict[str, Any]] = {}
            for tool in tools:
                canonical_name = canonical_tool_name(
                    tool.name,
                    getattr(self, "_encoded_to_original", {}),
                )
                metadata[canonical_name] = {
                    "description": getattr(tool, "description", None),
                    "input_schema": getattr(
                        tool,
                        "inputSchema",
                        getattr(tool, "input_schema", None),
                    ),
                    "annotations": getattr(tool, "annotations", None),
                }
            self._contextcanon_tool_metadata = metadata
            return tools

        async def call_runtime_control_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any] | None = None,
        ) -> Any:
            """Call a host-only control tool outside model-facing governance routing."""

            return await self._dispatch_canonical(tool_name, arguments or {})

        async def _dispatch_canonical(
            self,
            canonical_name: str,
            arguments: dict[str, Any],
            *,
            meta: dict[str, Any] | None = None,
        ) -> Any:
            encoded_name = self._encode(canonical_name)
            return await _NameSafeMCPServer.call_tool(
                self,
                encoded_name,
                arguments,
                meta=meta,
            )

        async def call_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any] | None,
            meta: dict[str, Any] | None = None,
        ) -> Any:
            canonical_name = canonical_tool_name(
                tool_name, getattr(self, "_encoded_to_original", {})
            )
            metadata = self._contextcanon_tool_metadata.get(canonical_name, {})
            result = await self._contextcanon_bridge.call_tool(
                canonical_name,
                arguments,
                meta=meta,
                tool_description=metadata.get("description"),
                input_schema=metadata.get("input_schema"),
                annotations=metadata.get("annotations"),
            )
            # The Agents SDK expects an MCP CallToolResult object.  The local
            # bridge deliberately uses JSON dictionaries for pure unit tests,
            # so adapt only at this official SDK boundary.
            if isinstance(result, dict):
                from mcp.types import CallToolResult

                structured = result.get("structuredContent", result.get("structured_content"))
                return CallToolResult(
                    content=[],
                    structuredContent=structured,
                    isError=bool(result.get("isError", result.get("is_error", False))),
                )
            return result

        @property
        def contextcanon_bridge(self) -> RuntimeMCPBridge:
            return self._contextcanon_bridge

    return ContextCanonOpenAIMCPServer
