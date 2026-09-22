"""Minimal non-artifact DeepSeek + official MCP smoke diagnostic."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .adapter import AgentAbstainAdapter
from .openai_runtime import build_contextcanon_server_class, official_server_env


async def run_smoke(repo: Path, data: Path, model_name: str, max_turns: int) -> dict:
    os.environ["AGENTABSTAIN_DATA"] = str(data.expanduser().resolve())
    os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")
    repo = repo.expanduser().resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from agents import Agent, ModelSettings, Runner, set_tracing_disabled
    from agents.models.openai_provider import OpenAIProvider
    from src.runtime.common import RUNTIME_EXPORT_TOOL_NAME, build_runtime_server_args
    from src.types.BaseAgent import BaseAgent

    set_tracing_disabled(True)
    bundle = BaseAgent.load_task_bundle("conflicting_evidence", "preview_008", "abstain")
    server_type = build_contextcanon_server_class(repo)
    server = server_type(
        name="task_env",
        params={
            "command": sys.executable,
            "args": build_runtime_server_args(bundle),
            "cwd": str(repo),
            "env": official_server_env(),
        },
        tool_filter={"blocked_tool_names": [RUNTIME_EXPORT_TOOL_NAME]},
        adapter=AgentAbstainAdapter(),
        condition="guard",
    )
    try:
        await server.connect()
        model = OpenAIProvider(use_responses=False).get_model(model_name)
        agent = Agent(
            name="preview_008_abstain_smoke",
            instructions=bundle.task_yaml["system_prompt"],
            model=model,
            model_settings=ModelSettings(
                temperature=0.0,
                extra_body={"thinking": {"type": "enabled"}},
                extra_args={"reasoning_effort": "high"},
            ),
            mcp_servers=[server],
        )
        result = await Runner.run(
            starting_agent=agent,
            input=bundle.task_yaml["instruction"],
            max_turns=max_turns,
        )
        payload = await server.call_runtime_control_tool(RUNTIME_EXPORT_TOOL_NAME, {})
        exported = getattr(payload, "structuredContent", None)
        if exported is None:
            exported = getattr(payload, "structured_content", None)
        diagnostics = server.contextcanon_bridge.diagnostics.to_dict()
        executed_tools = [
            entry.get("tool") for entry in (exported or {}).get("execution_log", [])
            if isinstance(entry, dict) and isinstance(entry.get("tool"), str)
        ]
        return {
            "tool_sequence": executed_tools,
            "diagnostics": diagnostics,
            "final_output": str(result.final_output),
        }
    finally:
        await server.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.environ.get("AGENTABSTAIN_REPO", "/tmp/agentabstain-m7"))
    parser.add_argument("--data", default=os.environ.get("AGENTABSTAIN_DATA", "/tmp/agentabstain-data"))
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--max-turns", type=int, default=8)
    args = parser.parse_args(argv)
    print(json.dumps(asyncio.run(run_smoke(Path(args.repo), Path(args.data), args.model, args.max_turns)), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
