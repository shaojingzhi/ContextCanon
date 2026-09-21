"""Explicit, opt-in live AgentAbstain model runner for the M7.1 spike."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from .adapter import AgentAbstainAdapter
from .openai_runtime import build_contextcanon_server_class, official_server_env

TASKS = ("preview_008", "preview_013", "preview_015")
SIDES = ("act", "abstain")
CONDITIONS = ("baseline", "governed", "guard")


def _upstream(repo: Path):
    repo = Path(repo)
    repo = repo.expanduser().resolve()
    if not repo.exists():
        raise SystemExit("Set --agentabstain-repo to a local AgentAbstain checkout.")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from agent.openaisdk.agent import OpenAISDKAgent
    from agents import Agent, ModelSettings, Runner
    from agents.mcp.server import MCPServerStdio
    from src.runtime.common import (
        RUNTIME_EXPORT_TOOL_NAME,
        build_runtime_server_args,
        build_task_run_result,
        coerce_final_output,
        normalize_runtime_export_payload,
    )
    from src.types.BaseAgent import BaseAgent
    return (OpenAISDKAgent, Agent, ModelSettings, Runner, MCPServerStdio,
            RUNTIME_EXPORT_TOOL_NAME, build_runtime_server_args,
            build_task_run_result, coerce_final_output,
            normalize_runtime_export_payload, BaseAgent)


def _extract_usage(run_result: Any) -> dict[str, int] | None:
    try:
        responses = getattr(run_result, "raw_responses", None) or []
        values = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "requests": 0}
        for response in responses:
            usage = getattr(response, "usage", None)
            if usage is None:
                continue
            values["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
            values["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
            values["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
            values["requests"] += int(getattr(usage, "requests", 0) or 1)
        values["num_responses"] = len(responses)
        return values
    except Exception:
        return None


async def run_one(args: argparse.Namespace, task: str, side: str) -> dict[str, Any]:
    (OpenAISDKAgent, Agent, ModelSettings, Runner, _MCPServer, export_name,
     build_server_args, build_result, coerce_output, normalize_export, BaseAgent) = _upstream(args.agentabstain_repo)
    agent = OpenAISDKAgent(args.model, 0.0, args.max_turns, args.results_root)
    bundle = BaseAgent.load_task_bundle("conflicting_evidence", task, side)
    artifact_dir = agent.build_artifact_dir(bundle.category, bundle.task_id, bundle.task_type)
    server_type = build_contextcanon_server_class(args.agentabstain_repo)
    server = server_type(
        name="task_env",
        params={"command": sys.executable, "args": build_server_args(bundle), "cwd": str(args.agentabstain_repo), "env": official_server_env()},
        tool_filter={"blocked_tool_names": [export_name]},
        adapter=AgentAbstainAdapter(),
        condition=args.condition,
    )
    final_output = None
    export_payload = None
    run_error = None
    usage = None
    try:
        await server.connect()
        sdk_agent = Agent(
            name=f"{task}_{side}_agent",
            instructions=bundle.task_yaml["system_prompt"],
            model=args.model,
            model_settings=ModelSettings(temperature=0.0),
            mcp_servers=[server],
        )
        try:
            result = await Runner.run(
                starting_agent=sdk_agent,
                input=bundle.task_yaml["instruction"],
                max_turns=args.max_turns,
            )
            final_output = coerce_output(result.final_output)
            usage = _extract_usage(result)
        except Exception as exc:
            run_error = str(exc)
        try:
            exported = await server.call_tool(export_name, {})
            payload = getattr(exported, "structuredContent", None)
            export_payload = normalize_export(payload)
        except Exception as exc:
            run_error = f"{run_error}; runtime export failed: {exc}" if run_error else str(exc)
    finally:
        try:
            await server.cleanup()
        except BaseException:
            pass
    metadata: dict[str, Any] = {
        "provider": "openai-compatible",
        "base_url": os.environ.get("OPENAI_BASE_URL"),
        "model": args.model,
        "condition": args.condition,
        "contextcanon": True,
        "commit_attempted": server.contextcanon_bridge.diagnostics.commit_attempted,
        "commit_dispatched": server.contextcanon_bridge.diagnostics.commit_dispatched,
    }
    if usage is not None:
        metadata["usage"] = usage
    result = build_result(
        agent=agent, bundle=bundle, artifact_dir=artifact_dir,
        final_output=final_output, export_payload=export_payload,
        run_error=run_error, provider_metadata=metadata,
    )
    return {
        "task_id": f"conflicting_evidence/{task}",
        "variant": side,
        "condition": args.condition,
        "artifact_dir": result.artifact_dir,
        "error": result.error,
        "diagnostics": server.contextcanon_bridge.diagnostics.to_dict(),
    }


def _validate(args: argparse.Namespace) -> int:
    repo = Path(args.agentabstain_repo).expanduser()
    data = Path(args.agentabstain_data).expanduser()
    missing = []
    for task in (TASKS if args.task == "all" else (args.task,)):
        for side in (SIDES if args.side == "all" else (args.side,)):
            path = data / "tasks" / "conflicting_evidence" / task / side / "task.yaml"
            if not path.exists():
                missing.append(str(path))
    print(json.dumps({"ready": not missing, "repo": str(repo), "data": str(data), "missing": missing}, indent=2))
    return 0 if not missing else 2


async def _run(args: argparse.Namespace) -> int:
    if not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set DEEPSEEK_API_KEY before --run (no key is persisted).")
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com")
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
    tasks = TASKS if args.task == "all" else (args.task,)
    sides = SIDES if args.side == "all" else (args.side,)
    results = [await run_one(args, task, side) for task in tasks for side in sides]
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agentabstain-repo", default=os.environ.get("AGENTABSTAIN_REPO", "/tmp/agentabstain-m7"))
    parser.add_argument("--agentabstain-data", default=os.environ.get("AGENTABSTAIN_DATA", "/tmp/agentabstain-data"))
    parser.add_argument("--task", choices=(*TASKS, "all"), default="preview_008")
    parser.add_argument("--side", choices=(*SIDES, "all"), default="abstain")
    parser.add_argument("--condition", choices=CONDITIONS, default="guard")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--results-root", default="experiments/agentabstain/results")
    parser.add_argument("--run", action="store_true", help="required before making model/API calls")
    args = parser.parse_args(argv)
    os.environ["AGENTABSTAIN_DATA"] = str(Path(args.agentabstain_data).expanduser())
    if not args.run:
        return _validate(args)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
