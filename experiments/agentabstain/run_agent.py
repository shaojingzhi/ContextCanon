"""Explicit, opt-in live AgentAbstain model runner for the M7.1 spike."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from time import monotonic
from pathlib import Path
from typing import Any

from contextcanon.governance import (
    GovernanceStore,
    LLMEvidenceAligner,
    LLMFactNeedExtractor,
    LLMRelationClassifier,
)
from contextcanon.semantic import (
    LLMStructuredExtractor,
    OpenAICompatibleExtractionClient,
    SemanticInferenceConfig,
)
from contextcanon.semantic_budget import BudgetedSemanticClient, SemanticBudget
from contextcanon.tool_semantics import LLMToolSemanticsResolver

from .adapter import AgentAbstainAdapter
from .openai_runtime import build_contextcanon_server_class, official_server_env
from .runtime_governance import RuntimeGovernance

TASKS = ("preview_008", "preview_013", "preview_015")
SIDES = ("act", "abstain")
CONDITIONS = ("baseline", "governed", "guard")
EXTRACTORS = ("legacy", "llm")
DEFAULT_SEMANTIC_MAX_OUTPUT_TOKENS = 256
DEFAULT_SEMANTIC_EXTRACTION_MAX_OUTPUT_TOKENS = 1024


def _format_exception(stage: str, exc: BaseException) -> str:
    """Keep provider/runtime diagnostics useful without serializing secrets."""

    parts = [f"{stage}: {type(exc).__name__}: {exc}"]
    if exc.__cause__ is not None:
        parts.append(f"cause={type(exc.__cause__).__name__}: {exc.__cause__}")
    if exc.__context__ is not None and exc.__context__ is not exc.__cause__:
        parts.append(f"context={type(exc.__context__).__name__}: {exc.__context__}")
    return " | ".join(parts)


def _configure_provider_environment() -> None:
    """Use the concrete DeepSeek key for both semantic and Agent SDK clients."""

    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        os.environ["OPENAI_API_KEY"] = deepseek_key
    elif not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set DEEPSEEK_API_KEY before --run (no key is persisted).")
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com")
    os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")


def _structured_content(result: Any) -> Any:
    """Read structured MCP content across SDK naming conventions."""

    content = getattr(result, "structuredContent", None)
    if content is None:
        content = getattr(result, "structured_content", None)
    return content


def _upstream(repo: Path):
    repo = Path(repo)
    repo = repo.expanduser().resolve()
    if not repo.exists():
        raise SystemExit("Set --agentabstain-repo to a local AgentAbstain checkout.")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from agent.openaisdk.agent import OpenAISDKAgent
    from agents import Agent, ModelSettings, Runner
    from agents import set_tracing_disabled
    from agents.models.openai_provider import OpenAIProvider
    from agents.mcp.server import MCPServerStdio
    from src.runtime.common import (
        RUNTIME_EXPORT_TOOL_NAME,
        build_runtime_server_args,
        build_task_run_result,
        coerce_final_output,
        normalize_runtime_export_payload,
    )
    from src.types.BaseAgent import BaseAgent
    set_tracing_disabled(True)
    return (OpenAISDKAgent, Agent, ModelSettings, Runner, MCPServerStdio, OpenAIProvider,
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


def _build_persisted_result(
    builder: Any,
    *,
    agent: Any,
    bundle: Any,
    artifact_dir: Any,
    final_output: Any,
    export_payload: Any,
    run_error: Any,
    provider_metadata: dict[str, Any],
) -> Any:
    """Keep experiment metadata attached to the official result artifact."""

    return builder(
        agent=agent,
        bundle=bundle,
        artifact_dir=artifact_dir,
        final_output=final_output,
        export_payload=export_payload,
        run_error=run_error,
        provider_metadata=provider_metadata,
    )


async def run_one(args: argparse.Namespace, task: str, side: str) -> dict[str, Any]:
    task_started = monotonic()
    (OpenAISDKAgent, Agent, ModelSettings, Runner, _MCPServer, OpenAIProvider, export_name,
     build_server_args, build_result, coerce_output, normalize_export, BaseAgent) = _upstream(args.agentabstain_repo)
    agent = OpenAISDKAgent(args.model, 0.0, args.max_turns, args.results_root)
    model = OpenAIProvider(use_responses=False).get_model(args.model)
    bundle = BaseAgent.load_task_bundle("conflicting_evidence", task, side)
    artifact_dir = agent.build_artifact_dir(bundle.category, bundle.task_id, bundle.task_type)
    server_type = build_contextcanon_server_class(args.agentabstain_repo)
    adapter = AgentAbstainAdapter() if args.extractor == "legacy" else None
    governance = None
    tool_semantics_resolver = None
    budget = None
    if args.extractor == "llm":
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ["OPENAI_API_KEY"]
        budget = SemanticBudget(
            max_requests=args.semantic_max_requests,
            max_total_seconds=args.semantic_max_total_seconds,
            per_request_deadline_seconds=args.semantic_request_deadline_seconds,
        )
        extraction_client = OpenAICompatibleExtractionClient(
            api_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
            timeout=args.semantic_request_deadline_seconds,
            inference=SemanticInferenceConfig(
                max_output_tokens=args.semantic_extraction_max_output_tokens
            ),
        )
        semantic_client = OpenAICompatibleExtractionClient(
            api_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
            timeout=args.semantic_request_deadline_seconds,
            inference=SemanticInferenceConfig(max_output_tokens=args.semantic_max_output_tokens),
        )
        extractor = LLMStructuredExtractor(
            BudgetedSemanticClient(extraction_client, budget, "extraction"), model=args.model
        )
        aligner = LLMEvidenceAligner(
            BudgetedSemanticClient(semantic_client, budget, "alignment"), model=args.model
        )
        relation_classifier = LLMRelationClassifier(
            BudgetedSemanticClient(semantic_client, budget, "relation"),
            model=args.model,
        )
        fact_need_extractor = LLMFactNeedExtractor(
            BudgetedSemanticClient(semantic_client, budget, "fact_need"),
            model=args.model,
        )
        tool_semantics_resolver = LLMToolSemanticsResolver(
            BudgetedSemanticClient(semantic_client, budget, "tool_semantics"),
            model=args.model,
        )
        governance = RuntimeGovernance(
            extractor=extractor,
            fact_need_extractor=fact_need_extractor,
            store=GovernanceStore(
                aligner=aligner,
                relation_classifier=relation_classifier,
            ),
            action_context=bundle.task_yaml["instruction"],
            budget=budget,
            tool_semantics_resolver=tool_semantics_resolver,
        )
    server = server_type(
        name="task_env",
        params={"command": sys.executable, "args": build_server_args(bundle), "cwd": str(args.agentabstain_repo), "env": official_server_env()},
        tool_filter={"blocked_tool_names": [export_name]},
        adapter=adapter,
        governance=governance,
        tool_semantics_resolver=tool_semantics_resolver,
        condition=args.condition,
    )
    final_output = None
    export_payload = None
    model_error = None
    runtime_export_error = None
    usage = None
    try:
        await server.connect()
        sdk_agent = Agent(
            name=f"{task}_{side}_agent",
            instructions=bundle.task_yaml["system_prompt"],
            model=model,
            model_settings=ModelSettings(
                temperature=0.0,
                extra_body={"thinking": {"type": "enabled"}},
                extra_args={"reasoning_effort": "high"},
            ),
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
            model_error = _format_exception("model_request", exc)
        try:
            exported = await server.call_runtime_control_tool(export_name, {})
            export_payload = normalize_export(_structured_content(exported))
        except Exception as exc:
            runtime_export_error = _format_exception("runtime_export", exc)
    finally:
        try:
            await server.cleanup()
        except BaseException:
            pass
    metadata: dict[str, Any] = {
        "provider": "openai-compatible",
        "base_url": os.environ.get("OPENAI_BASE_URL"),
        "model": args.model,
        "thinking": "enabled",
        "reasoning_effort": "high",
        "condition": args.condition,
        "semantic_extractor": args.extractor,
        "semantic_aligner": "llm" if governance is not None else "legacy-exact",
        "relation_classifier": "llm" if governance is not None else "legacy-rules",
        "fact_need_extractor": "llm" if governance is not None else "legacy-map",
        "tool_semantics_resolver": "llm" if governance is not None else "legacy-map",
        "contextcanon": True,
        "commit_attempted": server.contextcanon_bridge.diagnostics.commit_attempted,
        "commit_dispatched": server.contextcanon_bridge.diagnostics.commit_dispatched,
        "tracing": "disabled",
        "semantic_profile": {
            "thinking": None,
            "reasoning_effort": None,
            "max_output_tokens": args.semantic_max_output_tokens,
            "default_max_output_tokens": args.semantic_max_output_tokens,
            "extraction_max_output_tokens": args.semantic_extraction_max_output_tokens,
        },
        "task_wall_time_ms": round((monotonic() - task_started) * 1000),
    }
    if governance is not None:
        metadata["semantic_diagnostics"] = governance.metrics
        metadata["semantic_response_diagnostics"] = list(
            getattr(governance.extractor, "response_diagnostics", ())
        )
        metadata["extraction_diagnostics"] = list(
            getattr(governance.extractor, "diagnostics", ())
        )
        metadata["semantic_diagnostics"]["fast_path_alignment_hits"] = getattr(
            governance.store.aligner, "fast_path_hits", 0
        )
        metadata["semantic_diagnostics"]["fast_path_relation_hits"] = getattr(
            governance.store.relation_classifier, "fast_path_hits", 0
        )
        metadata["semantic_diagnostics"]["tool_semantics_cache_hits"] = getattr(
            tool_semantics_resolver, "cache_hits", 0
        )
        metadata["semantic_diagnostics"]["tool_semantics_cache_misses"] = getattr(
            tool_semantics_resolver, "cache_misses", 0
        )
    if usage is not None:
        metadata["usage"] = usage
    result = _build_persisted_result(
        build_result,
        agent=agent,
        bundle=bundle,
        artifact_dir=artifact_dir,
        final_output=final_output,
        export_payload=export_payload,
        run_error=model_error or runtime_export_error,
        provider_metadata=metadata,
    )
    return {
        "task_id": f"conflicting_evidence/{task}",
        "variant": side,
        "condition": args.condition,
        "artifact_dir": result.artifact_dir,
        "error": result.error,
        "model_error": model_error,
        "runtime_export_error": runtime_export_error,
        "final_output": final_output,
        "executed_tools": [
            entry.get("tool") for entry in (export_payload or {}).get("execution_log", [])
            if isinstance(entry, dict) and isinstance(entry.get("tool"), str)
        ],
        "diagnostics": server.contextcanon_bridge.diagnostics.to_dict(),
        "extraction_diagnostics": list(
            server.contextcanon_bridge.semantic_diagnostics
            or (
                server.contextcanon_bridge.adapter.extraction_diagnostics
                if server.contextcanon_bridge.adapter is not None
                else ()
            )
        ),
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
    _configure_provider_environment()
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
    parser.add_argument("--extractor", choices=EXTRACTORS, default="legacy")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--results-root", default="experiments/agentabstain/results")
    parser.add_argument("--run", action="store_true", help="required before making model/API calls")
    parser.add_argument("--semantic-max-requests", type=int, default=12)
    parser.add_argument("--semantic-max-total-seconds", type=float, default=30.0)
    parser.add_argument("--semantic-request-deadline-seconds", type=float, default=5.0)
    parser.add_argument(
        "--semantic-max-output-tokens",
        type=int,
        default=DEFAULT_SEMANTIC_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument(
        "--semantic-extraction-max-output-tokens",
        type=int,
        default=DEFAULT_SEMANTIC_EXTRACTION_MAX_OUTPUT_TOKENS,
    )
    args = parser.parse_args(argv)
    os.environ["AGENTABSTAIN_DATA"] = str(Path(args.agentabstain_data).expanduser())
    if not args.run:
        return _validate(args)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
