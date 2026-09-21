# M7.1 real Agent loop integration

Status: official-server Gates 1–3 pass; live model Gate 4 has not been run.

## Scope and provenance

- Upstream AgentAbstain checkout: `f581249704b26804e28a39e37396f1be00b71a4d`
- Official server entry point: `python -m src.runtime.task_mcp_server`
- Model configuration: `deepseek-v4-pro`, `https://api.deepseek.com`,
  `DEEPSEEK_API_KEY` mapped to `OPENAI_API_KEY` in memory.
- Conditions: baseline, governed, guard; six task variants only.

## Gates

1. **PASS.** The wrapper uses the official runtime server arguments and upstream
   OpenAI-safe tool-name encoding/decoding. The upstream MCP schema does not
   expose tool kind metadata, so the six-task spike uses an explicit bounded
   mapping for known tools.
2. **PASS.** Lookup/verify results are intercepted after MCP dispatch. MCP-level
   `isError` results are recorded as failed observations and do not create
   claims. Governed results receive a runtime-evidence block before the next
   model turn.
3. **PASS.** Commit calls are evaluated before dispatch; guard conflicts return an MCP
   error result without calling the upstream server.
4. Real DeepSeek smoke: **not run** (`DEEPSEEK_API_KEY` was unavailable and is
   intentionally not requested in M7.2).
5. Six guard rollouts: **not run**; they require Gate 4 and a paid API key.

The pinned release expects the downloaded dataset's `environments/` packages
to be exposed through `abstention_factory.environments.__path__`. The stdio
client intentionally inherits only a safe environment subset, so
`AGENTABSTAIN_DATA` must be passed explicitly in the server subprocess
parameters. M7.2 applies that transparent setup fix; no upstream files or
benchmark artifacts are modified. The registry imports all 42 environment
classes, and the official server lists tools for all six selected variants
with encoded-name round trips.

The deterministic official-server gate was run without a model:

```bash
python -m experiments.agentabstain.official_gate
```

All six variants loaded, listed tools, produced ContextCanon observations, and
showed the expected ALLOW/dispatch versus REQUIRE_CLARIFICATION/withheld commit
behavior. No model output or paid benchmark result is claimed.

## Recommendation

CONDITIONAL GO — the official MCP bridge and local guard gates are ready; run
the single abstain smoke before authorizing the six guard rollouts.
