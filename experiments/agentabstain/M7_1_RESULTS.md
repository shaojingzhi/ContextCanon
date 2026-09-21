# M7.1 real Agent loop integration

Status: implementation complete; live model gates pending an explicit
DeepSeek key.

## Scope and provenance

- Upstream AgentAbstain checkout: `f581249704b26804e28a39e37396f1be00b71a4d`
- Official server entry point: `python -m src.runtime.task_mcp_server`
- Model configuration: `deepseek-v4-pro`, `https://api.deepseek.com`,
  `DEEPSEEK_API_KEY` mapped to `OPENAI_API_KEY` in memory.
- Conditions: baseline, governed, guard; six task variants only.

## Gates

1. The wrapper uses the official runtime server arguments and upstream
   OpenAI-safe tool-name encoding/decoding. The upstream MCP schema does not
   expose tool kind metadata, so the six-task spike uses an explicit bounded
   mapping for known tools.
2. Lookup/verify results are intercepted after MCP dispatch. MCP-level
   `isError` results are recorded as failed observations and do not create
   claims. Governed results receive a runtime-evidence block before the next
   model turn.
3. Commit calls are evaluated before dispatch; guard conflicts return an MCP
   error result without calling the upstream server.
4. Real DeepSeek smoke: **not run** (`DEEPSEEK_API_KEY` was unavailable).
5. Six guard rollouts: **not run**; they require Gate 4 and a paid API key.

The deterministic official-server gate can be run without a model:

```bash
python -m experiments.agentabstain.official_gate
```

No model output or paid benchmark result is claimed by this spike.

## Recommendation

CONDITIONAL GO — the official MCP bridge and local guard gates are ready; run
the single abstain smoke before authorizing the six guard rollouts.
