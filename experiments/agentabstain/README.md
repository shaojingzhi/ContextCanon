# AgentAbstain adapter spike

## Milestone boundaries

M6 uses synthetic observations. M7 replays the real AgentAbstain environment
with deterministic tool calls. M7.1 adds an opt-in model loop through the
official `src.runtime.task_mcp_server` startup path; it does not change the
upstream checkout or benchmark evaluator.

The live runner is deliberately not automatic:

```bash
python -m experiments.agentabstain.run_agent --task preview_008 --side abstain
python -m experiments.agentabstain.official_gate
DEEPSEEK_API_KEY=... python -m experiments.agentabstain.run_agent \
  --task preview_008 --side abstain --condition guard --run
```

Use `--task all --side all --condition guard --run` only when explicitly
authorizing all six paid model rollouts. The runner uses the OpenAI-compatible
DeepSeek endpoint (`OPENAI_BASE_URL=https://api.deepseek.com`) and maps
`DEEPSEEK_API_KEY` to the SDK's `OPENAI_API_KEY` in memory only. API keys are
never written to result metadata.

The tool-kind map in `openai_runtime.py` is intentionally limited to this
spike because the upstream MCP schema does not expose the benchmark's internal
lookup/verify/commit labels. It is not a registry or a generic extraction
framework.

This spike proves a narrow runtime boundary for three public AgentAbstain S7
task shapes:

- `conflicting_evidence/preview_008` (event date / SMS)
- `conflicting_evidence/preview_013` (order lifecycle / return)
- `conflicting_evidence/preview_015` (shelter status / announcement)

Only runtime observations are accepted: tool name, tool kind, parameters,
result, success/error, and call index. Benchmark labels and evaluator metadata
are rejected before they reach the governance layer.

The flow is:

```text
runtime observation → task-local claim/evidence → conflict record → commit guard
```

The extraction rules are deliberately task-specific and are not the final
generic ContextCanon runtime evidence extraction design. The spike does not
prove benchmark performance improvement. The main unsolved problem is generic
semantic claim extraction from arbitrary tool observations.

Run a deterministic replay without API keys:

```text
python -m experiments.agentabstain.demo preview_008 --scenario conflict
python -m experiments.agentabstain.demo preview_008 --scenario consistent
```

The same `consistent` / `conflict` scenarios are supported for the other two
task identifiers.

## M7 real-harness integration

The real-harness spike was inspected against AgentAbstain upstream commit
`f581249704b26804e28a39e37396f1be00b71a4d`. It keeps the upstream checkout and
dataset outside this repository:

```text
git clone https://github.com/AntiQuality/agentabstain.git /tmp/agentabstain
cd /tmp/agentabstain
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('antiquality/agentabstain', repo_type='dataset', local_dir='data')"
```

For an OpenAI-compatible DeepSeek endpoint, configure the upstream process
without committing the key:

```text
export AGENTABSTAIN_REPO=/tmp/agentabstain
export AGENTABSTAIN_DATA=/tmp/agentabstain/data
export OPENAI_BASE_URL=https://api.deepseek.com
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
```

The deterministic real-environment gate runner performs only selected
read-only MCP calls and then exercises the actual environment MCP dispatch
object through the bridge:

```text
PYTHONPATH=/path/to/ContextCanon \
  .venv/bin/python -m experiments.agentabstain.run_real --condition guard
```

Use `--task`, `--side`, and `--condition` to narrow the run. `baseline` does
not send observations to ContextCanon, `governed` observes and renders
evidence without blocking, and `guard` also withholds a conflicting commit
before dispatch. This command covers all six variants and records only a
small JSON diagnostic summary; raw rollout artifacts are not stored here.

The bridge boundary is:

```text
real MCP result → RuntimeObservation → AgentAbstainAdapter
proposed commit → ProposedToolCall → ALLOW / REQUIRE_CLARIFICATION
```

The optional 18-rollout model experiment (six variants × baseline/governed/
guard) is intentionally not run by this spike command. A provider-specific
Agent SDK loop still needs to inject `render_governed_evidence()` between MCP
turns; the pre-dispatch guard itself is independently testable at the real MCP
boundary. The OpenAI-compatible route is the selected provider; no OpenRouter
dependency is required.

This M7 integration still uses task-specific semantic extraction for these
three AgentAbstain pairs. It does not solve generic semantic extraction, and
it does not establish benchmark performance improvement.
