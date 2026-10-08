# ContextCanon V2

[中文](README.zh-CN.md)

> Compile conflicting evidence into a small, traceable decision before an agent
> performs a side effect.

## Why this exists

ContextCanon V2 turns heterogeneous evidence into typed, provenance-backed
claims and applies deterministic conflict governance. It sits at the tool
boundary so an agent can read freely while side effects require `CLEAR`
knowledge. It does not own the agent loop, retrieval system, or tool backend.

The current implementation is the small [`contextcanon_v2/`](contextcanon_v2/)
package: about 742 physical lines, including an approximately 88-line
`GovernedToolProxy`. The older [`contextcanon/`](contextcanon/) package is the
legacy research prototype; experiments and tests are not part of the V2 core.

## Architecture

```text
Knowledge sources / runtime tool results
                  |
                  v
              Evidence
                  |
                  v
          SemanticCompiler
                  |
                  v
            typed Claims
                  |
                  v
       deterministic govern()
                  |
                  v
     CLEAR / UNRESOLVED / INCOMPLETE
```

Two independent agent runtimes use the same boundary:

```text
OpenAI Agents SDK --\
                     > GovernedToolProxy -> backend tools
LangGraph ----------/
```

The agent runtime chooses tools and controls its model loop. ContextCanon only
captures successful READ results as `Evidence`, compiles the facts needed by a
SIDE_EFFECT tool, and either dispatches or blocks that tool.

## Integration evidence

### OpenAI Agents SDK + AgentAbstain

One live integration smoke used a real OpenAI Agents SDK loop and real
AgentAbstain MCP tools. The agent selected six READ/VERIFY calls; all six tool
results became V2 `Evidence`. It independently found the March 22 versus March
23 event-date conflict and abstained before attempting the message-send tool.

That run proves the real Agent/MCP-to-Evidence path, but it did **not** exercise
the live pre-dispatch blocking branch because the agent never attempted the
side effect. Deterministic integration tests separately prove all three gate
outcomes:

- `CLEAR` -> dispatch
- `UNRESOLVED` -> block before backend dispatch
- `INCOMPLETE` -> block before backend dispatch

### LangGraph

[`experiments/v2_langgraph_demo.py`](experiments/v2_langgraph_demo.py) is a
second, independent runtime integration using LangGraph's graph/model/tool
loop and the same `GovernedToolProxy`. It required no V2 core changes.

An opt-in [live runner](experiments/v2_langgraph_live.py) used DeepSeek for
three LangGraph tool-selection turns.
The model selected `read_policy`, `read_runtime_state`, then attempted
`deploy_release`; two runtime results joined the static policy as three pieces
of `Evidence`. A fourth model call performed semantic compilation, after which
deterministic governance returned `UNRESOLVED` and blocked deployment before
the backend received the side-effect call. The run completed without retry or
error, and `side_effect_dispatched` was false.

## Engineering regression story

The frozen V2 RAMDocs run completed 78.8% of 500 cases, with 106 semantic
compilation failures. Trace analysis localized most failures to strict typed
representations: 82 date values and 20 numeric values were semantically useful
but not in the accepted deterministic form.

V2.2 added small generic normalization before existing strict Claim validation.
On the RAMDocs engineering regression, completion reached 91.6%; date
typed-value failures fell from 82 to 0 and number typed-value failures from 20
to 0. Governance, Gold mappings, and scoring stayed unchanged.

**This is an engineering regression on a dataset whose failure traces informed
V2.2, not untouched external validation.**

## Scope relative to guardrail systems

This is an architectural comparison, not a benchmark or superiority claim.

| System | Primary focus |
| --- | --- |
| ContextCanon | Evidence semantics, provenance, typed claims, and conflict-aware governance |
| Invariant Guardrails | Rule-based policy over agent traces, data flow, and tool calls |
| NVIDIA NeMo Guardrails | Broad input, retrieval, dialog, execution, and output rails |

These concerns can be complementary. See
[`docs/V2_AGENT_INTEGRATIONS.md`](docs/V2_AGENT_INTEGRATIONS.md) for the concise
integration and comparison notes.

## Repository map

| Path | Role |
| --- | --- |
| [`contextcanon_v2/`](contextcanon_v2/) | Current V2 models, compiler, deterministic governance, and tool proxy |
| [`experiments/agentabstain/`](experiments/agentabstain/) | OpenAI Agents SDK + MCP integration proof |
| [`experiments/v2_langgraph_demo.py`](experiments/v2_langgraph_demo.py) | Framework-independence proof with LangGraph |
| [`benchmarks/`](benchmarks/) | Frozen development, holdout, and public-dataset fixtures/adapters |
| [`tests/`](tests/) | Deterministic behavior and integration coverage |
| [`contextcanon/`](contextcanon/) | Legacy research implementation retained for comparison |

## Offline validation

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q .
```

The LangGraph demo is an optional dependency:

```bash
python3 -m pip install -e '.[langgraph-demo]'
python3 -m unittest tests.test_v2_langgraph_demo -v
```
