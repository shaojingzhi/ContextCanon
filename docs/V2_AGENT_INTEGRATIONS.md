# ContextCanon V2 agent integrations

ContextCanon V2 keeps agent orchestration outside its governance core. Both the
OpenAI Agents SDK demonstration and the LangGraph demonstration call the same
`GovernedToolProxy`; neither framework is represented in the V2 models or
governance rules.

```text
OpenAI Agents SDK --\
                     > GovernedToolProxy -> backend tools
LangGraph ----------/
```

The LangGraph example uses a `StateGraph` model/tool loop. Both its offline
scripted model and its opt-in live DeepSeek runner select tools through the graph.
In the live smoke, the model selected two read tools and then attempted a
deployment. The read results became `Evidence`; semantic compilation identified
the required `stable` channel and observed `canary` channel, and deterministic
governance blocked the side effect as `UNRESOLVED` before backend dispatch.

The live prompt intentionally instructed the model to attempt `deploy_release`
after both reads even if they conflicted. This was a path-coverage smoke proving
the real model -> LangGraph runtime -> Evidence -> `GovernedToolProxy` ->
`SemanticCompiler` -> deterministic governance -> blocked backend chain. It does
not claim that an unconstrained agent would naturally attempt an unsafe side
effect after recognizing a conflict.

Here, “LangGraph integration” means a LangGraph `StateGraph` with custom agent
and tool nodes calling the existing proxy. It does not claim LangChain native
`@tool`, LangGraph `ToolNode`, or LangChain MCP adapter integration. The purpose
is to demonstrate runtime/framework independence of `GovernedToolProxy`, not
coverage of every native LangChain tool API.

## Architectural comparison

These are complementary architectural emphases, not benchmark results or a
claim that one project is generally better than another.

| System | Primary focus | Typical governed object |
| --- | --- | --- |
| ContextCanon | Evidence semantics, provenance, typed claims, and conflict-aware decisions | Evidence required before a side-effect tool call |
| Invariant Guardrails | Declarative rules over agent traces, data flow, and tool-call policy | Messages, tool calls, tool outputs, and their ordering/data flow |
| NVIDIA NeMo Guardrails | Broad input, retrieval, dialog, execution, and output rails | Multiple stages around an LLM application and its tools |

ContextCanon is deliberately narrower: it compiles evidence into typed claims
and blocks a side effect when the required facts are incomplete or unresolved.
Invariant can enforce explicit behavioral and security policies across an agent
trace. NeMo Guardrails provides a broader configurable guardrail runtime across
the interaction lifecycle. A deployment may use these concerns together.

References: [Invariant Guardrails](https://github.com/invariantlabs-ai/invariant),
[NVIDIA NeMo Guardrails architecture](https://docs.nvidia.com/nemo/guardrails/latest/about-nemo-guardrails-library/how-it-works.html),
and the [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api).

## Size and boundaries

At this demonstration revision, `contextcanon_v2` is approximately 742 lines in
total and `GovernedToolProxy` is approximately 88 lines. Legacy implementations,
benchmarks, and experiment runners are not part of that V2 core count. The
LangGraph package is an optional demonstration dependency; the V2 core does not
import it. The live experiment reuses the legacy package's OpenAI-compatible
client only as experiment-side provider plumbing; `contextcanon_v2` neither
imports nor depends on the legacy runtime.
