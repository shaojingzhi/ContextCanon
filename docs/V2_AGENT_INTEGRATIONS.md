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

The LangGraph example uses a `StateGraph` model/tool loop. Its offline scripted
model selects two read tools and then attempts a deployment. Read results become
`Evidence`; the side effect is dispatched only when the existing semantic
compiler and deterministic governance return `CLEAR`.

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
import it.
