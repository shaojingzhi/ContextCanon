# ContextCanon M5.5 Evaluation Benchmark

This benchmark tests the hypothesis that supplying a deterministic ContextPackage
alongside the same raw repository material can improve correctness, conflict
recognition, traceability, and stability. It is a hypothesis, not an expected
result: the benchmark must not be tuned to guarantee that ContextCanon wins.

## Conditions

Each case is evaluated under two conditions with the same question, instructions,
raw source text, output schema, and model. Raw Context supplies only the source
files. ContextCanon supplies the exact same files plus the rendered package.
Prompts are exportable with:

```bash
python -m benchmarks.build_prompts --output benchmarks/prompts
```

No model SDK or API credentials are required. Saved predictions are JSON files
under `benchmarks/results/{raw,contextcanon}/`.

## Cases and scoring

There are exactly 15 frozen `auth.protocol` cases covering agreement, divergence,
runtime conflicts, intent conflicts, and ordering/noise. Ground truth is authored
in each `case.yaml` before any model run. Each prediction receives four binary
points: current accuracy, target accuracy, status accuracy, and traceability.
Traceability requires valid source citations for known current/target values and
all required conflict sources. Paths are normalized only by trimming whitespace,
converting backslashes, and removing `./`.

The scorer also reports conflict detection and overall points. Stability variants
use canonical, reverse, and a fixed even-then-odd permutation of source order.
A case is stable only when `(current, target, status)` is identical across all
three variants.

The benchmark does not evaluate retrieval quality: it assumes relevant repository
material is already supplied. It focuses only on `auth.protocol` because the
V0.1 extractor is intentionally narrow. It does not use an LLM judge, retrieval,
persistence, MCP, or live model execution.

