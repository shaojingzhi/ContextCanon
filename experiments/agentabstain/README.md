# AgentAbstain adapter spike

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

