# ContextCanon Development Instructions

Before making architectural changes:
- Read docs/SPEC.md.
- Read docs/MILESTONES.md.

Engineering priorities:
1. Keep the core small and easy to understand.
2. Do not implement future milestones early.
3. Prefer deterministic behavior over LLM judgment.
4. Avoid new infrastructure unless the current milestone requires it.
5. Avoid abstractions without a concrete current use case.
6. Keep dependencies minimal.
7. Add or update tests for every meaningful behavior change.
8. Run the relevant test suite before finishing.

Milestone discipline:
- Work only on the milestone explicitly requested by the user.
- Do not silently start the next milestone.
- If the current milestone exposes a flaw in SPEC.md, explain it before making a major redesign.
- At completion, summarize:
  - files changed
  - architectural decisions
  - tests run
  - deviations from the spec
  - remaining limitations