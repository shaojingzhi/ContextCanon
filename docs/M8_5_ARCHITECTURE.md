# M8.5 online latency architecture

## Before

The live bridge handled a non-baseline read/verify call as:

```text
tool semantics (possibly synchronous model call)
→ MCP dispatch
→ extraction
→ alignment
→ relation classification
→ governance store update
→ evidence injection
→ return to Agent
```

The model calls were synchronous. Governance ran in a worker thread, so it did
not block the asyncio event loop, but the MCP call still awaited the complete
serial governance chain. Tool semantics were first resolved on the first tool
call, and there was no pending-work or bounded side-effect barrier.

## After

Tool discovery snapshots stable metadata and primes the task-local semantics
cache. Normal calls use the cached classification; unresolved metadata remains
fail-closed as `UNKNOWN`.

The read path is now:

```text
cached tool semantics
→ MCP dispatch
→ capture raw observation
→ schedule task-local governance work
→ inject currently available evidence
→ return to Agent
```

Background work has explicit `PENDING`, `COMPLETE`, and `FAILED` states, keeps
raw observations and provenance, and records exceptions. The worker is
serialized around the existing in-memory governance store so state transitions
remain deterministic.

The side-effect path is a bounded governance barrier:

```text
FactNeed extraction
→ wait for possibly relevant pending work
→ evaluate the existing GovernanceStore
→ ALLOW or REQUIRE_CLARIFICATION
```

Timeouts, failed background work, incomplete semantic budgets, and unresolved
governance all fail closed. No side effect is dispatched while relevant work is
unfinished. The existing task-local caches for tool semantics and alignment are
preserved; relation results now also use stable-input caching.

M8.5 does not add persistence, a generic queue, a plugin system, or a benchmark
specific rule. Batch extraction is intentionally deferred because the current
extractor contract is one observation per provenance-bearing call.
