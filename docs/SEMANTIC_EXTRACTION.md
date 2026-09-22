# Semantic Extraction

The first AgentAbstain spike used task-specific rules to validate the evidence
ledger, deterministic conflict detection, and pre-action guard. M8 adds a
small generic boundary for turning a runtime observation into structured
`ClaimCandidate` values.

The boundary is:

```text
RuntimeObservation -> SemanticExtractor -> ClaimCandidate
                   -> deterministic normalization -> EvidenceCandidate
                   -> semantic alignment/relation recognition
                   -> deterministic governance transition and guard
```

`ClaimCandidate` contains a subject, open-ended semantic dimension, value,
value type, semantic evidence role, temporal scope, extractor confidence, and
system-supplied provenance. The current Python field remains named `property`
for M8 compatibility, but it is not a registered or fixed business schema key.
The model
does not create trusted observation identifiers or decide whether an action is
safe. Malformed, unsupported, ambiguous, or low-confidence extraction returns
no claim and records a diagnostic.

The default `LegacyRuleExtractor` remains available for deterministic offline
fixtures and as a fallback. `LLMStructuredExtractor` accepts any callable
OpenAI-compatible Chat Completions client, so provider details stay outside the
governance layer. The live runner exposes this choice with `--extractor
legacy|llm`; the default remains `legacy` until the generic extractor is
validated in a separately authorized smoke run.

Open-world alignment and deterministic governance are described in
[`OPEN_WORLD_GOVERNANCE.md`](OPEN_WORLD_GOVERNANCE.md). `LegacyRuleExtractor`
is a compatibility mechanism, not the intended generic runtime architecture.

Normalization is deliberately small: strings are trimmed, dates accept common
ISO, US numeric, and English month forms and become `YYYY-MM-DD`, datetimes
become ISO strings, booleans accept only explicit true/false forms, numbers are
parsed as finite numeric values, and enum values are case-folded. No generic
entity-linking, ontology evolution, fuzzy clustering, or action-safety
judgment is included.

**Probabilistic understanding, deterministic governance.**

M8 does not solve generic entity linking or ontology evolution.
