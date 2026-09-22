# Semantic Extraction

The first AgentAbstain spike used task-specific rules to validate the evidence
ledger, deterministic conflict detection, and pre-action guard. M8 adds a
small generic boundary for turning a runtime observation into structured
`ClaimCandidate` values.

The boundary is:

```text
RuntimeObservation -> SemanticExtractor -> ClaimCandidate
                   -> deterministic normalization -> RuntimeClaim/Evidence
                   -> deterministic conflict detection and guard
```

`ClaimCandidate` contains an entity, property, value, value type, semantic
evidence role, extractor confidence, and system-supplied provenance. The model
does not create trusted observation identifiers or decide whether an action is
safe. Malformed, unsupported, ambiguous, or low-confidence extraction returns
no claim and records a diagnostic.

The default `LegacyRuleExtractor` remains available for deterministic offline
fixtures and as a fallback. `LLMStructuredExtractor` accepts any callable
OpenAI-compatible Chat Completions client, so provider details stay outside the
governance layer. The live runner exposes this choice with `--extractor
legacy|llm`; the default remains `legacy` until the generic extractor is
validated in a separately authorized smoke run.

Normalization is deliberately small: strings are trimmed, dates accept common
ISO, US numeric, and English month forms and become `YYYY-MM-DD`, datetimes
become ISO strings, booleans accept only explicit true/false forms, numbers are
parsed as finite numeric values, and enum values are case-folded. No generic
entity-linking, ontology evolution, fuzzy clustering, or action-safety
judgment is included.

**Probabilistic understanding, deterministic governance.**

M8 does not solve generic entity linking or ontology evolution.
