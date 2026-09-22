# Open-world Evidence Governance

ContextCanon does not attempt to understand every domain in advance.

The core therefore does not define an event, order, shelter, authentication,
or other business ontology. A `FactDescriptor` identifies a fact with an
open-ended subject, a natural-language semantic dimension, and an optional
small `TemporalScope`. A dimension such as “the date when this event takes
place” is valid without first registering an `event_date` property.

Evidence remains separate from fact identity. An `EvidenceCandidate` joins a
fact descriptor and value to traceable `Evidence`, including its source,
location, role, temporal scope, extraction confidence, verification state,
and any available validity timestamps. Multiple evidence items can therefore
support, conflict with, or supersede one fact without changing that fact's
identity.

## Semantic and deterministic boundaries

The ingestion flow is:

```text
observation -> semantic extraction -> evidence candidate
            -> fact alignment -> relation classification
            -> deterministic governance transition
```

`EvidenceAligner` answers whether incoming evidence describes the same fact,
a related but distinct fact, an unrelated fact, or an uncertain match.
`RelationClassifier` describes aligned evidence as supporting, equivalent,
conflicting, superseding, compatible, or unknown. `LLMEvidenceAligner` and
`LLMRelationClassifier` use one shared `SemanticModelClient`, but neither
selects final truth or authorizes an action.

The built-in aligner is deliberately conservative and lexical. The live
generic runtime uses the LLM aligner. Low-confidence or malformed alignment
becomes `UNKNOWN` and creates a separate fact instead of risking a false merge.
Before model alignment, the store applies a lexical subject filter and a
bounded recent-fact limit. Pairwise relation classification is likewise
bounded to recent active evidence. Large-scale semantic retrieval is future
work.

The deterministic `GovernanceStore` owns state transitions:

```text
conflicting active evidence -> DIVERGED
explicit superseding relation -> old evidence becomes inactive
no unexpired active evidence -> STALE
unknown evidence relation -> AMBIGUOUS
consistent active evidence -> RESOLVED
no usable evidence -> UNVERIFIED
```

`RESOLVED` is a consistency state: active evidence has no unresolved semantic
conflict or ambiguity. It does not assert absolute truth and does not require
`Evidence.verified is True`. Verification remains an orthogonal quality signal
that a stricter consumer policy may inspect.

Current and future evidence are kept distinct by temporal scope, so a current
JWT deployment and a future OAuth2 intent do not become a runtime conflict.
Freshness reads an injectable live clock on every ingestion, refresh, query,
and action evaluation. Evidence is active only after `valid_from` and through
`valid_until`. Malformed timestamps produce ambiguous freshness rather than an
invented time.

## Consumers

Ingestion records evidence and recomputes state. Query governance returns the
state, candidate values, evidence references, and sources; a diverged query
therefore exposes both sides instead of silently choosing one. Action
governance reads the same state and deterministically requires clarification
when a required fact is missing, stale, ambiguous, unverified, or diverged.

`FactNeed` describes a fact needed by either a query or an action.
`LLMFactNeedExtractor` derives these ephemeral needs only from runtime-visible
tool details and task context; it never emits the final action decision. Dynamic
dependencies describe what the current consumer needs; they are not a
permanent global ontology.

The intended live path is now:

```text
RuntimeObservation
-> LLMStructuredExtractor
-> EvidenceCandidate
-> LLMEvidenceAligner
-> LLMRelationClassifier
-> GovernanceStore
-> Query / Action policy
```

Semantic conflict recognition may use an LLM. The transition from a semantic
relation to `GovernanceState`, evidence lifecycle changes, and action policy is
deterministic: **semantic conflict recognition + deterministic governance
transition**.

The AgentAbstain `LegacyRuleExtractor` remains only for offline deterministic
tests, backward compatibility, and Gate 1–3 validation. It is not the intended
generic runtime architecture.

**Probabilistic understanding, deterministic governance.**

## Deliberate limits

M8.2 remains an in-memory architecture slice, not a knowledge graph, temporal
database, RAG system, or ontology platform. It does not perform generic entity
linking, automatic ontology evolution, semantic embedding retrieval, or
automatic entity linking, or large-scale candidate retrieval. It also does not
run the paid AgentAbstain benchmark.
