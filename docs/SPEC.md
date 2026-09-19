# ContextCanon V0.1

**Working tagline**

> Compile conflicting knowledge into verifiable context for AI agents.

**Core question**

> When multiple sources disagree, what should an AI agent believe right now — and why?

---

## 1. Project Positioning

ContextCanon is a lightweight **knowledge resolution and context compilation layer for coding agents**.

It is not primarily a RAG framework, vector database, knowledge graph, memory system, or agent framework.

Its job is to transform fragmented and potentially conflicting project knowledge into a small, explicit, traceable and, where possible, machine-verifiable context package that an AI agent can safely consume.

The core workflow is:

```text
Sources
  ↓
Claims + Evidence
  ↓
Verification
  ↓
Resolution Policy
  ↓
Resolved Knowledge State
  ↓
Context Compilation
  ↓
Agent Context
```

---

## 2. Target User

V0.1 targets:

**Developers using coding agents on repositories containing inconsistent or evolving project knowledge.**

Typical agents include:

* Pi
* DeepSeek Harness
* MCP-compatible coding agents
* Other future coding-agent runtimes

V0.1 is deliberately not a generic enterprise knowledge-management system.

---

## 3. Problem

Real repositories often contain contradictory knowledge.

Example:

```text
README.md
Authentication uses JWT.

ADR-015.md
Accepted decision: migrate authentication to OAuth2.

config/auth.yaml
provider: jwt

src/auth/provider.py
JWTProvider

tests/test_auth.py
Tests current JWT behaviour.
```

A conventional retrieval system may return all of these documents and leave the model to infer the truth.

ContextCanon instead distinguishes:

```text
Current observed state:
JWT

Accepted architecture intent:
OAuth2

Knowledge state:
DIVERGED
```

The system explains why and preserves the conflict instead of inventing consensus.

---

## 4. Core Principles

### 4.1 Claims are separate from evidence

A claim represents knowledge:

```text
auth.protocol = JWT
```

Evidence explains why the claim may be trusted:

```text
config/auth.yaml
src/auth/provider.py
tests/test_auth.py
```

Multiple pieces of evidence may support one claim.

---

### 4.2 Authority is contextual

ContextCanon must never assume a universal rule such as:

```text
code > ADR > docs
```

Different questions require different evidence policies.

For current runtime state:

```text
runtime config
→ executable tests
→ current implementation
→ documentation
→ future architectural intent
```

For architectural intent:

```text
accepted ADR
→ architecture specification
→ implementation
→ documentation
```

Therefore resolution depends on:

```text
query intent
+
claim type
+
evidence role
+
verification state
```

---

### 4.3 Verification is preferred over heuristic confidence

Whenever possible, ContextCanon should verify claims using deterministic mechanisms.

Examples:

```text
YAML / JSON lookup
AST inspection
file existence
configuration inspection
tests
structured source inspection
```

LLM confidence must not substitute for executable evidence where deterministic verification is available.

---

### 4.4 Never invent consensus

ContextCanon does not have to select a winner.

Valid knowledge states include:

```text
RESOLVED
DIVERGED
AMBIGUOUS
UNVERIFIED
SUPERSEDED
```

When evidence is insufficient or equally authoritative sources disagree, the conflict must remain explicit.

---

### 4.5 Model owns interpretation; system owns invariants

LLMs may help with:

```text
claim extraction
semantic classification
query intent classification
natural-language explanation
```

Deterministic code owns:

```text
provenance
policy application
verification
resolution state
source revisions
context-package generation
```

The LLM must not silently override resolution policy.

---

### 4.6 Small core, replaceable components

V0.1 should prefer:

```text
Protocol / ABC
+
dependency injection
```

over building a custom plugin framework.

Core mechanisms should remain independent from specific:

```text
LLM providers
vector databases
agent frameworks
storage engines
```

---

## 5. V0.1 Supported Sources

V0.1 supports local Git repositories containing:

```text
Markdown documents
ADRs
YAML configuration
JSON configuration
source code
tests
```

The initial implementation may give richer semantic extraction to Markdown, ADR, YAML and JSON than arbitrary source languages.

Source-code support may initially focus on provenance and basic structured inspection rather than full language-semantic analysis.

---

## 6. Explicit Non-Goals

V0.1 will NOT implement:

```text
Slack integration
Confluence integration
Google Drive integration
OKR systems
RBAC / ACL
multi-tenancy
distributed ingestion
Kafka
Redis
Neo4j
custom vector database
Web UI
generic enterprise search
multi-agent orchestration
full ontology management
custom plugin lifecycle system
```

These must not be added unless required by the core V0.1 workflow.

---

# 7. Core Data Model

## 7.1 Claim

A Claim represents one semantic assertion.

Conceptually:

```python
Claim(
    id,
    subject,
    predicate,
    value,
    claim_type,
    evidence,
    confidence,
)
```

Initial claim types:

```text
RUNTIME_STATE
ARCHITECTURE_INTENT
OPERATIONAL_RULE
```

Example:

```text
subject:
auth

predicate:
protocol

value:
JWT

claim_type:
RUNTIME_STATE
```

---

## 7.2 Evidence

Evidence supports or contradicts a Claim.

Conceptually:

```python
Evidence(
    id,
    source_id,
    source_type,
    location,
    role,
    content,
    timestamp,
    verifier,
    verified,
)
```

Initial evidence roles:

```text
OBSERVED
INTENDED
DOCUMENTED
```

Examples:

```text
config/auth.yaml
→ OBSERVED + potentially machine-verified

ADR-015
→ INTENDED

README.md
→ DOCUMENTED
```

---

## 7.3 Resolution

Resolution represents the governance result for competing knowledge.

Statuses:

```text
RESOLVED
DIVERGED
AMBIGUOUS
UNVERIFIED
SUPERSEDED
```

Conceptually:

```python
Resolution(
    status,
    selected_claims,
    conflicting_claims,
    reason_codes,
    explanation,
    policy,
    policy_version,
)
```

Initial reason codes:

```text
VERIFIED_EVIDENCE
HIGHER_ROLE_PRIORITY
INTENT_IMPLEMENTATION_DIVERGENCE
CONFLICTING_AUTHORITATIVE_EVIDENCE
INSUFFICIENT_EVIDENCE
SUPERSEDED_BY_NEWER_CLAIM
```

Natural-language explanation is useful, but machine-readable reason codes are canonical.

---

# 8. Resolution Policy

ResolutionPolicy is replaceable.

Conceptual interface:

```python
class ResolutionPolicy(Protocol):
    name: str
    version: str

    def resolve(
        self,
        claims: list[Claim],
    ) -> Resolution:
        ...
```

V0.1 resolves the claim types already present on each Claim. Query-specific
resolution context is deferred until task-aware context compilation requires
it.

V0.1 implements exactly one:

```text
DefaultResolutionPolicy
```

No DSL is required.

No dynamic plugin loading is required.

---

## 8.1 Runtime-state policy

When the intent is to understand current behaviour, prioritize evidence approximately as:

```text
machine-verified evidence
→ OBSERVED
→ DOCUMENTED
→ INTENDED
```

Architectural intent must not override verified runtime reality.

---

## 8.2 Architecture-intent policy

When determining intended architecture:

```text
INTENDED
→ machine-verified evidence
→ OBSERVED
→ DOCUMENTED
```

An accepted architectural decision may describe the desired state even when implementation has not converged.

---

## 8.3 Divergence

If:

```text
verified runtime state != accepted architecture intent
```

the result is:

```text
DIVERGED
```

Example:

```text
Current:
JWT

Target:
OAuth2

Status:
DIVERGED
```

Neither Claim is discarded.

---

## 8.4 Ambiguity

If two comparable current authoritative sources disagree and the system lacks evidence to determine which is effective:

```text
AMBIGUOUS
```

The system must not ask an LLM to arbitrarily select one.

---

# 9. Verification

Verification is optional per Evidence item.

Verification checks whether recorded Evidence can still be reproduced from
its source and location. It validates Evidence provenance; it does not
reinterpret or validate the normalized semantic truth in `Claim.value`.

Conceptual interface:

```python
class Verifier(Protocol):
    def verify(
        self,
        evidence: Evidence,
        document: SourceDocument,
    ) -> VerificationResult:
        ...
```

Verification results have three outcomes:

```text
VERIFIED
→ the current source reproduces the recorded Evidence

FAILED
→ verification completed normally, but the current source no longer matches
  the Evidence

ERROR
→ verification could not be completed
```

Initial deterministic verifiers may include:

```text
YamlPathVerifier
JsonPathVerifier
TextPresenceVerifier
```

Potential later verifiers:

```text
ASTVerifier
TestVerifier
CommandVerifier
RuntimeVerifier
```

V0.1 should prioritize simple, reliable verification over cleverness.

---

# 10. ContextPackage

ContextPackage is the primary output artifact.

It is structured data, not a prompt string.

Conceptually:

```python
ContextPackage(
    id,
    task,
    items,
    unresolved_conflicts,
    source_revision,
    policy_name,
    policy_version,
    token_budget,
    created_at,
)
```

Each ContextItem includes:

```text
claim
resolution status
supporting evidence
reason codes
explanation
```

---

## 10.1 Reproducibility

Each ContextPackage should record at least:

```text
Git revision
policy version
selected claims
task/query
```

Package IDs should preferably derive from stable input hashes rather than random UUIDs.

Conceptually:

```text
hash(
    task
    + source_revision
    + selected_claim_ids
    + policy_version
)
```

This allows two agent executions to compare exactly what knowledge was available.

---

## 10.2 ContextPackage is not tied to an agent

Use renderers/adapters:

```text
ContextPackage
      ↓
┌─────┼─────────┐
JSON  Markdown  MCP
                │
                ↓
              Agents
```

Conceptual interface:

```python
class ContextRenderer(Protocol):
    def render(self, package: ContextPackage) -> str:
        ...
```

V0.1 implements:

```text
JSONRenderer
MarkdownRenderer
```

Agent-specific integration stays outside the core.

---

# 11. Context Assembly

ContextAssembler selects resolved knowledge within a token budget.

V0.1 uses a simple deterministic policy.

Approximate priority:

```text
supported by machine-verified evidence
↓
RESOLVED
↓
DIVERGED
↓
UNVERIFIED
```

Within the same status, retrieval relevance may determine ordering.

No sophisticated LLM context compression is required in V0.1.

---

# 12. Architecture

```text
                 Repository
                     │
                     ▼
               SourceLoader
                     │
                     ▼
               ClaimExtractor
                     │
                     ▼
              Claim + Evidence
                     │
                     ▼
                  Verifier
                     │
                     ▼
             ResolutionPolicy
                     │
                     ▼
                 Resolution
                     │
                     ▼
                  Storage
                     │
                     ▼
                 Retriever
                     │
                     ▼
             ContextAssembler
                     │
                     ▼
               ContextPackage
                     │
             ┌───────┼───────┐
             ▼       ▼       ▼
           JSON   Markdown   MCP
```

---

# 13. Extension Interfaces

V0.1 should expose small Python interfaces for:

```text
SourceLoader
ClaimExtractor
Verifier
ResolutionPolicy
Storage
Retriever
ContextAssembler
ContextRenderer
```

Avoid introducing:

```text
PluginManager
EventBus
DependencyGraph
LifecycleManager
Manifest system
```

until real use cases require them.

---

# 14. Default Storage

V0.1 is local-first.

Recommended default:

```text
SQLite
```

Optional filesystem JSON artifacts may be used for generated ContextPackages.

Core must not depend on:

```text
Qdrant
Pinecone
Milvus
Elasticsearch
Neo4j
PostgreSQL
```

Future implementations may provide adapters.

---

# 15. Retrieval

Retrieval is not the primary innovation of ContextCanon.

V0.1 should therefore keep it simple.

A reasonable first implementation:

```text
lexical / FTS search
+
metadata filtering
```

Embedding-based semantic retrieval may be added behind the Retriever interface if useful, but the project must not become a vector-search project.

---

# 16. CLI

V0.1 exposes three primary commands.

## Scan

```bash
contextcanon scan .
```

Purpose:

```text
discover sources
extract claims
collect evidence
run supported verifiers
persist knowledge state
```

Example output:

```text
Sources:        18
Claims:         27
Verified:       12
Conflicts:       4
Divergences:     2
```

---

## Doctor

```bash
contextcanon doctor [PATH]
```

Purpose:

Run source loading, demo claim extraction, evidence verification and resolution
in memory, then render every supported semantic property for human diagnosis.
`PATH` defaults to the current directory. M4 does not depend on persisted scan
state.

Example:

```text
Property: auth.protocol

Current runtime:
  JWT

Architecture intent:
  OAuth2

Status:
  DIVERGED
```

Potential issue classes:

```text
DIVERGED
AMBIGUOUS
STALE
UNVERIFIED
SUPERSEDED
```

---

## Build

```bash
contextcanon build "How does authentication currently work?"
```

Output:

```text
Context Package: ctx_81fa3c
Repository: a728fd2
Policy: default@0.1

Current State
-------------
Authentication currently uses JWT.

Status:
VERIFIED

Evidence:
- config/auth.yaml
- src/auth/provider.py
- tests/test_auth.py

Architecture Intent
-------------------
OAuth2 is the accepted target architecture.

Status:
DIVERGED

Evidence:
- ADR-015

Explanation:
The accepted architecture differs from the current verified implementation.
```

Optional output format:

```bash
contextcanon build "..." --format json
```

---

# 17. Optional Explain Command

If trivial to support:

```bash
contextcanon explain auth.protocol
```

should show:

```text
selected claims
rejected/secondary claims
evidence
policy
reason codes
verification results
```

This command is useful but not required for the first minimal milestone.

---

# 18. Demo Repository

The repository should include a deliberately inconsistent demo project.

Example:

```text
examples/demo-auth/

README.md
docs/adr/ADR-015-oauth.md
config/auth.yaml
src/auth/provider.py
tests/test_auth.py
```

Content:

```text
README
→ says JWT

ADR
→ accepted target OAuth2

config
→ JWT

implementation
→ JWT

tests
→ JWT behaviour
```

Expected resolution:

```text
Current State:
JWT

Target State:
OAuth2

Knowledge State:
DIVERGED
```

This scenario should be understandable from a GIF or terminal recording in under 15 seconds.

---

# 19. V0.1 Acceptance Criteria

V0.1 is complete when all of the following are true:

### AC1

A fresh user can install ContextCanon locally and run:

```bash
contextcanon scan examples/demo-auth
```

without external infrastructure.

### AC2

The system extracts enough information to identify:

```text
JWT current runtime
OAuth2 architecture intent
```

### AC3

It deterministically identifies:

```text
DIVERGED
```

rather than selecting one state as universally correct.

### AC4

At least one current-state claim is supported by machine-verifiable evidence.

### AC5

`contextcanon doctor` clearly explains the divergence.

### AC6

`contextcanon build ...` produces a structured ContextPackage.

### AC7

The ContextPackage includes:

```text
source revision
policy version
claims
evidence
resolution
reason codes
unresolved conflicts
```

### AC8

The package can be rendered as both:

```text
JSON
Markdown
```

### AC9

Core tests cover:

```text
claim extraction
verification
resolution
divergence detection
context assembly
```

### AC10

No external database or distributed infrastructure is required.

---

# 20. Suggested Package Structure

```text
contextcanon/
├── __init__.py
│
├── core/
│   ├── models.py
│   ├── types.py
│   └── engine.py
│
├── sources/
│   ├── base.py
│   ├── markdown.py
│   ├── config.py
│   └── repository.py
│
├── extraction/
│   ├── base.py
│   └── default.py
│
├── verification/
│   ├── base.py
│   ├── yaml.py
│   ├── json.py
│   └── text.py
│
├── resolution/
│   ├── base.py
│   └── default.py
│
├── retrieval/
│   ├── base.py
│   └── lexical.py
│
├── assembly/
│   ├── base.py
│   └── default.py
│
├── storage/
│   ├── base.py
│   └── sqlite.py
│
├── renderers/
│   ├── json.py
│   └── markdown.py
│
├── integrations/
│   └── mcp/
│
└── cli.py

tests/

examples/
└── demo-auth/

docs/
```

This is a guideline, not a requirement. Avoid unnecessary files or abstraction layers.

---

# 21. Engineering Constraints

The implementation should prioritize:

```text
clarity
small modules
typed interfaces
testability
deterministic behaviour
low dependency count
```

Prefer boring technology where possible.

Avoid abstractions that do not yet have at least two plausible implementations.

Do not create interfaces merely for architectural aesthetics.

Target approximately:

```text
2,000–4,000 lines
```

of meaningful V0.1 implementation code where practical.

This is not a hard limit.

---

# 22. Role of LLMs

The system must be usable in deterministic demo scenarios without requiring an LLM for every operation.

LLMs may optionally support:

```text
natural-language claim extraction
query intent classification
semantic normalization
human-readable explanations
```

However:

> Resolution truth must not depend solely on an opaque LLM judgment.

Whenever an LLM contributes to a decision, the contribution should remain distinguishable from deterministic evidence.

---

# 23. Future Integrations

After the core proves useful, integrations may include:

```text
Pi extension
DeepSeek Harness plugin
MCP server
Claude Code / other coding-agent adapters
```

Integration code must depend on ContextCanon Core.

ContextCanon Core must not depend on those agent frameworks.

---

# 24. Future Features — Not V0.1

Potential later work:

```text
contextcanon diff
context history
knowledge blast radius
Git hooks / CI checks
AST verification
test-based verification
embeddings / hybrid retrieval
custom company policies
Slack / docs connectors
access control
context rewind integration
knowledge-health metrics
evaluation datasets
```

These are explicitly deferred until the V0.1 core workflow is working.

---

# 25. README Product Message

The first screen should communicate the problem rather than architecture.

Suggested copy:

> Your repo says OAuth2.
> Your README says JWT.
> Your ADR says “migrate to OAuth2.”
> Your agent sees all three.
>
> **Which one should it believe?**
>
> ContextCanon resolves conflicting project knowledge using explicit policies, executable evidence, and provenance.

Then immediately show:

```bash
contextcanon scan .
contextcanon doctor
contextcanon build "How does authentication work?"
```

---

# 26. Definition of Success

V0.1 is successful if a developer can understand, within a few minutes, that ContextCanon is **not another RAG library**.

They should understand the core proposition as:

> Retrieval finds relevant knowledge.
> ContextCanon determines how that knowledge should be interpreted, verified and presented to an agent.

The implementation should demonstrate this idea with the smallest reasonable amount of code.
