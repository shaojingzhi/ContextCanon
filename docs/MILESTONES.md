# ContextCanon Milestones

## M0 — Repository scaffold + core models

Goal:
建立最小可运行项目骨架和核心数据模型。

Scope:
- Python package scaffold
- Claim
- Evidence
- Resolution
- ContextPackage
- enums / shared types
- basic unit tests
- pyproject.toml
- CLI placeholder

Do NOT implement:
- extraction
- verification
- resolution logic
- storage
- retrieval
- MCP

Acceptance criteria:
- package imports successfully
- tests pass
- models serialize cleanly
- no unnecessary abstractions

---

## M1 — Demo source ingestion + extraction

Goal:
从 demo-auth 中读取 Markdown / ADR / YAML / JSON，
生成 Claim + Evidence。

Scope:
- SourceLoader abstraction
- Markdown loader
- YAML/JSON loader
- deterministic extraction for demo
- demo-auth fixture
- unit tests

Do NOT implement:
- LLM extractor
- vector DB
- resolution
- MCP

Acceptance criteria:
- scan demo-auth
- identify JWT runtime claim
- identify OAuth2 architecture-intent claim
- preserve provenance

---

## M2 — Verification

Goal:
给结构化 evidence 增加 deterministic verification。

Scope:
- Verifier protocol
- YamlPathVerifier
- JsonPathVerifier
- TextPresenceVerifier
- verification result

Acceptance criteria:
- auth provider in config can be machine verified
- failed verification is explicit
- verifier errors do not crash scan

---

## M3 — Resolution

Goal:
实现 DefaultResolutionPolicy。

Scope:
- runtime-state policy
- architecture-intent policy
- RESOLVED / DIVERGED / AMBIGUOUS / UNVERIFIED
- reason codes

Acceptance criteria:
demo-auth resolves to:
- runtime = JWT
- intent = OAuth2
- status = DIVERGED

---

## M4 — Doctor CLI

...

## M5 — Context assembly + build

...

## M6 — persistence / retrieval cleanup

...

## M7 — MCP / Pi integration

...

## M8 — README / demo / release polish