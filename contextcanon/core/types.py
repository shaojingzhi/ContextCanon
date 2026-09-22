"""Shared types used by the ContextCanon core models."""

from enum import StrEnum
from typing import TypeAlias


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class ClaimType(StrEnum):
    """The kind of knowledge represented by a claim."""

    RUNTIME_STATE = "RUNTIME_STATE"
    ARCHITECTURE_INTENT = "ARCHITECTURE_INTENT"
    OPERATIONAL_RULE = "OPERATIONAL_RULE"


class EvidenceRole(StrEnum):
    """How an evidence item relates to a claim."""

    OBSERVED = "OBSERVED"
    INTENDED = "INTENDED"
    DOCUMENTED = "DOCUMENTED"


class ResolutionStatus(StrEnum):
    """Possible governance outcomes for competing claims."""

    RESOLVED = "RESOLVED"
    DIVERGED = "DIVERGED"
    AMBIGUOUS = "AMBIGUOUS"
    UNVERIFIED = "UNVERIFIED"
    SUPERSEDED = "SUPERSEDED"


class ReasonCode(StrEnum):
    """Machine-readable explanations for a resolution outcome."""

    VERIFIED_EVIDENCE = "VERIFIED_EVIDENCE"
    HIGHER_ROLE_PRIORITY = "HIGHER_ROLE_PRIORITY"
    INTENT_IMPLEMENTATION_DIVERGENCE = "INTENT_IMPLEMENTATION_DIVERGENCE"
    CONFLICTING_AUTHORITATIVE_EVIDENCE = "CONFLICTING_AUTHORITATIVE_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    SUPERSEDED_BY_NEWER_CLAIM = "SUPERSEDED_BY_NEWER_CLAIM"


class SourceType(StrEnum):
    """Source categories supported by the V0.1 design."""

    MARKDOWN = "MARKDOWN"
    ADR = "ADR"
    YAML = "YAML"
    JSON = "JSON"
    SOURCE_CODE = "SOURCE_CODE"
    TEST = "TEST"
    OTHER = "OTHER"


class TemporalScope(StrEnum):
    """Small temporal vocabulary used to keep current and intended facts apart."""

    CURRENT = "CURRENT"
    FUTURE = "FUTURE"
    HISTORICAL = "HISTORICAL"
    INTERVAL = "INTERVAL"
    UNKNOWN = "UNKNOWN"


class FactAlignment(StrEnum):
    """How confidently an incoming observation refers to an existing fact."""

    SAME_FACT = "SAME_FACT"
    RELATED_BUT_DISTINCT = "RELATED_BUT_DISTINCT"
    UNRELATED = "UNRELATED"
    UNKNOWN = "UNKNOWN"


class EvidenceRelation(StrEnum):
    """Semantic relation proposed for two evidence items."""

    SUPPORTING = "SUPPORTING"
    EQUIVALENT = "EQUIVALENT"
    CONFLICTING = "CONFLICTING"
    SUPERSEDING = "SUPERSEDING"
    COMPATIBLE = "COMPATIBLE"
    UNKNOWN = "UNKNOWN"


class GovernanceState(StrEnum):
    """Deterministic state exposed to query and action consumers."""

    RESOLVED = "RESOLVED"
    DIVERGED = "DIVERGED"
    STALE = "STALE"
    AMBIGUOUS = "AMBIGUOUS"
    UNVERIFIED = "UNVERIFIED"
    SUPERSEDED = "SUPERSEDED"
