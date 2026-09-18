"""Shared enums used by the M0 data model.

The enums inherit from ``str`` so their values remain pleasant to use in JSON
and in command-line output while retaining an explicit vocabulary in Python.
"""

from enum import Enum


class _StringEnum(str, Enum):
    """A string-valued enum with readable string conversion."""

    def __str__(self) -> str:
        return self.value


class ClaimType(_StringEnum):
    """The kind of knowledge represented by a claim."""

    RUNTIME_STATE = "RUNTIME_STATE"
    ARCHITECTURE_INTENT = "ARCHITECTURE_INTENT"
    OPERATIONAL_RULE = "OPERATIONAL_RULE"


class EvidenceRole(_StringEnum):
    """How an evidence item relates to the claim it supports."""

    OBSERVED = "OBSERVED"
    INTENDED = "INTENDED"
    DOCUMENTED = "DOCUMENTED"
    VERIFIED = "VERIFIED"


class ResolutionStatus(_StringEnum):
    """Possible governance outcomes for competing claims."""

    RESOLVED = "RESOLVED"
    DIVERGED = "DIVERGED"
    AMBIGUOUS = "AMBIGUOUS"
    UNVERIFIED = "UNVERIFIED"
    SUPERSEDED = "SUPERSEDED"


class ReasonCode(_StringEnum):
    """Machine-readable explanations for a resolution outcome."""

    VERIFIED_EVIDENCE = "VERIFIED_EVIDENCE"
    HIGHER_ROLE_PRIORITY = "HIGHER_ROLE_PRIORITY"
    INTENT_IMPLEMENTATION_DIVERGENCE = "INTENT_IMPLEMENTATION_DIVERGENCE"
    CONFLICTING_AUTHORITATIVE_EVIDENCE = "CONFLICTING_AUTHORITATIVE_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    SUPERSEDED_BY_NEWER_CLAIM = "SUPERSEDED_BY_NEWER_CLAIM"


class SourceType(_StringEnum):
    """Common source categories supported by the V0.1 design."""

    MARKDOWN = "MARKDOWN"
    ADR = "ADR"
    YAML = "YAML"
    JSON = "JSON"
    SOURCE_CODE = "SOURCE_CODE"
    TEST = "TEST"
    OTHER = "OTHER"
