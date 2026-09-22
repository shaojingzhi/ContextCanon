"""Dependency-free ContextCanon core models and shared types."""

from .models import Claim, ContextItem, ContextPackage, Evidence, Resolution
from .types import (
    ClaimType,
    EvidenceRelation,
    EvidenceRole,
    FactAlignment,
    GovernanceState,
    JSONValue,
    ReasonCode,
    ResolutionStatus,
    SourceType,
    TemporalScope,
)

__all__ = [
    "Claim",
    "ClaimType",
    "ContextItem",
    "ContextPackage",
    "Evidence",
    "EvidenceRelation",
    "EvidenceRole",
    "FactAlignment",
    "GovernanceState",
    "JSONValue",
    "ReasonCode",
    "Resolution",
    "ResolutionStatus",
    "SourceType",
    "TemporalScope",
]
