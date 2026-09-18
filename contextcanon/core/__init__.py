"""Dependency-free ContextCanon core models and shared types."""

from .models import Claim, ContextItem, ContextPackage, Evidence, Resolution
from .types import (
    ClaimType,
    EvidenceRole,
    JSONValue,
    ReasonCode,
    ResolutionStatus,
    SourceType,
)

__all__ = [
    "Claim",
    "ClaimType",
    "ContextItem",
    "ContextPackage",
    "Evidence",
    "EvidenceRole",
    "JSONValue",
    "ReasonCode",
    "Resolution",
    "ResolutionStatus",
    "SourceType",
]
