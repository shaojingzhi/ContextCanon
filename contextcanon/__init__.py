"""Core data models for ContextCanon.

Milestone M0 intentionally provides only the dependency-free model layer and a
CLI placeholder. Extraction, verification, resolution, storage, and rendering
are introduced by later milestones.
"""

from .core.models import Claim, ContextItem, ContextPackage, Evidence, Resolution
from .core.types import (
    ClaimType,
    EvidenceRole,
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
    "ReasonCode",
    "Resolution",
    "ResolutionStatus",
    "SourceType",
]
