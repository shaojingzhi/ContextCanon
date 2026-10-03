"""Small offline ContextCanon V2 governance core."""

from .governance import govern
from .models import (
    Cardinality,
    Claim,
    Evidence,
    FactNeed,
    GovernanceResult,
    Modality,
    Relation,
    RelationRecord,
    SummaryState,
)

__all__ = [
    "Cardinality",
    "Claim",
    "Evidence",
    "FactNeed",
    "GovernanceResult",
    "Modality",
    "Relation",
    "RelationRecord",
    "SummaryState",
    "govern",
]
