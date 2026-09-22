"""Dependency-free, serializable data models for ContextCanon M0.

The models contain data and invariants only. Extraction, verification,
resolution behavior, persistence, and rendering belong to later milestones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any

from .types import (
    ClaimType,
    EvidenceRole,
    JSONValue,
    ReasonCode,
    ResolutionStatus,
    SourceType,
    TemporalScope,
)


def _canonical_json_value(value: object, *, field_name: str) -> JSONValue:
    """Return a deterministic JSON value or reject unsupported input."""

    if value is None:
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{field_name} must not contain NaN or infinity")
        return value
    if isinstance(value, list):
        return [
            _canonical_json_value(item, field_name=f"{field_name}[]")
            for item in value
        ]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{field_name} must use string object keys")
        return {
            key: _canonical_json_value(value[key], field_name=f"{field_name}.{key}")
            for key in sorted(value)
        }
    raise TypeError(f"{field_name} contains unsupported value {type(value).__name__}")


def _require_enum(value: object, enum_type: type[StrEnum], *, field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{field_name} must be a {enum_type.__name__}")


def _require_items(values: list[Any], item_type: type[Any], *, field_name: str) -> None:
    if not all(isinstance(value, item_type) for value in values):
        raise TypeError(f"{field_name} must contain only {item_type.__name__} values")


@dataclass(slots=True)
class Evidence:
    """A traceable source item supporting or contradicting a claim."""

    id: str
    source_id: str
    source_type: SourceType
    location: str
    role: EvidenceRole
    content: JSONValue = None
    timestamp: str | None = None
    verifier: str | None = None
    verified: bool | None = None
    temporal_scope: TemporalScope | None = None
    confidence: float | None = None
    observed_at: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    observation_id: str | None = None
    provenance: JSONValue = None

    def __post_init__(self) -> None:
        _require_enum(self.source_type, SourceType, field_name="source_type")
        _require_enum(self.role, EvidenceRole, field_name="role")
        if self.temporal_scope is not None:
            _require_enum(
                self.temporal_scope,
                TemporalScope,
                field_name="temporal_scope",
            )
        self.content = _canonical_json_value(self.content, field_name="content")
        if self.timestamp is not None and not isinstance(self.timestamp, str):
            raise TypeError("timestamp must be an ISO-8601 string or None")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        for name in ("observed_at", "valid_from", "valid_until"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be an ISO-8601 string or None")
        if self.observation_id is not None and not isinstance(
            self.observation_id,
            str,
        ):
            raise TypeError("observation_id must be a string or None")
        self.provenance = _canonical_json_value(
            self.provenance,
            field_name="provenance",
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "location": self.location,
            "role": self.role.value,
            "content": _canonical_json_value(self.content, field_name="content"),
            "timestamp": self.timestamp,
            "verifier": self.verifier,
            "verified": self.verified,
            "temporal_scope": (
                self.temporal_scope.value if self.temporal_scope else None
            ),
            "confidence": self.confidence,
            "observed_at": self.observed_at,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "observation_id": self.observation_id,
            "provenance": _canonical_json_value(
                self.provenance,
                field_name="provenance",
            ),
        }


@dataclass(slots=True)
class Claim:
    """One semantic assertion and the evidence associated with it."""

    id: str
    subject: str
    predicate: str
    value: JSONValue
    claim_type: ClaimType
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float | None = None

    def __post_init__(self) -> None:
        _require_enum(self.claim_type, ClaimType, field_name="claim_type")
        _require_items(self.evidence, Evidence, field_name="evidence")
        self.value = _canonical_json_value(self.value, field_name="value")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": _canonical_json_value(self.value, field_name="value"),
            "claim_type": self.claim_type.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class Resolution:
    """The governance result for a group of competing claims."""

    status: ResolutionStatus
    selected_claims: list[Claim] = field(default_factory=list)
    conflicting_claims: list[Claim] = field(default_factory=list)
    reason_codes: list[ReasonCode] = field(default_factory=list)
    explanation: str | None = None
    policy: str | None = None
    policy_version: str | None = None

    def __post_init__(self) -> None:
        _require_enum(self.status, ResolutionStatus, field_name="status")
        _require_items(self.selected_claims, Claim, field_name="selected_claims")
        _require_items(
            self.conflicting_claims,
            Claim,
            field_name="conflicting_claims",
        )
        _require_items(self.reason_codes, ReasonCode, field_name="reason_codes")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "status": self.status.value,
            "selected_claims": [item.to_dict() for item in self.selected_claims],
            "conflicting_claims": [item.to_dict() for item in self.conflicting_claims],
            "reason_codes": [item.value for item in self.reason_codes],
            "explanation": self.explanation,
            "policy": self.policy,
            "policy_version": self.policy_version,
        }


@dataclass(slots=True)
class ContextItem:
    """A claim plus its compiled resolution metadata."""

    claim: Claim
    resolution_status: ResolutionStatus
    supporting_evidence: list[Evidence] = field(default_factory=list)
    reason_codes: list[ReasonCode] = field(default_factory=list)
    explanation: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.claim, Claim):
            raise TypeError("claim must be a Claim")
        _require_enum(
            self.resolution_status,
            ResolutionStatus,
            field_name="resolution_status",
        )
        _require_items(
            self.supporting_evidence,
            Evidence,
            field_name="supporting_evidence",
        )
        _require_items(self.reason_codes, ReasonCode, field_name="reason_codes")
        if any(item not in self.claim.evidence for item in self.supporting_evidence):
            raise ValueError("supporting_evidence must belong to the claim")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "claim": self.claim.to_dict(),
            "resolution_status": self.resolution_status.value,
            "supporting_evidence": [
                item.to_dict() for item in self.supporting_evidence
            ],
            "reason_codes": [item.value for item in self.reason_codes],
            "explanation": self.explanation,
        }


@dataclass(slots=True)
class ContextPackage:
    """Structured, reproducible context prepared for an agent consumer."""

    id: str
    task: str
    source_revision: str
    policy_name: str
    policy_version: str
    created_at: str
    items: list[ContextItem] = field(default_factory=list)
    unresolved_conflicts: list[Resolution] = field(default_factory=list)
    token_budget: int | None = None

    def __post_init__(self) -> None:
        _require_items(self.items, ContextItem, field_name="items")
        _require_items(
            self.unresolved_conflicts,
            Resolution,
            field_name="unresolved_conflicts",
        )
        if self.token_budget is not None and self.token_budget < 0:
            raise ValueError("token_budget must be non-negative")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "task": self.task,
            "items": [item.to_dict() for item in self.items],
            "unresolved_conflicts": [
                item.to_dict() for item in self.unresolved_conflicts
            ],
            "source_revision": self.source_revision,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "token_budget": self.token_budget,
            "created_at": self.created_at,
        }
