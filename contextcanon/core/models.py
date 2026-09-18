"""Small, serializable data models for ContextCanon M0.

These models deliberately contain no extraction, verification, or resolution
behavior. They are plain data passed between those components in later
milestones.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
import json
from typing import Any, ClassVar, Mapping, TypeVar

from .types import (
    ClaimType,
    EvidenceRole,
    ReasonCode,
    ResolutionStatus,
    SourceType,
)

T = TypeVar("T", bound="Serializable")


def _serialize(value: Any) -> Any:
    """Convert model values into JSON-compatible primitives."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _serialize(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_serialize(item) for item in value]
    return value


def _enum_or_value(value: Any, enum_type: type[Any]) -> Any:
    """Restore a known enum value while preserving unknown extension values."""

    if value is None or isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        return value


class Serializable:
    """Mixin providing deterministic dictionary and JSON serialization."""

    _json_sort_keys: ClassVar[bool] = True

    def to_dict(self) -> dict[str, Any]:
        if not is_dataclass(self):
            raise TypeError("Serializable models must be dataclasses")
        return _serialize(self)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=self._json_sort_keys,
        )

    @classmethod
    def from_json(cls: type[T], payload: str) -> T:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object for {cls.__name__}")
        return cls.from_dict(value)


@dataclass(slots=True)
class Evidence(Serializable):
    """A traceable source item supporting or contradicting a claim."""

    id: str
    source_id: str
    source_type: SourceType | str
    location: str
    role: EvidenceRole | str
    content: Any = None
    timestamp: str | datetime | date | None = None
    verifier: str | None = None
    verified: bool | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Evidence":
        return cls(
            id=str(value["id"]),
            source_id=str(value["source_id"]),
            source_type=_enum_or_value(value["source_type"], SourceType),
            location=str(value["location"]),
            role=_enum_or_value(value["role"], EvidenceRole),
            content=value.get("content"),
            timestamp=value.get("timestamp"),
            verifier=value.get("verifier"),
            verified=value.get("verified"),
        )


@dataclass(slots=True)
class Claim(Serializable):
    """One semantic assertion and the evidence associated with it."""

    id: str
    subject: str
    predicate: str
    value: Any
    claim_type: ClaimType | str
    evidence: list[Evidence] = field(default_factory=list)
    status: ResolutionStatus | str | None = None
    confidence: float | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Claim":
        evidence = [Evidence.from_dict(item) for item in value.get("evidence", [])]
        return cls(
            id=str(value["id"]),
            subject=str(value["subject"]),
            predicate=str(value["predicate"]),
            value=value.get("value"),
            claim_type=_enum_or_value(value["claim_type"], ClaimType),
            evidence=evidence,
            status=_enum_or_value(value.get("status"), ResolutionStatus),
            confidence=value.get("confidence"),
        )


@dataclass(slots=True)
class Resolution(Serializable):
    """The governance result for a group of competing claims."""

    status: ResolutionStatus | str
    selected_claims: list[Claim] = field(default_factory=list)
    conflicting_claims: list[Claim] = field(default_factory=list)
    reason_codes: list[ReasonCode | str] = field(default_factory=list)
    explanation: str | None = None
    policy: str | None = None
    policy_version: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Resolution":
        return cls(
            status=_enum_or_value(value["status"], ResolutionStatus),
            selected_claims=[
                Claim.from_dict(item) for item in value.get("selected_claims", [])
            ],
            conflicting_claims=[
                Claim.from_dict(item)
                for item in value.get("conflicting_claims", [])
            ],
            reason_codes=[
                _enum_or_value(item, ReasonCode)
                for item in value.get("reason_codes", [])
            ],
            explanation=value.get("explanation"),
            policy=value.get("policy"),
            policy_version=value.get("policy_version"),
        )


@dataclass(slots=True)
class ContextItem(Serializable):
    """A claim plus the resolution metadata needed by a context consumer."""

    claim: Claim
    status: ResolutionStatus | str | None = None
    supporting_evidence: list[Evidence] = field(default_factory=list)
    reason_codes: list[ReasonCode | str] = field(default_factory=list)
    explanation: str | None = None

    @property
    def resolution_status(self) -> ResolutionStatus | str | None:
        """Alias matching the terminology used in the specification."""

        return self.status

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContextItem":
        evidence = value.get("supporting_evidence", value.get("evidence", []))
        status = value.get("status", value.get("resolution_status"))
        return cls(
            claim=Claim.from_dict(value["claim"]),
            status=_enum_or_value(status, ResolutionStatus),
            supporting_evidence=[Evidence.from_dict(item) for item in evidence],
            reason_codes=[
                _enum_or_value(item, ReasonCode)
                for item in value.get("reason_codes", [])
            ],
            explanation=value.get("explanation"),
        )


@dataclass(slots=True)
class ContextPackage(Serializable):
    """Structured context prepared for an agent consumer."""

    id: str
    task: str
    items: list[ContextItem] = field(default_factory=list)
    unresolved_conflicts: list[Resolution] = field(default_factory=list)
    source_revision: str | None = None
    policy_name: str | None = None
    policy_version: str | None = None
    token_budget: int | None = None
    created_at: str | datetime | date | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContextPackage":
        return cls(
            id=str(value["id"]),
            task=str(value["task"]),
            items=[ContextItem.from_dict(item) for item in value.get("items", [])],
            unresolved_conflicts=[
                Resolution.from_dict(item)
                for item in value.get("unresolved_conflicts", [])
            ],
            source_revision=value.get("source_revision"),
            policy_name=value.get("policy_name"),
            policy_version=value.get("policy_version"),
            token_budget=value.get("token_budget"),
            created_at=value.get("created_at"),
        )
