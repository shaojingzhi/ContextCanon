"""Deterministic resolution policy for ContextCanon M3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from .core import (
    Claim,
    ClaimType,
    EvidenceRole,
    JSONValue,
    ReasonCode,
    Resolution,
    ResolutionStatus,
)


class ResolutionPolicy(Protocol):
    """Minimal interface for resolving claims about one semantic property."""

    name: str
    version: str

    def resolve(self, claims: list[Claim]) -> Resolution:
        """Resolve claims without inspecting sources or rerunning verification."""


@dataclass(frozen=True, slots=True)
class _Selection:
    claim: Claim | None = None
    conflicting_claims: tuple[Claim, ...] = ()
    verified: bool = False
    reason_codes: tuple[ReasonCode, ...] = ()
    explanation: str | None = None


_RUNTIME_ROLE_PRIORITY = {
    EvidenceRole.OBSERVED: 0,
    EvidenceRole.DOCUMENTED: 1,
    EvidenceRole.INTENDED: 2,
}
_INTENT_ROLE_PRIORITY = {
    EvidenceRole.INTENDED: 0,
    EvidenceRole.OBSERVED: 1,
    EvidenceRole.DOCUMENTED: 2,
}
_REASON_ORDER = {
    ReasonCode.VERIFIED_EVIDENCE: 0,
    ReasonCode.HIGHER_ROLE_PRIORITY: 1,
    ReasonCode.INTENT_IMPLEMENTATION_DIVERGENCE: 2,
    ReasonCode.CONFLICTING_AUTHORITATIVE_EVIDENCE: 3,
    ReasonCode.INSUFFICIENT_EVIDENCE: 4,
}


def _value_key(value: JSONValue) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _claim_key(claim: Claim) -> tuple[str, str, str, str, str]:
    return (
        claim.subject,
        claim.predicate,
        claim.claim_type.value,
        _value_key(claim.value),
        claim.id,
    )


def _ordered_claims(claims: list[Claim] | tuple[Claim, ...]) -> list[Claim]:
    return sorted(claims, key=_claim_key)


def _ordered_reasons(reason_codes: list[ReasonCode]) -> list[ReasonCode]:
    return sorted(set(reason_codes), key=_REASON_ORDER.__getitem__)


def _display_value(value: JSONValue) -> str:
    return value if isinstance(value, str) else _value_key(value)


def _has_verified_evidence(claim: Claim) -> bool:
    return any(evidence.verified is True for evidence in claim.evidence)


def _best_role_rank(
    claim: Claim,
    priority: dict[EvidenceRole, int],
) -> int | None:
    ranks = [
        priority[evidence.role]
        for evidence in claim.evidence
        if evidence.verified is not False
    ]
    return min(ranks) if ranks else None


def _has_verified_evidence_at_rank(
    claims: list[Claim],
    rank: int,
    priority: dict[EvidenceRole, int],
) -> bool:
    return any(
        evidence.verified is True and priority[evidence.role] == rank
        for claim in claims
        for evidence in claim.evidence
    )


def _distinct_values(claims: list[Claim]) -> set[str]:
    return {_value_key(claim.value) for claim in claims}


def _select_runtime(claims: list[Claim]) -> _Selection:
    runtime_claims = _ordered_claims(
        [claim for claim in claims if claim.claim_type is ClaimType.RUNTIME_STATE]
    )
    verified_claims = [
        claim for claim in runtime_claims if _has_verified_evidence(claim)
    ]

    if len(_distinct_values(verified_claims)) > 1:
        return _Selection(
            conflicting_claims=tuple(verified_claims),
            reason_codes=(ReasonCode.CONFLICTING_AUTHORITATIVE_EVIDENCE,),
            explanation=(
                "Multiple verified runtime claims disagree and no deterministic "
                "winner exists."
            ),
        )
    if verified_claims:
        return _Selection(
            claim=verified_claims[0],
            verified=True,
            reason_codes=(ReasonCode.VERIFIED_EVIDENCE,),
        )
    if not runtime_claims:
        return _Selection()

    ranked_claims = [
        (rank, claim)
        for claim in runtime_claims
        if (rank := _best_role_rank(claim, _RUNTIME_ROLE_PRIORITY)) is not None
    ]
    if not ranked_claims:
        return _Selection(
            reason_codes=(ReasonCode.INSUFFICIENT_EVIDENCE,),
            explanation="No runtime claim has reproducible supporting evidence.",
        )

    best_rank = min(rank for rank, _ in ranked_claims)
    best_claims = [claim for rank, claim in ranked_claims if rank == best_rank]
    used_role_priority = any(rank > best_rank for rank, _ in ranked_claims)

    if len(_distinct_values(best_claims)) > 1:
        reasons = [
            ReasonCode.CONFLICTING_AUTHORITATIVE_EVIDENCE,
            ReasonCode.INSUFFICIENT_EVIDENCE,
        ]
        if used_role_priority:
            reasons.append(ReasonCode.HIGHER_ROLE_PRIORITY)
        return _Selection(
            conflicting_claims=tuple(best_claims),
            reason_codes=tuple(_ordered_reasons(reasons)),
            explanation=(
                "Multiple equally preferred unverified runtime claims disagree "
                "and no deterministic winner exists."
            ),
        )

    reasons = [ReasonCode.INSUFFICIENT_EVIDENCE]
    if used_role_priority:
        reasons.append(ReasonCode.HIGHER_ROLE_PRIORITY)
    return _Selection(
        claim=best_claims[0],
        reason_codes=tuple(_ordered_reasons(reasons)),
    )


def _select_architecture_intent(claims: list[Claim]) -> _Selection:
    intent_claims = _ordered_claims(
        [
            claim
            for claim in claims
            if claim.claim_type is ClaimType.ARCHITECTURE_INTENT
        ]
    )
    if not intent_claims:
        return _Selection()

    ranked_claims = [
        (rank, claim)
        for claim in intent_claims
        if (rank := _best_role_rank(claim, _INTENT_ROLE_PRIORITY)) is not None
    ]
    if not ranked_claims:
        return _Selection(
            reason_codes=(ReasonCode.INSUFFICIENT_EVIDENCE,),
            explanation=(
                "No architecture-intent claim has reproducible supporting evidence."
            ),
        )

    best_rank = min(rank for rank, _ in ranked_claims)
    best_claims = [claim for rank, claim in ranked_claims if rank == best_rank]
    used_role_priority = any(rank > best_rank for rank, _ in ranked_claims)

    if len(_distinct_values(best_claims)) > 1:
        reasons = [ReasonCode.CONFLICTING_AUTHORITATIVE_EVIDENCE]
        if used_role_priority:
            reasons.append(ReasonCode.HIGHER_ROLE_PRIORITY)
        return _Selection(
            conflicting_claims=tuple(best_claims),
            reason_codes=tuple(_ordered_reasons(reasons)),
            explanation=(
                "Multiple equally authoritative architecture-intent claims "
                "disagree and no deterministic winner exists."
            ),
        )

    selected = best_claims[0]
    verified = _has_verified_evidence_at_rank(
        best_claims,
        best_rank,
        _INTENT_ROLE_PRIORITY,
    )
    reasons = [
        ReasonCode.VERIFIED_EVIDENCE
        if verified
        else ReasonCode.INSUFFICIENT_EVIDENCE
    ]
    if used_role_priority:
        reasons.append(ReasonCode.HIGHER_ROLE_PRIORITY)
    return _Selection(
        claim=selected,
        verified=verified,
        reason_codes=tuple(_ordered_reasons(reasons)),
    )


def _resolved_explanation(
    runtime: _Selection,
    intent: _Selection,
    status: ResolutionStatus,
) -> str:
    parts: list[str] = []
    if runtime.claim is not None:
        value = _display_value(runtime.claim.value)
        if runtime.verified:
            parts.append(
                f"Current runtime state {value} is supported by verified evidence."
            )
        else:
            parts.append(
                f"Possible runtime state {value} lacks successfully verified evidence."
            )
    if intent.claim is not None:
        value = _display_value(intent.claim.value)
        if intent.verified:
            parts.append(f"Accepted architecture intent is {value}.")
        else:
            parts.append(
                f"Possible architecture intent {value} lacks successfully verified "
                "evidence."
            )
    if status is ResolutionStatus.DIVERGED:
        parts.append("Runtime state differs from architecture intent.")
    elif (
        status is ResolutionStatus.RESOLVED
        and runtime.claim is not None
        and intent.claim is not None
    ):
        parts.append("Runtime state matches architecture intent.")
    if not parts:
        parts.append("No relevant claim has reproducible supporting evidence.")
    return " ".join(parts)


class DefaultResolutionPolicy:
    """The explicit deterministic resolution rules implemented for M3."""

    name = "default"
    version = "0.1"

    def resolve(self, claims: list[Claim]) -> Resolution:
        ordered_claims = _ordered_claims(claims)
        properties = {
            (claim.subject, claim.predicate) for claim in ordered_claims
        }
        if len(properties) > 1:
            raise ValueError(
                "DefaultResolutionPolicy resolves one subject/predicate property "
                "at a time"
            )

        runtime = _select_runtime(ordered_claims)
        intent = _select_architecture_intent(ordered_claims)
        selected = [
            selection.claim
            for selection in (runtime, intent)
            if selection.claim is not None
        ]
        conflicts = [
            *runtime.conflicting_claims,
            *intent.conflicting_claims,
        ]
        reasons = [*runtime.reason_codes, *intent.reason_codes]

        if conflicts:
            explanations = [
                explanation
                for explanation in (runtime.explanation, intent.explanation)
                if explanation is not None
            ]
            return Resolution(
                status=ResolutionStatus.AMBIGUOUS,
                selected_claims=_ordered_claims(selected),
                conflicting_claims=_ordered_claims(conflicts),
                reason_codes=_ordered_reasons(reasons),
                explanation=" ".join(explanations),
                policy=self.name,
                policy_version=self.version,
            )

        if not selected:
            status = ResolutionStatus.UNVERIFIED
            reasons.append(ReasonCode.INSUFFICIENT_EVIDENCE)
        elif not all(
            selection.verified
            for selection in (runtime, intent)
            if selection.claim is not None
        ):
            status = ResolutionStatus.UNVERIFIED
            reasons.append(ReasonCode.INSUFFICIENT_EVIDENCE)
        elif runtime.claim is not None and intent.claim is not None:
            if _value_key(runtime.claim.value) == _value_key(intent.claim.value):
                status = ResolutionStatus.RESOLVED
            else:
                status = ResolutionStatus.DIVERGED
                reasons.append(ReasonCode.INTENT_IMPLEMENTATION_DIVERGENCE)
        else:
            status = ResolutionStatus.RESOLVED

        return Resolution(
            status=status,
            selected_claims=_ordered_claims(selected),
            conflicting_claims=[],
            reason_codes=_ordered_reasons(reasons),
            explanation=_resolved_explanation(runtime, intent, status),
            policy=self.name,
            policy_version=self.version,
        )
