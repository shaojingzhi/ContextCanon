"""Deterministic ContextPackage assembly for ContextCanon M5."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .core import (
    Claim,
    ContextItem,
    ContextPackage,
    Evidence,
    JSONValue,
    Resolution,
    ResolutionStatus,
)


def _canonical_json(value: JSONValue) -> str:
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
        _canonical_json(claim.value),
        claim.id,
    )


def _evidence_key(evidence: Evidence) -> tuple[str, str, str]:
    return (evidence.source_id, evidence.location, evidence.id)


def _evidence_identity(evidence: Evidence) -> dict[str, JSONValue]:
    return {
        "id": evidence.id,
        "source_id": evidence.source_id,
        "location": evidence.location,
        "role": evidence.role.value,
        "content": evidence.content,
        "verified": evidence.verified,
        "verifier": evidence.verifier,
    }


def _resolution_identity(resolution: Resolution) -> dict[str, JSONValue]:
    return {
        "status": resolution.status.value,
        "selected_claim_ids": sorted(
            claim.id for claim in resolution.selected_claims
        ),
        "conflicting_claim_ids": sorted(
            claim.id for claim in resolution.conflicting_claims
        ),
        "reason_codes": sorted(reason.value for reason in resolution.reason_codes),
    }


def _resolution_key(resolution: Resolution) -> str:
    return _canonical_json(_resolution_identity(resolution))


def source_revision(root: Path) -> str:
    """Return the enclosing Git repository's HEAD, or ``unknown``."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"

    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else "unknown"


def utc_timestamp() -> str:
    """Return a seconds-precision ISO-8601 UTC timestamp."""

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _package_id(
    *,
    task: str,
    source_revision: str,
    policy_name: str,
    policy_version: str,
    items: list[ContextItem],
    unresolved_conflicts: list[Resolution],
    token_budget: int | None,
) -> str:
    identity: dict[str, JSONValue] = {
        "task": task,
        "source_revision": source_revision,
        "policy_name": policy_name,
        "policy_version": policy_version,
        "items": [
            {
                "claim_id": item.claim.id,
                "resolution_status": item.resolution_status.value,
                "reason_codes": [reason.value for reason in item.reason_codes],
                "supporting_evidence": [
                    _evidence_identity(evidence)
                    for evidence in item.supporting_evidence
                ],
            }
            for item in items
        ],
        "unresolved_conflicts": [
            _resolution_identity(resolution)
            for resolution in unresolved_conflicts
        ],
        "token_budget": token_budget,
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    return f"ctx-{digest[:12]}"


def assemble_context(
    *,
    task: str,
    resolutions: list[Resolution],
    source_revision: str,
    created_at: str,
    policy_name: str,
    policy_version: str,
    token_budget: int | None = None,
) -> ContextPackage:
    """Compile resolution output into a deterministic structured package."""

    items = [
        ContextItem(
            claim=claim,
            resolution_status=resolution.status,
            supporting_evidence=sorted(
                (
                    evidence
                    for evidence in claim.evidence
                    if evidence.verified is not False
                ),
                key=_evidence_key,
            ),
            reason_codes=list(resolution.reason_codes),
            explanation=resolution.explanation,
        )
        for resolution in resolutions
        for claim in resolution.selected_claims
    ]
    items.sort(key=lambda item: _claim_key(item.claim))

    unresolved_conflicts = sorted(
        (
            resolution
            for resolution in resolutions
            if resolution.status is not ResolutionStatus.RESOLVED
        ),
        key=_resolution_key,
    )
    package_id = _package_id(
        task=task,
        source_revision=source_revision,
        policy_name=policy_name,
        policy_version=policy_version,
        items=items,
        unresolved_conflicts=unresolved_conflicts,
        token_budget=token_budget,
    )
    return ContextPackage(
        id=package_id,
        task=task,
        source_revision=source_revision,
        policy_name=policy_name,
        policy_version=policy_version,
        created_at=created_at,
        items=items,
        unresolved_conflicts=unresolved_conflicts,
        token_budget=token_budget,
    )
