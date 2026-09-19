"""Deterministic claim extraction for the M1 demo repository."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

from .core import Claim, ClaimType, Evidence, EvidenceRole, JSONValue, SourceType
from .sources import SourceDocument


_CURRENT_PROTOCOL = re.compile(
    r"^\s*Current protocol\s*:\s*(?P<value>[^\s#]+)\s*$",
    re.IGNORECASE,
)
_TARGET_PROTOCOL = re.compile(
    r"^\s*Target protocol\s*:\s*(?P<value>[^\s#]+)\s*$",
    re.IGNORECASE,
)
_ACCEPTED_STATUS = re.compile(
    r"^\s*Status\s*:\s*Accepted\s*$",
    re.IGNORECASE,
)


def _normalize_protocol(value: str) -> str:
    stripped = value.strip()
    known_values = {"jwt": "JWT", "oauth2": "OAuth2"}
    return known_values.get(stripped.casefold(), stripped)


def _stable_id(prefix: str, parts: JSONValue) -> str:
    encoded = json.dumps(
        parts,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _claim_id(
    claim_type: ClaimType,
    subject: str,
    predicate: str,
    value: JSONValue,
) -> str:
    return _stable_id(
        "claim",
        [claim_type.value, subject, predicate, value],
    )


def _evidence(
    document: SourceDocument,
    *,
    location: str,
    role: EvidenceRole,
    content: JSONValue,
) -> Evidence:
    return Evidence(
        id=_stable_id("evidence", [document.path, location, role.value]),
        source_id=document.path,
        source_type=document.source_type,
        location=location,
        role=role,
        content=content,
        verified=None,
    )


def _text_assertions(
    document: SourceDocument,
) -> list[tuple[ClaimType, str, Evidence]]:
    if not isinstance(document.content, str):
        return []

    lines = document.content.splitlines()
    accepted_adr = (
        document.source_type is SourceType.ADR
        and any(_ACCEPTED_STATUS.match(line) for line in lines)
    )
    assertions: list[tuple[ClaimType, str, Evidence]] = []
    for line_number, line in enumerate(lines, start=1):
        current_match = _CURRENT_PROTOCOL.match(line)
        if document.source_type is SourceType.MARKDOWN and current_match:
            raw_value = current_match.group("value")
            assertions.append(
                (
                    ClaimType.RUNTIME_STATE,
                    _normalize_protocol(raw_value),
                    _evidence(
                        document,
                        location=f"line:{line_number}",
                        role=EvidenceRole.DOCUMENTED,
                        content=line,
                    ),
                )
            )

        target_match = _TARGET_PROTOCOL.match(line)
        if accepted_adr and target_match:
            raw_value = target_match.group("value")
            assertions.append(
                (
                    ClaimType.ARCHITECTURE_INTENT,
                    _normalize_protocol(raw_value),
                    _evidence(
                        document,
                        location=f"line:{line_number}",
                        role=EvidenceRole.INTENDED,
                        content=line,
                    ),
                )
            )
    return assertions


def _config_assertions(
    document: SourceDocument,
) -> list[tuple[ClaimType, str, Evidence]]:
    if document.source_type not in {SourceType.YAML, SourceType.JSON}:
        return []
    if not isinstance(document.content, dict):
        return []
    auth = document.content.get("auth")
    if not isinstance(auth, dict):
        return []
    provider = auth.get("provider")
    if not isinstance(provider, str):
        return []
    return [
        (
            ClaimType.RUNTIME_STATE,
            _normalize_protocol(provider),
            _evidence(
                document,
                location="auth.provider",
                role=EvidenceRole.OBSERVED,
                content=provider,
            ),
        )
    ]


def extract_demo_claims(documents: Iterable[SourceDocument]) -> list[Claim]:
    """Extract and aggregate the explicit authentication facts used in M1."""

    claims: dict[str, Claim] = {}
    for document in sorted(documents, key=lambda item: item.path):
        assertions = _text_assertions(document) + _config_assertions(document)
        for claim_type, value, evidence in assertions:
            claim_id = _claim_id(claim_type, "auth", "protocol", value)
            claim = claims.get(claim_id)
            if claim is None:
                claim = Claim(
                    id=claim_id,
                    subject="auth",
                    predicate="protocol",
                    value=value,
                    claim_type=claim_type,
                    confidence=None,
                )
                claims[claim_id] = claim
            if all(item.id != evidence.id for item in claim.evidence):
                claim.evidence.append(evidence)

    return sorted(
        claims.values(),
        key=lambda claim: (
            claim.claim_type.value,
            claim.subject,
            claim.predicate,
            json.dumps(claim.value, sort_keys=True),
        ),
    )
