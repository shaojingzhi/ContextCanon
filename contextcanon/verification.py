"""Deterministic verification of extracted evidence against loaded sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .core import Claim, Evidence, JSONValue, SourceType
from .sources import SourceDocument


class VerificationStatus(StrEnum):
    """The outcome of one deterministic evidence reproduction attempt.

    VERIFIED means the source matches the evidence, FAILED means verification
    completed but it did not match, and ERROR means verification could not be
    completed.
    """

    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """A machine-readable result for one evidence item."""

    evidence_id: str
    verifier: str | None
    status: VerificationStatus
    message: str

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "evidence_id": self.evidence_id,
            "verifier": self.verifier,
            "status": self.status.value,
            "message": self.message,
        }


class Verifier(Protocol):
    """Minimal interface for checking evidence against one loaded source."""

    name: str

    def verify(
        self,
        evidence: Evidence,
        document: SourceDocument,
    ) -> VerificationResult:
        """Check whether the document still contains the recorded evidence."""


def _result(
    evidence: Evidence,
    verifier: str | None,
    status: VerificationStatus,
    message: str,
) -> VerificationResult:
    return VerificationResult(
        evidence_id=evidence.id,
        verifier=verifier,
        status=status,
        message=message,
    )


def _verify_structured_path(
    evidence: Evidence,
    document: SourceDocument,
    *,
    verifier_name: str,
    source_type: SourceType,
) -> VerificationResult:
    if (
        evidence.source_type is not source_type
        or document.source_type is not source_type
    ):
        return _result(
            evidence,
            verifier_name,
            VerificationStatus.ERROR,
            f"{verifier_name} requires {source_type.value} evidence and source",
        )
    if document.path != evidence.source_id:
        return _result(
            evidence,
            verifier_name,
            VerificationStatus.ERROR,
            "source document path does not match evidence source_id",
        )

    current: JSONValue = document.content
    for segment in evidence.location.split("."):
        if not segment:
            return _result(
                evidence,
                verifier_name,
                VerificationStatus.ERROR,
                f"invalid structured location: {evidence.location!r}",
            )
        if not isinstance(current, dict) or segment not in current:
            return _result(
                evidence,
                verifier_name,
                VerificationStatus.FAILED,
                f"source does not contain path {evidence.location!r}",
            )
        current = current[segment]

    if current != evidence.content:
        return _result(
            evidence,
            verifier_name,
            VerificationStatus.FAILED,
            f"expected {evidence.content!r} at {evidence.location!r}, found {current!r}",
        )
    return _result(
        evidence,
        verifier_name,
        VerificationStatus.VERIFIED,
        f"matched source value at {evidence.location!r}",
    )


class YamlPathVerifier:
    """Verify an exact value at a dotted path in a YAML document."""

    name = "yaml-path"

    def verify(
        self,
        evidence: Evidence,
        document: SourceDocument,
    ) -> VerificationResult:
        return _verify_structured_path(
            evidence,
            document,
            verifier_name=self.name,
            source_type=SourceType.YAML,
        )


class JsonPathVerifier:
    """Verify an exact value at a dotted path in a JSON document."""

    name = "json-path"

    def verify(
        self,
        evidence: Evidence,
        document: SourceDocument,
    ) -> VerificationResult:
        return _verify_structured_path(
            evidence,
            document,
            verifier_name=self.name,
            source_type=SourceType.JSON,
        )


class TextPresenceVerifier:
    """Verify an exact Markdown or ADR excerpt at its recorded line."""

    name = "text-presence"

    def verify(
        self,
        evidence: Evidence,
        document: SourceDocument,
    ) -> VerificationResult:
        text_source_types = {SourceType.MARKDOWN, SourceType.ADR}
        if (
            evidence.source_type not in text_source_types
            or document.source_type is not evidence.source_type
        ):
            return _result(
                evidence,
                self.name,
                VerificationStatus.ERROR,
                "text-presence requires Markdown or ADR evidence and source",
            )
        if document.path != evidence.source_id:
            return _result(
                evidence,
                self.name,
                VerificationStatus.ERROR,
                "source document path does not match evidence source_id",
            )
        if not isinstance(document.content, str) or not isinstance(
            evidence.content,
            str,
        ):
            return _result(
                evidence,
                self.name,
                VerificationStatus.ERROR,
                "text-presence requires string source and evidence content",
            )

        prefix, separator, raw_line_number = evidence.location.partition(":")
        if prefix != "line" or not separator or not raw_line_number.isdigit():
            return _result(
                evidence,
                self.name,
                VerificationStatus.ERROR,
                f"invalid text location: {evidence.location!r}",
            )

        line_number = int(raw_line_number)
        lines = document.content.splitlines()
        if line_number < 1 or line_number > len(lines):
            return _result(
                evidence,
                self.name,
                VerificationStatus.FAILED,
                f"source does not contain line {line_number}",
            )

        actual = lines[line_number - 1]
        if actual != evidence.content:
            return _result(
                evidence,
                self.name,
                VerificationStatus.FAILED,
                f"expected {evidence.content!r} at line {line_number}, found {actual!r}",
            )
        return _result(
            evidence,
            self.name,
            VerificationStatus.VERIFIED,
            f"matched source excerpt at line {line_number}",
        )


def _default_verifiers() -> dict[SourceType, Verifier]:
    text_verifier = TextPresenceVerifier()
    return {
        SourceType.YAML: YamlPathVerifier(),
        SourceType.JSON: JsonPathVerifier(),
        SourceType.MARKDOWN: text_verifier,
        SourceType.ADR: text_verifier,
    }


def _apply_result(evidence: Evidence, result: VerificationResult) -> None:
    evidence.verifier = result.verifier
    if result.status is VerificationStatus.VERIFIED:
        evidence.verified = True
    elif result.status is VerificationStatus.FAILED:
        evidence.verified = False
    else:
        evidence.verified = None


def verify_claims(
    claims: Iterable[Claim],
    documents: Iterable[SourceDocument],
    *,
    verifiers: Mapping[SourceType, Verifier] | None = None,
) -> list[VerificationResult]:
    """Verify all claim evidence without allowing one verifier error to abort."""

    documents_by_path = {document.path: document for document in documents}
    active_verifiers = _default_verifiers() if verifiers is None else verifiers
    results: list[VerificationResult] = []

    for claim in claims:
        for evidence in claim.evidence:
            verifier = active_verifiers.get(evidence.source_type)
            if verifier is None:
                result = _result(
                    evidence,
                    None,
                    VerificationStatus.ERROR,
                    f"no verifier for source type {evidence.source_type.value}",
                )
            else:
                document = documents_by_path.get(evidence.source_id)
                if document is None:
                    result = _result(
                        evidence,
                        verifier.name,
                        VerificationStatus.ERROR,
                        f"source document {evidence.source_id!r} was not loaded",
                    )
                else:
                    try:
                        result = verifier.verify(evidence, document)
                    except Exception as error:  # verifier boundary must be resilient
                        result = _result(
                            evidence,
                            verifier.name,
                            VerificationStatus.ERROR,
                            f"{type(error).__name__}: {error}",
                        )

            _apply_result(evidence, result)
            results.append(result)

    return results
