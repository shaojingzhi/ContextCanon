"""Render structured ContextPackages for human and machine consumers."""

from __future__ import annotations

import json
from typing import Protocol

from .core import ClaimType, ContextPackage, Evidence, JSONValue


class ContextRenderer(Protocol):
    """Minimal interface shared by the two M5 renderers."""

    def render(self, package: ContextPackage) -> str:
        """Render a ContextPackage without changing its semantics."""


class JSONRenderer:
    """Render the core model's existing serialization as deterministic JSON."""

    def render(self, package: ContextPackage) -> str:
        return json.dumps(
            package.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def _value_text(value: JSONValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _verification_label(evidence: Evidence) -> str:
    if evidence.verified is True:
        return "verified"
    if evidence.verified is False:
        return "failed"
    return "unverified"


def _claim_type_label(claim_type: ClaimType) -> str:
    labels = {
        ClaimType.RUNTIME_STATE: "Runtime State",
        ClaimType.ARCHITECTURE_INTENT: "Architecture Intent",
        ClaimType.OPERATIONAL_RULE: "Operational Rule",
    }
    return labels[claim_type]


class MarkdownRenderer:
    """Render a concise, provenance-preserving Markdown artifact."""

    def render(self, package: ContextPackage) -> str:
        lines = [
            "# Context Package",
            "",
            f"- ID: {package.id}",
            f"- Task: {package.task}",
            f"- Source revision: {package.source_revision}",
            f"- Policy: {package.policy_name}@{package.policy_version}",
            f"- Created at: {package.created_at}",
        ]
        if package.token_budget is not None:
            lines.append(f"- Token budget: {package.token_budget}")

        for item in package.items:
            claim = item.claim
            lines.extend(
                [
                    "",
                    (
                        f"## {claim.subject}.{claim.predicate} — "
                        f"{_claim_type_label(claim.claim_type)}"
                    ),
                    "",
                    f"Value: {_value_text(claim.value)}",
                    "",
                    f"Status: {item.resolution_status.value}",
                ]
            )
            if item.supporting_evidence:
                lines.extend(["", "Evidence:"])
                lines.extend(
                    (
                        f"- {evidence.source_id} — "
                        f"{_verification_label(evidence)}"
                    )
                    for evidence in item.supporting_evidence
                )
            if item.reason_codes:
                lines.extend(["", "Reason codes:"])
                lines.extend(f"- {reason.value}" for reason in item.reason_codes)
            if item.explanation:
                lines.extend(["", "Explanation:", item.explanation])

        if package.unresolved_conflicts:
            lines.extend(["", "## Unresolved Knowledge"])
            for resolution in package.unresolved_conflicts:
                lines.extend(["", f"### {resolution.status.value}"])
                if resolution.reason_codes:
                    lines.extend(["", "Reason codes:"])
                    lines.extend(
                        f"- {reason.value}" for reason in resolution.reason_codes
                    )
                if resolution.explanation:
                    lines.extend(["", "Explanation:", resolution.explanation])

        return "\n".join(lines).rstrip() + "\n"

