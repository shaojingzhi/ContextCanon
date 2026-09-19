"""Command-line interface for ContextCanon."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .core import Claim, ClaimType, Evidence, JSONValue, Resolution
from .extraction import extract_demo_claims
from .resolution import DefaultResolutionPolicy
from .sources import load_sources
from .verification import verify_claims


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextcanon",
        description="Compile repository knowledge into verifiable context.",
    )
    subparsers = parser.add_subparsers(dest="command")
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="diagnose supported repository knowledge",
    )
    doctor_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="repository directory to diagnose (default: current directory)",
    )
    return parser


def _group_claims(claims: list[Claim]) -> dict[tuple[str, str], list[Claim]]:
    groups: dict[tuple[str, str], list[Claim]] = {}
    for claim in claims:
        groups.setdefault((claim.subject, claim.predicate), []).append(claim)
    return groups


def _format_value(value: JSONValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _verification_label(evidence: Evidence) -> str:
    if evidence.verified is True:
        return "verified"
    if evidence.verified is False:
        return "failed"
    return "unverified"


def _render_selected_claim(
    lines: list[str],
    heading: str,
    claim: Claim,
) -> None:
    lines.extend([f"{heading}:", f"  {_format_value(claim.value)}"])
    if claim.evidence:
        lines.append("  Evidence:")
        for evidence in sorted(
            claim.evidence,
            key=lambda item: (item.source_id, item.location, item.id),
        ):
            lines.append(
                f"    - {evidence.source_id} [{_verification_label(evidence)}]"
            )
    lines.append("")


def _render_resolution(
    property_key: tuple[str, str],
    resolution: Resolution,
) -> str:
    subject, predicate = property_key
    lines = [f"Property: {subject}.{predicate}", ""]

    for claim_type, heading in (
        (ClaimType.RUNTIME_STATE, "Current runtime"),
        (ClaimType.ARCHITECTURE_INTENT, "Architecture intent"),
    ):
        for claim in resolution.selected_claims:
            if claim.claim_type == claim_type:
                _render_selected_claim(lines, heading, claim)

    lines.extend(["Status:", f"  {resolution.status.value}", ""])

    if resolution.conflicting_claims:
        lines.append("Conflicting claims:")
        for claim in resolution.conflicting_claims:
            lines.append(f"  - {_format_value(claim.value)}")
            for evidence in sorted(
                claim.evidence,
                key=lambda item: (item.source_id, item.location, item.id),
            ):
                lines.append(
                    f"    - {evidence.source_id} "
                    f"[{_verification_label(evidence)}]"
                )
        lines.append("")

    if resolution.reason_codes:
        lines.append("Reasons:")
        lines.extend(f"  - {reason.value}" for reason in resolution.reason_codes)
        lines.append("")

    if resolution.explanation:
        lines.extend(["Explanation:", f"  {resolution.explanation}"])

    return "\n".join(lines).rstrip()


def _run_doctor(root: Path) -> int:
    if not root.exists() or not root.is_dir():
        print(
            f"contextcanon doctor: error: {root} is not a directory",
            file=sys.stderr,
        )
        return 2

    documents = load_sources(root)
    # M4 intentionally uses the deterministic V0.1 demo extractor.
    claims = extract_demo_claims(documents)
    if not claims:
        print("No supported knowledge claims found.")
        return 0

    verify_claims(claims, documents)
    groups = _group_claims(claims)
    policy = DefaultResolutionPolicy()
    rendered = [
        _render_resolution(property_key, policy.resolve(groups[property_key]))
        for property_key in sorted(groups)
    ]
    print("\n\n".join(rendered))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _run_doctor(Path(args.path))
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the entry point
    raise SystemExit(main())
