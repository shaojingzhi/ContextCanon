"""Command-line interface for ContextCanon."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .assembly import assemble_context, source_revision, utc_timestamp
from .core import Claim, ClaimType, Evidence, JSONValue, Resolution
from .extraction import extract_demo_claims
from .renderers import JSONRenderer, MarkdownRenderer
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
    build_parser = subparsers.add_parser(
        "build",
        help="build a structured context package",
    )
    build_parser.add_argument(
        "task",
        help="task recorded in the context package metadata",
    )
    build_parser.add_argument(
        "--path",
        default=".",
        help="repository directory to build from (default: current directory)",
    )
    build_parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="output format (default: markdown)",
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


def _validate_root(root: Path, *, command: str) -> bool:
    if not root.exists() or not root.is_dir():
        print(
            f"contextcanon {command}: error: {root} is not a directory",
            file=sys.stderr,
        )
        return False
    return True


def _resolve_repository(
    root: Path,
) -> list[tuple[tuple[str, str], Resolution]]:
    documents = load_sources(root)
    # V0.1 intentionally uses the deterministic demo extractor.
    claims = extract_demo_claims(documents)
    verify_claims(claims, documents)
    groups = _group_claims(claims)
    policy = DefaultResolutionPolicy()
    return [
        (property_key, policy.resolve(groups[property_key]))
        for property_key in sorted(groups)
    ]


def _run_doctor(root: Path) -> int:
    if not _validate_root(root, command="doctor"):
        return 2

    resolved_properties = _resolve_repository(root)
    if not resolved_properties:
        print("No supported knowledge claims found.")
        return 0

    rendered = [
        _render_resolution(property_key, resolution)
        for property_key, resolution in resolved_properties
    ]
    print("\n\n".join(rendered))
    return 0


def _run_build(root: Path, *, task: str, output_format: str) -> int:
    if not _validate_root(root, command="build"):
        return 2

    policy = DefaultResolutionPolicy()
    package = assemble_context(
        task=task,
        resolutions=[
            resolution
            for _, resolution in _resolve_repository(root)
        ],
        source_revision=source_revision(root),
        created_at=utc_timestamp(),
        policy_name=policy.name,
        policy_version=policy.version,
    )
    renderer = JSONRenderer() if output_format == "json" else MarkdownRenderer()
    print(renderer.render(package), end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _run_doctor(Path(args.path))
    if args.command == "build":
        return _run_build(
            Path(args.path),
            task=args.task,
            output_format=args.format,
        )
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the entry point
    raise SystemExit(main())
