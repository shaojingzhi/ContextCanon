"""Placeholder command-line interface for ContextCanon M0."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextcanon",
        description="ContextCanon CLI (M0 placeholder)",
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="A future command such as scan, doctor, or build.",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(argv)
    if parsed.command is None:
        parser.print_help()
        return 0
    print(
        f"contextcanon {parsed.command!r} is not implemented yet; "
        "M0 provides the core models only."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the entry point
    raise SystemExit(main())
