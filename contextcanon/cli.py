"""Placeholder command-line interface for ContextCanon M0."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="contextcanon",
        description=(
            "ContextCanon CLI placeholder; commands arrive in later milestones."
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the entry point
    raise SystemExit(main())
