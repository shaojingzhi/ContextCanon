"""Stable local paths for the optional AgentAbstain runtime assets."""

from __future__ import annotations

import os
from pathlib import Path


RUNTIME_ROOT = Path.home() / ".contextcanon" / "agentabstain"
DEFAULT_REPO = RUNTIME_ROOT / "repo"
DEFAULT_DATA = RUNTIME_ROOT / "data"


def configured_repo() -> str:
    return os.environ.get("AGENTABSTAIN_REPO", str(DEFAULT_REPO))


def configured_data() -> str:
    return os.environ.get("AGENTABSTAIN_DATA", str(DEFAULT_DATA))


def validate_runtime_paths(repo: Path, data: Path) -> list[str]:
    missing: list[str] = []
    if not repo.is_dir():
        missing.append(f"AgentAbstain repo not found: {repo}")
    if not (data / "tasks").is_dir():
        missing.append(f"AgentAbstain dataset not found: {data}")
    return missing


__all__ = [
    "DEFAULT_DATA",
    "DEFAULT_REPO",
    "RUNTIME_ROOT",
    "configured_data",
    "configured_repo",
    "validate_runtime_paths",
]
