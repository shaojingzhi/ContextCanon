"""Load the structured and prose sources used by the M1 demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Protocol

import yaml

from .core import JSONValue, SourceType


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One loaded source with repository-relative provenance."""

    path: str
    source_type: SourceType
    content: JSONValue


class SourceLoader(Protocol):
    """Minimal interface for loading one supported source file."""

    def supports(self, path: Path) -> bool:
        """Return whether this loader handles the path."""

    def load(self, path: Path, *, root: Path) -> SourceDocument:
        """Load a file whose path is contained by root."""


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _json_value(value: object, *, location: str) -> JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{location} must not contain NaN or infinity")
        return value
    if isinstance(value, list):
        return [_json_value(item, location=f"{location}[]") for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{location} must use string object keys")
        return {
            key: _json_value(value[key], location=f"{location}.{key}")
            for key in sorted(value)
        }
    raise TypeError(f"{location} contains non-JSON value {type(value).__name__}")


class MarkdownLoader:
    """Load Markdown as raw text, classifying ADR paths separately."""

    _suffixes = frozenset({".md", ".markdown"})

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self._suffixes

    def load(self, path: Path, *, root: Path) -> SourceDocument:
        relative_path = _relative_path(path, root)
        is_adr = path.stem.upper().startswith("ADR-") or "adr" in {
            part.lower() for part in Path(relative_path).parts[:-1]
        }
        return SourceDocument(
            path=relative_path,
            source_type=SourceType.ADR if is_adr else SourceType.MARKDOWN,
            content=path.read_text(encoding="utf-8"),
        )


class ConfigLoader:
    """Load YAML and JSON into JSON-compatible Python values."""

    _source_types = {
        ".json": SourceType.JSON,
        ".yaml": SourceType.YAML,
        ".yml": SourceType.YAML,
    }

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self._source_types

    def load(self, path: Path, *, root: Path) -> SourceDocument:
        suffix = path.suffix.lower()
        source_type = self._source_types[suffix]
        text = path.read_text(encoding="utf-8")
        parsed = (
            json.loads(text)
            if source_type is SourceType.JSON
            else yaml.safe_load(text)
        )
        return SourceDocument(
            path=_relative_path(path, root),
            source_type=source_type,
            content=_json_value(parsed, location=path.as_posix()),
        )


def load_sources(
    root: Path,
    loaders: tuple[SourceLoader, ...] | None = None,
) -> list[SourceDocument]:
    """Load supported files below root in deterministic path order."""

    active_loaders = (
        (MarkdownLoader(), ConfigLoader()) if loaders is None else loaders
    )
    documents: list[SourceDocument] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        loader = next(
            (candidate for candidate in active_loaders if candidate.supports(path)),
            None,
        )
        if loader is not None:
            documents.append(loader.load(path, root=root))
    return documents
