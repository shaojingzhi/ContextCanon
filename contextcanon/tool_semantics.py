"""Tool semantic classification for runtime routing."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Protocol

from .core import JSONValue
from .semantic import (
    SemanticModelClient,
    parse_semantic_response,
    sanitize_runtime_value,
)


class ToolSemantics(StrEnum):
    READ = "READ"
    VERIFY = "VERIFY"
    SIDE_EFFECT = "SIDE_EFFECT"
    UNKNOWN = "UNKNOWN"


class ToolSemanticsResolver(Protocol):
    def classify(
        self,
        tool_name: str,
        *,
        description: str | None = None,
        input_schema: JSONValue = None,
        annotations: JSONValue = None,
    ) -> ToolSemantics:
        ...

    def prepare(self, tools: Mapping[str, Mapping[str, JSONValue]]) -> None:
        """Resolve stable tool metadata before normal calls begin."""
        ...


class StaticToolSemanticsResolver:
    """Explicit compatibility mapping for environments with known tool metadata."""

    def __init__(self, semantics: Mapping[str, ToolSemantics]) -> None:
        self.semantics = dict(semantics)

    def classify(
        self,
        tool_name: str,
        *,
        description: str | None = None,
        input_schema: JSONValue = None,
        annotations: JSONValue = None,
    ) -> ToolSemantics:
        return self.semantics.get(tool_name, ToolSemantics.UNKNOWN)

    def prepare(self, tools: Mapping[str, Mapping[str, JSONValue]]) -> None:
        """Static mappings need no discovery-time work."""
        return None


def _annotation_semantics(annotations: JSONValue) -> ToolSemantics | None:
    if not isinstance(annotations, Mapping):
        return None
    explicit = annotations.get("contextcanon_semantics")
    if isinstance(explicit, str):
        try:
            return ToolSemantics(explicit.upper())
        except ValueError:
            return None
    read_only = annotations.get("readOnlyHint", annotations.get("read_only_hint"))
    if read_only is True:
        return ToolSemantics.READ
    destructive = annotations.get(
        "destructiveHint", annotations.get("destructive_hint")
    )
    if destructive is True:
        return ToolSemantics.SIDE_EFFECT
    return None


class LLMToolSemanticsResolver:
    """Classify tool behavior without making an action authorization decision."""

    def __init__(
        self,
        client: SemanticModelClient,
        *,
        model: str = "deepseek-v4-pro",
        minimum_confidence: float = 0.7,
    ) -> None:
        self.client = client
        self.model = model
        self.minimum_confidence = minimum_confidence
        self.diagnostics: list[str] = []
        self._cache: dict[str, ToolSemantics] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def prepare(self, tools: Mapping[str, Mapping[str, JSONValue]]) -> None:
        """Prime the task-local cache from one tool-discovery snapshot.

        Metadata-derived classifications are free.  Only unresolved tools use
        the model, and each stable metadata identity is classified once.
        """
        for name in sorted(tools):
            metadata = tools[name]
            self.classify(
                name,
                description=metadata.get("description"),
                input_schema=metadata.get("input_schema"),
                annotations=metadata.get("annotations"),
            )

    @staticmethod
    def _cache_key(
        tool_name: str,
        description: str | None,
        input_schema: JSONValue,
        annotations: JSONValue,
    ) -> str:
        return json.dumps(
            [tool_name, description, input_schema, annotations],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    def classify(
        self,
        tool_name: str,
        *,
        description: str | None = None,
        input_schema: JSONValue = None,
        annotations: JSONValue = None,
    ) -> ToolSemantics:
        explicit = _annotation_semantics(annotations)
        if explicit is not None:
            return explicit
        key = self._cache_key(tool_name, description, input_schema, annotations)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1
        prompt = (
            "Classify only the operational semantics of this tool. Return JSON with "
            "semantics and confidence. semantics must be READ, VERIFY, SIDE_EFFECT, or "
            "UNKNOWN. READ retrieves information without changing external state. VERIFY "
            "checks information without changing external state. SIDE_EFFECT changes "
            "external state. Do not decide whether the tool should be allowed. When the "
            "available metadata is insufficient, return UNKNOWN.\n"
            + json.dumps(
                {
                    "tool_name": tool_name,
                    "description": description,
                    "input_schema": sanitize_runtime_value(input_schema),
                    "annotations": sanitize_runtime_value(annotations),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        try:
            payload = parse_semantic_response(
                self.client.complete(prompt, model=self.model)
            )
            if not isinstance(payload, Mapping):
                raise ValueError("tool semantics response must be an object")
            semantics = ToolSemantics(str(payload.get("semantics", "UNKNOWN")).upper())
            confidence = float(payload.get("confidence", 0.0))
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("tool semantics confidence is out of range")
            result = ToolSemantics.UNKNOWN if confidence < self.minimum_confidence else semantics
            self._cache[key] = result
            return result
        except Exception as exc:
            self.diagnostics.append(f"tool_semantics_error:{type(exc).__name__}")
            self._cache[key] = ToolSemantics.UNKNOWN
            return ToolSemantics.UNKNOWN


__all__ = [
    "LLMToolSemanticsResolver",
    "StaticToolSemanticsResolver",
    "ToolSemantics",
    "ToolSemanticsResolver",
]
