"""Small provider-neutral semantic compiler for the frozen V2 IR."""

from __future__ import annotations

from datetime import date
from math import isfinite
import json
import re
from collections.abc import Callable, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from .models import Cardinality, Claim, Evidence, FactNeed, Modality


class SemanticCompilationError(ValueError):
    """Raised when a semantic compiler response violates the bounded contract."""


_CLAIM_FIELDS = {
    "subject",
    "predicate",
    "value",
    "value_type",
    "scope",
    "modality",
    "cardinality",
    "evidence_refs",
}
_VALUE_TYPES = {"string", "enum", "boolean", "number", "date"}
_UNKNOWN_SCOPE_VALUES = {"unknown", "?"}
_GROUPED_INTEGER = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+")
_ENGLISH_MONTHS = {
    name: number
    for number, month in enumerate(
        (
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ),
        start=1,
    )
    for name in (month, month[:3])
}


def _canonical(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _json_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    raise SemanticCompilationError(f"{field} must be a JSON scalar")


def _normalize_english_date(value: str) -> str | None:
    day_first = re.fullmatch(r"(\d{1,2}) ([A-Za-z]+\.?) (\d{4})", value)
    month_first = re.fullmatch(r"([A-Za-z]+\.?) (\d{1,2}), (\d{4})", value)
    if day_first:
        day_text, month_text, year_text = day_first.groups()
    elif month_first:
        month_text, day_text, year_text = month_first.groups()
    else:
        return None
    month = _ENGLISH_MONTHS.get(month_text.removesuffix(".").casefold())
    if month is None:
        return None
    return date(int(year_text), month, int(day_text)).isoformat()


def _normalize_value(value: Any, value_type: str) -> Any:
    if value_type in {"string", "enum"}:
        if not isinstance(value, str) or not value.strip():
            raise SemanticCompilationError(f"{value_type} value must be a non-empty string")
        return value.strip()
    if value_type == "boolean":
        if not isinstance(value, bool):
            raise SemanticCompilationError("boolean value must be JSON true or false")
        return value
    if value_type == "number":
        if isinstance(value, bool):
            raise SemanticCompilationError("number value cannot be boolean")
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not isfinite(value):
                raise SemanticCompilationError("number value must be finite")
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if _GROUPED_INTEGER.fullmatch(normalized):
                normalized = normalized.replace(",", "")
            try:
                parsed = Decimal(normalized)
            except (InvalidOperation, ValueError):
                raise SemanticCompilationError("number value is not numeric") from None
            if not parsed.is_finite():
                raise SemanticCompilationError("number value must be finite")
            return int(parsed) if parsed == parsed.to_integral_value() else float(parsed)
        raise SemanticCompilationError("number value must be numeric")
    if value_type == "date":
        if not isinstance(value, str):
            raise SemanticCompilationError("date value must be an ISO date string")
        normalized = value.strip()
        try:
            return date.fromisoformat(normalized).isoformat()
        except ValueError:
            try:
                english_date = _normalize_english_date(normalized)
            except ValueError:
                english_date = None
            if english_date is not None:
                return english_date
            raise SemanticCompilationError("date value must be an ISO date string") from None
    raise SemanticCompilationError(f"unsupported value_type: {value_type}")


def _matches_need(claim: Claim, needs: Sequence[FactNeed]) -> bool:
    subject = _canonical(claim.subject)
    predicate = _canonical(claim.predicate)
    value_type = _canonical(claim.value_type)
    return any(
        subject == _canonical(need.subject)
        and predicate == _canonical(need.dimension)
        and (need.value_type is None or value_type == _canonical(need.value_type))
        and _scope_matches_need(claim, need)
        for need in needs
    )


def _scope_matches_need(claim: Claim, need: FactNeed) -> bool:
    if need.scope_constraint is None:
        return True
    for key, expected in need.scope_constraint.items():
        if key not in claim.scope:
            return False
        actual = claim.scope[key]
        if actual is None or (
            isinstance(actual, str) and _canonical(actual) in _UNKNOWN_SCOPE_VALUES
        ):
            return False
        if _canonical(str(actual)) != _canonical(str(expected)):
            return False
    return True


def _prompt(evidence: Sequence[Evidence], needs: Sequence[FactNeed]) -> str:
    evidence_payload = [
        {
            "ref": f"E{index}",
            "source": item.source_id,
            "location": item.location,
            "content": item.content,
        }
        for index, item in enumerate(evidence, start=1)
    ]
    needs_payload = [
        {
            "subject": item.subject,
            "dimension": item.dimension,
            "scope_constraint": item.scope_constraint,
            "value_type": item.value_type,
        }
        for item in needs
    ]
    schema = {
        "claims": [
            {
                "subject": "...",
                "predicate": "...",
                "value": "typed value",
                "value_type": "string|enum|boolean|number|date",
                "scope": {"qualifier": "value"},
                "modality": "OBSERVED|DOCUMENTED|INTENDED|REQUIRED",
                "cardinality": "SINGLE|MULTI|UNKNOWN",
                "evidence_refs": ["E1"],
            }
        ]
    }
    return "\n".join(
        (
            "Compile only semantic Claims relevant to the supplied FactNeeds.",
            "For every emitted Claim, subject MUST exactly equal the canonical subject from one supplied FactNeed, and predicate MUST exactly equal the dimension from that same FactNeed.",
            "Evidence wording may use aliases or paraphrases, but normalize them onto those supplied FactNeed labels; do not invent facts.",
            'Use value_type="date" only for an exact calendar date with day, English month, and four-digit year, emitting ISO YYYY-MM-DD when possible; for partial, ranged, approximate, or relative temporal expressions such as 1856, January 1862, 1926-27, about 1870, or early 1970s, use value_type="string" and preserve the supported expression without inventing month, day, or other temporal precision.',
            "scope contains qualifiers that restrict where or when a Claim applies; preserve every qualifier explicitly stated by its supporting evidence, compiling each Claim independently.",
            "Map explicit current, currently, or now to scope.time=current; future, later, or planned future to scope.time=future; production to scope.environment=production; staging to scope.environment=staging; and explicit EU or US regions to scope.region with that region.",
            "If no relevant qualifier is stated, scope may be {}; do not invent qualifiers or drop an explicit scope because another source disagrees.",
            "Do not resolve contradictions, rank sources, or emit relations or summaries.",
            "Cardinality describes the semantic shape of the predicate: SINGLE is one scalar value within the Claim scope; MULTI means multiple distinct values may simultaneously be true; UNKNOWN means the evidence does not determine SINGLE versus MULTI.",
            "A contradiction between sources does not imply UNKNOWN cardinality: disagreeing scalar settings still produce separate SINGLE Claims, and governance compares them later.",
            "Modality meanings: OBSERVED is actual/runtime/current state; DOCUMENTED is documentation without clear runtime, requirement, or future intent; INTENDED is a plan or future target; REQUIRED is a requirement, policy, or must constraint.",
            "Do not change scope, modality, or cardinality to reconcile contradictory sources. Compile each supported Claim faithfully; governance resolves disagreement later.",
            "Never emit SUPPORTING, COMPATIBLE, CONFLICTING, or DIVERGENT; those are governance relations.",
            "evidence_refs may contain only the provided compact refs (E1, E2, ...).",
            "Never emit evidence IDs, source IDs, paths, locations, confidence, reasoning, relation, conflict, or claim_id.",
            "Return only JSON matching this schema:",
            json.dumps(schema, ensure_ascii=True, separators=(",", ":")),
            "FactNeeds:",
            json.dumps(needs_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
            "Evidence:",
            json.dumps(evidence_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        )
    )


def _parse(response: str, evidence: Sequence[Evidence], needs: Sequence[FactNeed]) -> tuple[Claim, ...]:
    if not isinstance(response, str):
        raise SemanticCompilationError("completion must return a string")
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as error:
        raise SemanticCompilationError("completion is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"claims"} or not isinstance(payload["claims"], list):
        raise SemanticCompilationError("response must contain only a claims list")

    aliases = {f"E{index}": item.evidence_id for index, item in enumerate(evidence, start=1)}
    claims: list[Claim] = []
    for item in payload["claims"]:
        if not isinstance(item, dict) or set(item) != _CLAIM_FIELDS:
            raise SemanticCompilationError("claim fields do not match the compiler schema")
        subject = item["subject"]
        predicate = item["predicate"]
        value_type = item["value_type"]
        if not isinstance(subject, str) or not subject.strip():
            raise SemanticCompilationError("subject must be a non-empty string")
        if not isinstance(predicate, str) or not predicate.strip():
            raise SemanticCompilationError("predicate must be a non-empty string")
        if not isinstance(value_type, str) or _canonical(value_type) not in _VALUE_TYPES:
            raise SemanticCompilationError("value_type is invalid")
        value_type = _canonical(value_type)
        scope = item["scope"]
        if not isinstance(scope, dict) or any(not isinstance(key, str) or not key for key in scope):
            raise SemanticCompilationError("scope must be an object with string keys")
        scope = {key: _json_value(value, field="scope") for key, value in scope.items()}
        modality = item["modality"]
        cardinality = item["cardinality"]
        try:
            modality = Modality(modality)
            cardinality = Cardinality(cardinality)
        except (TypeError, ValueError):
            raise SemanticCompilationError("modality or cardinality is invalid") from None
        refs = item["evidence_refs"]
        if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) for ref in refs):
            raise SemanticCompilationError("evidence_refs must be a non-empty string list")
        if len(set(refs)) != len(refs) or any(ref not in aliases for ref in refs):
            raise SemanticCompilationError("evidence_refs contains an unknown or repeated alias")
        claim = Claim(
            subject=subject.strip(),
            predicate=predicate.strip(),
            value=_normalize_value(item["value"], value_type),
            value_type=value_type,
            scope=scope,
            modality=modality,
            cardinality=cardinality,
            evidence_ids=tuple(aliases[ref] for ref in refs),
        )
        if _matches_need(claim, needs):
            claims.append(claim)
    return tuple(claims)


class SemanticCompiler:
    """Compile raw Evidence into validated Claims using an injected completion."""

    def __init__(self, complete: Callable[[str], str]):
        self._complete = complete

    def compile(
        self,
        evidence: Sequence[Evidence],
        fact_needs: Sequence[FactNeed],
    ) -> tuple[Claim, ...]:
        evidence = tuple(evidence)
        fact_needs = tuple(fact_needs)
        if not evidence or not fact_needs:
            return ()
        prompt = _prompt(evidence, fact_needs)
        try:
            response = self._complete(prompt)
        except Exception as error:
            raise SemanticCompilationError("semantic completion failed") from error
        return _parse(response, evidence, fact_needs)


__all__ = ["SemanticCompilationError", "SemanticCompiler"]
