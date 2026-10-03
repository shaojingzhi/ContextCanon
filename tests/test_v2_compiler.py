from __future__ import annotations

import json
import unittest

from contextcanon_v2 import (
    Cardinality,
    Claim,
    Evidence,
    FactNeed,
    Modality,
    Relation,
    SemanticCompilationError,
    SemanticCompiler,
    SummaryState,
    govern,
)


def _evidence(evidence_id: str, content: str, location: str = "record") -> Evidence:
    return Evidence(evidence_id, f"source-{evidence_id}", location, content)


def _response(*claims: dict) -> str:
    return json.dumps({"claims": list(claims)})


def _claim(**overrides: object) -> dict:
    value = {
        "subject": "auth",
        "predicate": "protocol",
        "value": "JWT",
        "value_type": "enum",
        "scope": {"time": "current"},
        "modality": "OBSERVED",
        "cardinality": "SINGLE",
        "evidence_refs": ["E1"],
    }
    value.update(overrides)
    return value


class V2CompilerTests(unittest.TestCase):
    def test_simple_scalar_extraction_preserves_runtime_provenance(self) -> None:
        evidence = [Evidence("ev-secret", "source-runtime", "record", "provider: jwt")]
        prompts: list[str] = []
        compiler = SemanticCompiler(lambda prompt: prompts.append(prompt) or _response(_claim()))

        claims = compiler.compile(evidence, (FactNeed("auth", "protocol", value_type="enum"),))

        self.assertEqual(claims[0].value, "JWT")
        self.assertEqual(claims[0].evidence_ids, ("ev-secret",))
        self.assertIn('"ref":"E1"', prompts[0])
        self.assertNotIn("ev-secret", prompts[0])

    def test_batch_normalizes_entity_and_predicate_aliases_via_contract(self) -> None:
        evidence = [_evidence("EVIDENCE-A", "The authentication method is JWT")]
        compiler = SemanticCompiler(lambda _prompt: _response(_claim(subject="auth", predicate="protocol")))

        claims = compiler.compile(evidence, (FactNeed("auth", "protocol", value_type="enum"),))

        self.assertEqual((claims[0].subject, claims[0].predicate), ("auth", "protocol"))

    def test_current_and_future_scope_and_modalities_are_preserved(self) -> None:
        evidence = [_evidence("current", "JWT now"), _evidence("future", "OAuth2 later")]
        compiler = SemanticCompiler(
            lambda _prompt: _response(
                _claim(value="JWT", scope={"time": "current"}, evidence_refs=["E1"]),
                _claim(value="OAuth2", scope={"time": "future"}, modality="INTENDED", evidence_refs=["E2"]),
            )
        )

        claims = compiler.compile(evidence, (FactNeed("auth", "protocol", value_type="enum"),))

        self.assertEqual(claims[0].scope, {"time": "current"})
        self.assertEqual(claims[1].scope, {"time": "future"})
        self.assertEqual(claims[1].modality, Modality.INTENDED)

    def test_required_multi_and_unknown_cardinality_are_valid(self) -> None:
        evidence = [_evidence("a", "roles"), _evidence("b", "roles")]
        compiler = SemanticCompiler(
            lambda _prompt: _response(
                _claim(predicate="roles", value="admin", modality="REQUIRED", cardinality="MULTI"),
                _claim(predicate="roles", value="auditor", modality="REQUIRED", cardinality="UNKNOWN", evidence_refs=["E2"]),
            )
        )

        claims = compiler.compile(evidence, (FactNeed("auth", "roles", value_type="enum"),))

        self.assertEqual([claim.cardinality for claim in claims], [Cardinality.MULTI, Cardinality.UNKNOWN])
        self.assertEqual([claim.modality for claim in claims], [Modality.REQUIRED, Modality.REQUIRED])

    def test_irrelevant_claims_are_not_materialized(self) -> None:
        evidence = [_evidence("auth-source", "JWT"), _evidence("db-source", "Postgres")]
        compiler = SemanticCompiler(
            lambda _prompt: _response(
                _claim(),
                _claim(subject="database", predicate="engine", value="Postgres", evidence_refs=["E2"]),
            )
        )

        claims = compiler.compile(evidence, (FactNeed("auth", "protocol", value_type="enum"),))

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].evidence_ids, ("auth-source",))

    def test_multiple_fact_needs_produce_multiple_claims(self) -> None:
        evidence = [_evidence("enabled", "enabled"), _evidence("retry", "retry")]
        compiler = SemanticCompiler(
            lambda _prompt: _response(
                _claim(subject="service", predicate="enabled", value=True, value_type="boolean", evidence_refs=["E1"]),
                _claim(subject="service", predicate="max_retry", value=3, value_type="number", evidence_refs=["E2"]),
            )
        )

        claims = compiler.compile(
            evidence,
            (
                FactNeed("service", "enabled", value_type="boolean"),
                FactNeed("service", "max_retry", value_type="number"),
            ),
        )

        self.assertEqual([(claim.predicate, claim.value) for claim in claims], [("enabled", True), ("max_retry", 3)])

    def test_invalid_json_fails_cleanly(self) -> None:
        compiler = SemanticCompiler(lambda _prompt: "not json")
        with self.assertRaises(SemanticCompilationError):
            compiler.compile([_evidence("e1", "JWT")], (FactNeed("auth", "protocol"),))

    def test_unknown_evidence_alias_fails_cleanly(self) -> None:
        compiler = SemanticCompiler(lambda _prompt: _response(_claim(evidence_refs=["E9"])))
        with self.assertRaises(SemanticCompilationError):
            compiler.compile([_evidence("e1", "JWT")], (FactNeed("auth", "protocol"),))

    def test_missing_evidence_refs_fails_cleanly(self) -> None:
        compiler = SemanticCompiler(lambda _prompt: _response(_claim(evidence_refs=[])))
        with self.assertRaises(SemanticCompilationError):
            compiler.compile([_evidence("e1", "JWT")], (FactNeed("auth", "protocol"),))

    def test_relation_and_conflict_fields_are_rejected(self) -> None:
        invalid = _claim(relation="CONFLICTING")
        compiler = SemanticCompiler(lambda _prompt: _response(invalid))
        with self.assertRaises(SemanticCompilationError):
            compiler.compile([_evidence("e1", "JWT")], (FactNeed("auth", "protocol"),))

    def test_invalid_typed_value_and_scope_shapes_fail_cleanly(self) -> None:
        bad_value = SemanticCompiler(lambda _prompt: _response(_claim(value="enabled", value_type="boolean")))
        with self.assertRaises(SemanticCompilationError):
            bad_value.compile([_evidence("e1", "enabled")], (FactNeed("auth", "protocol"),))

        bad_scope = SemanticCompiler(lambda _prompt: _response(_claim(scope={"time": {"nested": True}})))
        with self.assertRaises(SemanticCompilationError):
            bad_scope.compile([_evidence("e1", "JWT")], (FactNeed("auth", "protocol"),))

    def test_compilation_failure_can_feed_governance_incomplete(self) -> None:
        compiler = SemanticCompiler(lambda _prompt: "{}")
        needs = (FactNeed("auth", "protocol"),)
        with self.assertRaises(SemanticCompilationError):
            compiler.compile([_evidence("e1", "JWT")], needs)
        result = govern((), needs, compilation_error=True)
        self.assertEqual(result.summary_state, SummaryState.INCOMPLETE)

    def test_offline_compiler_to_governance_conflict(self) -> None:
        evidence = [_evidence("date-a", "March 22"), _evidence("date-b", "March 23")]
        compiler = SemanticCompiler(
            lambda _prompt: _response(
                _claim(subject="Spring Gala", predicate="event_date", value="2026-03-22", value_type="date", evidence_refs=["E1"]),
                _claim(subject="Spring Gala", predicate="event_date", value="2026-03-23", value_type="date", evidence_refs=["E2"]),
            )
        )
        needs = (FactNeed("Spring Gala", "event_date", value_type="date"),)

        claims = compiler.compile(evidence, needs)
        result = govern(claims, needs)

        self.assertEqual(result.relations[0].relation, Relation.CONFLICTING)
        self.assertEqual(result.summary_state, SummaryState.UNRESOLVED)


if __name__ == "__main__":
    unittest.main()
