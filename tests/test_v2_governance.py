from __future__ import annotations

import unittest
from dataclasses import fields
from pathlib import Path

import yaml

from contextcanon_v2 import (
    Cardinality,
    Claim,
    FactNeed,
    Modality,
    Relation,
    SummaryState,
    govern,
)


CASES = Path(__file__).parents[1] / "benchmarks" / "v2_design_cases.yaml"


def _load_case(case: dict) -> tuple[list[Claim], tuple[FactNeed, ...]]:
    claims = [
        Claim(
            subject=item["subject"],
            predicate=item["predicate"],
            value=item["value"],
            value_type=item["value_type"],
            scope=dict(item.get("scope") or {}),
            modality=Modality(item["modality"]),
            cardinality=Cardinality(item["cardinality"]),
            evidence_ids=tuple(item["evidence_ids"]),
        )
        for item in case["gold_claims"]
    ]
    needs = tuple(
        FactNeed(
            item["subject"],
            item["dimension"],
            item.get("scope_constraint"),
            item.get("value_type"),
        )
        for item in case["fact_needs"]
    )
    return claims, needs


class V2GovernanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = yaml.safe_load(CASES.read_text())

    def test_all_36_cases_pass(self) -> None:
        self.assertEqual(len(self.cases), 36)
        for case in self.cases:
            with self.subTest(case=case["case_id"]):
                claims, needs = _load_case(case)
                result = govern(claims, needs)
                actual = sorted(record.relation.value for record in result.relations)
                expected = sorted(case["gold_relations"])
                self.assertEqual(actual, expected)
                self.assertEqual(result.summary_state.value, case["gold_summary"])

    def test_event_date_cases_are_only_three(self) -> None:
        event_date = [
            case for case in self.cases
            if any(item["dimension"] == "event_date" for item in case["fact_needs"])
        ]
        self.assertEqual(len(event_date), 3)

    def test_provenance_evidence_ids_survive_unchanged(self) -> None:
        for case in self.cases:
            claims, _needs = _load_case(case)
            expected = [
                tuple(item["evidence_ids"])
                for item in case["gold_claims"]
            ]
            self.assertEqual([claim.evidence_ids for claim in claims], expected)

    def test_unrelated_subject_and_predicate_do_not_create_relations(self) -> None:
        claims = [
            Claim("a", "status", "open", "string", {"time": "current"}, Modality.OBSERVED, Cardinality.SINGLE, ("e1",)),
            Claim("b", "status", "closed", "string", {"time": "current"}, Modality.OBSERVED, Cardinality.SINGLE, ("e2",)),
            Claim("a", "owner", "Alice", "string", {"time": "current"}, Modality.OBSERVED, Cardinality.SINGLE, ("e3",)),
        ]
        result = govern(claims, (FactNeed("a", "status"),))
        self.assertEqual(result.relations, ())
        self.assertEqual(result.summary_state, SummaryState.CLEAR)

    def test_single_usable_claim_is_clear(self) -> None:
        claim = Claim("auth", "protocol", "JWT", "enum", {}, Modality.OBSERVED, Cardinality.SINGLE, ("e1",))
        self.assertEqual(govern([claim], (FactNeed("auth", "protocol"),)).summary_state, SummaryState.CLEAR)

    def test_zero_usable_claim_is_incomplete(self) -> None:
        self.assertEqual(
            govern([], (FactNeed("auth", "protocol"),)).summary_state,
            SummaryState.INCOMPLETE,
        )

    def test_compilation_failure_is_incomplete(self) -> None:
        claim = Claim("auth", "protocol", "JWT", "enum", {}, Modality.OBSERVED, Cardinality.SINGLE, ("e1",))
        self.assertEqual(
            govern([claim], (FactNeed("auth", "protocol"),), compilation_error=True).summary_state,
            SummaryState.INCOMPLETE,
        )

    def test_governance_result_has_no_persistent_claim_id_requirement(self) -> None:
        claim = Claim("auth", "protocol", "JWT", "enum", {}, Modality.OBSERVED, Cardinality.SINGLE, ("e1",))
        result = govern([claim], (FactNeed("auth", "protocol"),))
        self.assertFalse(hasattr(claim, "claim_id"))
        self.assertNotIn("unresolved_claim_ids", {field.name for field in fields(result)})
        self.assertEqual(result.unresolved_claims(), ())

    def test_unresolved_claims_are_derived_from_relation_endpoints(self) -> None:
        claims = [
            Claim("auth", "protocol", "JWT", "enum", {"time": "current"}, Modality.OBSERVED, Cardinality.SINGLE, ("e1",)),
            Claim("auth", "protocol", "OAuth2", "enum", {"time": "current"}, Modality.OBSERVED, Cardinality.SINGLE, ("e2",)),
        ]
        result = govern(claims, (FactNeed("auth", "protocol"),))
        self.assertEqual(result.summary_state, SummaryState.UNRESOLVED)
        self.assertEqual(
            {claim.evidence_ids for claim in result.unresolved_claims()},
            {claim.evidence_ids for claim in result.claims},
        )

    def test_multiple_fact_needs_aggregate_all_required_dimensions(self) -> None:
        claims = [
            Claim("service", "enabled", True, "boolean", {"time": "current"}, Modality.OBSERVED, Cardinality.SINGLE, ("e1",)),
            Claim("service", "max_retry", 3, "number", {"time": "current"}, Modality.OBSERVED, Cardinality.SINGLE, ("e2",)),
        ]
        needs = (
            FactNeed("service", "enabled", value_type="boolean"),
            FactNeed("service", "max_retry", value_type="number"),
        )
        self.assertEqual(govern(claims, needs).summary_state, SummaryState.CLEAR)

    def test_one_incomplete_fact_need_makes_result_incomplete(self) -> None:
        claim = Claim("service", "enabled", True, "boolean", {"time": "current"}, Modality.OBSERVED, Cardinality.SINGLE, ("e1",))
        needs = (FactNeed("service", "enabled"), FactNeed("service", "max_retry"))
        self.assertEqual(govern([claim], needs).summary_state, SummaryState.INCOMPLETE)

    def test_complete_needs_with_one_unresolved_dimension_are_unresolved(self) -> None:
        claims = [
            Claim("service", "enabled", True, "boolean", {"time": "current"}, Modality.OBSERVED, Cardinality.SINGLE, ("e1",)),
            Claim("service", "max_retry", 3, "number", {"time": "current"}, Modality.OBSERVED, Cardinality.SINGLE, ("e2",)),
            Claim("service", "max_retry", 5, "number", {"time": "current"}, Modality.OBSERVED, Cardinality.SINGLE, ("e3",)),
        ]
        needs = (FactNeed("service", "enabled"), FactNeed("service", "max_retry"))
        self.assertEqual(govern(claims, needs).summary_state, SummaryState.UNRESOLVED)

    def test_p1_and_p2_use_precise_multiple_fact_needs(self) -> None:
        cases = {case["case_id"]: case for case in self.cases}
        self.assertEqual(
            [(item["subject"], item["dimension"]) for item in cases["P1"]["fact_needs"]],
            [("service", "enabled"), ("service", "max_retry")],
        )
        self.assertEqual(
            [(item["subject"], item["dimension"]) for item in cases["P2"]["fact_needs"]],
            [("security", "retention"), ("security", "mfa_required")],
        )

    def test_unknown_is_not_silently_converted_to_conflicting(self) -> None:
        claims = [
            Claim("product", "region", "US", "enum", {"time": "current"}, Modality.OBSERVED, Cardinality.UNKNOWN, ("e1",)),
            Claim("product", "region", "EU", "enum", {"time": "current"}, Modality.OBSERVED, Cardinality.UNKNOWN, ("e2",)),
        ]
        result = govern(claims, (FactNeed("product", "region"),))
        self.assertEqual([record.relation for record in result.relations], [Relation.UNKNOWN])

    def test_documented_unequal_values_are_unknown(self) -> None:
        claims = [
            Claim("auth", "protocol", "JWT", "enum", {"time": "current"}, Modality.DOCUMENTED, Cardinality.SINGLE, ("e1",)),
            Claim("auth", "protocol", "OAuth2", "enum", {"time": "current"}, Modality.DOCUMENTED, Cardinality.SINGLE, ("e2",)),
        ]
        result = govern(claims, (FactNeed("auth", "protocol"),))
        self.assertEqual(result.relations[0].relation, Relation.UNKNOWN)

    def test_multi_multi_unequal_is_compatible(self) -> None:
        claims = [
            Claim("product", "region", "US", "enum", {"time": "current"}, Modality.OBSERVED, Cardinality.MULTI, ("e1",)),
            Claim("product", "region", "EU", "enum", {"time": "current"}, Modality.OBSERVED, Cardinality.MULTI, ("e2",)),
        ]
        self.assertEqual(govern(claims, (FactNeed("product", "region"),)).relations[0].relation, Relation.COMPATIBLE)

    def test_single_multi_unequal_is_unknown(self) -> None:
        claims = [
            Claim("product", "region", "US", "enum", {"time": "current"}, Modality.OBSERVED, Cardinality.SINGLE, ("e1",)),
            Claim("product", "region", "EU", "enum", {"time": "current"}, Modality.OBSERVED, Cardinality.MULTI, ("e2",)),
        ]
        self.assertEqual(govern(claims, (FactNeed("product", "region"),)).relations[0].relation, Relation.UNKNOWN)

    def test_current_observed_and_required_are_divergent(self) -> None:
        claims = [
            Claim("auth", "protocol", "JWT", "enum", {"time": "current"}, Modality.OBSERVED, Cardinality.SINGLE, ("e1",)),
            Claim("auth", "protocol", "OAuth2", "enum", {"time": "current"}, Modality.REQUIRED, Cardinality.SINGLE, ("e2",)),
        ]
        self.assertEqual(govern(claims, (FactNeed("auth", "protocol"),)).relations[0].relation, Relation.DIVERGENT)

    def test_current_and_future_values_are_compatible(self) -> None:
        claims = [
            Claim("auth", "protocol", "JWT", "enum", {"time": "current"}, Modality.OBSERVED, Cardinality.SINGLE, ("e1",)),
            Claim("auth", "protocol", "OAuth2", "enum", {"time": "future"}, Modality.INTENDED, Cardinality.SINGLE, ("e2",)),
        ]
        self.assertEqual(govern(claims, (FactNeed("auth", "protocol"),)).relations[0].relation, Relation.COMPATIBLE)


if __name__ == "__main__":
    unittest.main()
