from __future__ import annotations

from pathlib import Path
import unittest

from experiments import v2_real_benchmark


HOLDOUT = Path(__file__).parents[1] / "benchmarks" / "v2_holdout_cases.yaml"


class V2HoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = v2_real_benchmark.load_cases(HOLDOUT, expected_count=14)

    def test_holdout_shape_and_new_domains(self) -> None:
        self.assertEqual(len(self.cases), 14)
        domains = {case.domain for case in self.cases}
        self.assertIn("healthcare", domains)
        self.assertIn("finance", domains)
        merged = {case.case_id: case for case in self.cases}
        self.assertEqual(merged["H08"].gold_claims[0].evidence_ids, ("E1", "E2"))
        self.assertEqual(merged["H09"].gold_claims[0].evidence_ids, ("E1", "E2"))

    def test_holdout_gold_governance_self_check_passes(self) -> None:
        records = [
            v2_real_benchmark.evaluate_case(case, case.gold_claims)
            for case in self.cases
        ]
        self.assertTrue(all(record["claim_exact_match"] for record in records))
        self.assertTrue(all(record["relation_exact_match"] for record in records))
        self.assertTrue(all(record["summary_match"] for record in records))
        self.assertTrue(all(record["failure_classification"] == "PASS" for record in records))

    def test_frozen_development_benchmark_still_has_36_cases(self) -> None:
        self.assertEqual(len(v2_real_benchmark.load_cases()), 36)


if __name__ == "__main__":
    unittest.main()
