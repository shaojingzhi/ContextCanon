from __future__ import annotations

from dataclasses import replace
import io
import json
from contextlib import redirect_stdout
import unittest
from unittest import mock

from contextcanon_v2 import Cardinality, GovernanceResult, SummaryState
from experiments import v2_real_benchmark


class V2RealBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = v2_real_benchmark.load_cases()
        cls.by_id = {case.case_id: case for case in cls.cases}

    def test_all_36_gold_cases_pass_evaluator_self_check(self) -> None:
        records = [
            v2_real_benchmark.evaluate_case(case, case.gold_claims)
            for case in self.cases
        ]
        self.assertEqual(len(records), 36)
        self.assertTrue(all(record["claim_exact_match"] for record in records))
        self.assertTrue(all(record["relation_exact_match"] for record in records))
        self.assertTrue(all(record["summary_match"] for record in records))
        self.assertTrue(all(record["failure_classification"] == "PASS" for record in records))
        metrics = v2_real_benchmark.aggregate_metrics(records)
        self.assertEqual(metrics["case_claim_exact_accuracy"]["rate"], 1.0)
        self.assertEqual(metrics["full_end_to_end_exact_accuracy"]["rate"], 1.0)

    def test_wrong_cardinality_with_correct_governance_is_localized(self) -> None:
        case = self.by_id["E2"]
        predicted = tuple(replace(claim, cardinality=Cardinality.MULTI) for claim in case.gold_claims)
        record = v2_real_benchmark.evaluate_case(case, predicted)
        self.assertFalse(record["cardinality_match"])
        self.assertTrue(record["governance_exact_match"])
        self.assertEqual(record["failure_classification"], "CLAIM_WRONG_GOVERNANCE_RIGHT")

    def test_wrong_scope_with_correct_governance_is_localized(self) -> None:
        case = self.by_id["E2"]
        predicted = (replace(case.gold_claims[0], scope={"region": "CA"}), case.gold_claims[1])
        record = v2_real_benchmark.evaluate_case(case, predicted)
        self.assertFalse(record["scope_match"])
        self.assertTrue(record["governance_exact_match"])
        self.assertEqual(record["failure_classification"], "CLAIM_WRONG_GOVERNANCE_RIGHT")

    def test_correct_claims_with_wrong_governance_is_localized(self) -> None:
        case = self.by_id["E2"]
        wrong = GovernanceResult(case.gold_claims, (), SummaryState.INCOMPLETE)
        record = v2_real_benchmark.evaluate_case(
            case, case.gold_claims, governance_result=wrong
        )
        self.assertTrue(record["claim_exact_match"])
        self.assertFalse(record["governance_exact_match"])
        self.assertEqual(record["failure_classification"], "DETERMINISTIC_GOVERNANCE_ERROR")

    def test_missing_claim_is_claim_semantics_error(self) -> None:
        case = self.by_id["A1"]
        record = v2_real_benchmark.evaluate_case(case, case.gold_claims[:1])
        self.assertEqual(record["predicted_claim_count"], 1)
        self.assertFalse(record["claim_exact_match"])
        self.assertEqual(record["failure_classification"], "CLAIM_SEMANTICS_ERROR")

    def test_default_mode_makes_no_provider_call(self) -> None:
        output = io.StringIO()
        with mock.patch.object(
            v2_real_benchmark,
            "OpenAICompatibleExtractionClient",
            side_effect=AssertionError("provider constructed"),
        ):
            with redirect_stdout(output):
                status = v2_real_benchmark.main([])
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(payload["case_count"], 36)
        self.assertEqual(payload["gold_self_check_passes"], 36)
        self.assertFalse(payload["live_execution"])


if __name__ == "__main__":
    unittest.main()
