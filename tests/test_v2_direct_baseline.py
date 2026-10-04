from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import unittest
from unittest import mock

from experiments import v2_direct_baseline


class V2DirectBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = v2_direct_baseline.load_cases()
        cls.by_id = {case.case_id: case for case in cls.cases}

    def test_default_mode_loads_36_cases_without_provider(self) -> None:
        output = io.StringIO()
        with mock.patch.object(
            v2_direct_baseline,
            "OpenAICompatibleExtractionClient",
            side_effect=AssertionError("provider constructed"),
        ):
            with redirect_stdout(output):
                status = v2_direct_baseline.main([])
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(payload["case_count"], 36)
        self.assertFalse(payload["live_execution"])

    def test_prompt_contains_only_runtime_input_and_general_contract(self) -> None:
        case = self.by_id["A1"]
        prompt = v2_direct_baseline.build_prompt(case)
        self.assertIn("date=2026-03-22", prompt)
        self.assertIn('"source":"event-record"', prompt)
        self.assertIn('"location":"date"', prompt)
        self.assertIn('"subject":"Spring Gala"', prompt)
        self.assertIn("Allowed relations", prompt)
        self.assertIn("DOCUMENTED participates", prompt)
        self.assertIn("MULTI + MULTI", prompt)
        self.assertIn("UNKNOWN + anything", prompt)
        self.assertNotIn("gold_claims", prompt)
        self.assertNotIn("gold_summary", prompt)
        self.assertNotIn("same current single-valued event date differs", prompt)

    def test_relation_and_endpoint_order_do_not_affect_exact_match(self) -> None:
        case = self.by_id["P1"]
        raw = json.dumps({
            "relations": [
                {
                    "left_evidence_refs": ["E4"],
                    "right_evidence_refs": ["E3"],
                    "relation": "CONFLICTING",
                },
                {
                    "left_evidence_refs": ["E2"],
                    "right_evidence_refs": ["E1"],
                    "relation": "CONFLICTING",
                },
            ],
            "summary_state": "UNRESOLVED",
        })
        record = v2_direct_baseline.evaluate_response(case, raw, 1.0)
        self.assertTrue(record["relation_exact_match"])
        self.assertTrue(record["summary_match"])
        self.assertTrue(record["decision_exact_match"])
        self.assertIsNone(record["schema_error"])

    def test_unknown_evidence_ref_is_a_schema_failure(self) -> None:
        case = self.by_id["A1"]
        raw = json.dumps({
            "relations": [{
                "left_evidence_refs": ["E1"],
                "right_evidence_refs": ["E99"],
                "relation": "CONFLICTING",
            }],
            "summary_state": "UNRESOLVED",
        })
        record = v2_direct_baseline.evaluate_response(case, raw, 1.0)
        self.assertEqual(record["schema_error"], "relation references unknown Evidence")
        self.assertFalse(record["decision_exact_match"])


if __name__ == "__main__":
    unittest.main()
