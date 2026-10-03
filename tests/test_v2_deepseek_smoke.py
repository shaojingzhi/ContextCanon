from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest import mock

from contextcanon_v2 import SummaryState
from experiments import v2_deepseek_smoke


class V2DeepSeekSmokeTests(unittest.TestCase):
    def test_five_cases_are_constructed_without_network(self) -> None:
        with mock.patch.object(v2_deepseek_smoke, "OpenAICompatibleExtractionClient", side_effect=AssertionError("network")):
            cases = v2_deepseek_smoke.build_cases()
        self.assertEqual(len(cases), 5)
        self.assertEqual(
            [case.name for case in cases],
            [
                "scalar_current_conflict",
                "current_future_compatible",
                "observed_required_divergence",
                "multi_supported_regions",
                "semantic_normalization",
            ],
        )

    def test_default_main_mode_makes_no_api_call(self) -> None:
        output = io.StringIO()
        with mock.patch.object(v2_deepseek_smoke, "OpenAICompatibleExtractionClient", side_effect=AssertionError("network")):
            with redirect_stdout(output):
                status = v2_deepseek_smoke.main([])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["live_execution"], False)

    def test_live_path_uses_one_compiler_and_governance_pass_per_case_offline(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, _prompt: str, *, model: str) -> SimpleNamespace:
                self.calls += 1
                return SimpleNamespace(content=json.dumps({"claims": []}))

        client = FakeClient()
        output = io.StringIO()
        with redirect_stdout(output):
            records = v2_deepseek_smoke._run_live(v2_deepseek_smoke.build_cases(), client, "offline-test")
        self.assertEqual(client.calls, 5)
        self.assertEqual(len(records), 5)
        self.assertTrue(all(record["summary_state"] == SummaryState.INCOMPLETE.value for record in records))


if __name__ == "__main__":
    unittest.main()
