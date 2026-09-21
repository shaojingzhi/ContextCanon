from __future__ import annotations

import unittest

from experiments.agentabstain.openai_runtime import canonical_tool_name, official_server_env
from experiments.agentabstain.official_gate import validate_gate_result


class OpenAIRuntimeTests(unittest.TestCase):
    def test_encoded_tool_names_decode_to_canonical_names(self) -> None:
        mapping = {"industrial_and_infrastructure_control__event_search": "industrial_and_infrastructure_control.event_search"}
        self.assertEqual(
            canonical_tool_name("industrial_and_infrastructure_control__event_search", mapping),
            "industrial_and_infrastructure_control.event_search",
        )

    def test_official_server_env_is_small_and_deterministic(self) -> None:
        import os
        old = os.environ.get("AGENTABSTAIN_DATA")
        try:
            os.environ["AGENTABSTAIN_DATA"] = "/tmp/agentabstain-data"
            env = official_server_env()
            self.assertEqual(env["AGENTABSTAIN_DATA"], "/tmp/agentabstain-data")
            self.assertNotIn("DEEPSEEK_API_KEY", env)
            self.assertEqual(env, official_server_env())
        finally:
            if old is None:
                os.environ.pop("AGENTABSTAIN_DATA", None)
            else:
                os.environ["AGENTABSTAIN_DATA"] = old

    def test_gate_validation_accepts_valid_act(self) -> None:
        self.assertEqual(validate_gate_result({
            "task": "preview_008", "side": "act", "tool_count": 1,
            "encoded_names_roundtrip": True, "exported": True,
            "diagnostics": {"observations_seen": 1, "claims_created": 1,
                             "conflicts_detected": 0, "guard_decision": "ALLOW",
                             "commit_dispatched": True},
        }), [])

    def test_gate_validation_rejects_act_regressions(self) -> None:
        result = {"task": "preview_008", "side": "act", "tool_count": 0,
                  "encoded_names_roundtrip": False, "exported": False,
                  "diagnostics": {"observations_seen": 0, "claims_created": 0,
                                   "conflicts_detected": 1, "guard_decision": "REQUIRE_CLARIFICATION",
                                   "commit_dispatched": False}}
        failures = validate_gate_result(result)
        self.assertGreaterEqual(len(failures), 7)

    def test_gate_validation_accepts_and_rejects_abstain(self) -> None:
        valid = {"task": "preview_008", "side": "abstain", "tool_count": 1,
                 "encoded_names_roundtrip": True, "exported": True,
                 "diagnostics": {"observations_seen": 1, "claims_created": 1,
                                  "conflicts_detected": 1, "guard_decision": "REQUIRE_CLARIFICATION",
                                  "commit_dispatched": False}}
        self.assertEqual(validate_gate_result(valid), [])
        invalid = dict(valid)
        invalid["diagnostics"] = dict(valid["diagnostics"], conflicts_detected=0, commit_dispatched=True)
        self.assertEqual(len(validate_gate_result(invalid)), 2)


if __name__ == "__main__":
    unittest.main()
