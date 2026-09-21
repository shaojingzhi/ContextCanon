from __future__ import annotations

import unittest

from experiments.agentabstain.openai_runtime import canonical_tool_name, official_server_env


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


if __name__ == "__main__":
    unittest.main()
