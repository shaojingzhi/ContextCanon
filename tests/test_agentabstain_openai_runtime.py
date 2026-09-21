from __future__ import annotations

import unittest

from experiments.agentabstain.openai_runtime import canonical_tool_name


class OpenAIRuntimeTests(unittest.TestCase):
    def test_encoded_tool_names_decode_to_canonical_names(self) -> None:
        mapping = {"industrial_and_infrastructure_control__event_search": "industrial_and_infrastructure_control.event_search"}
        self.assertEqual(
            canonical_tool_name("industrial_and_infrastructure_control__event_search", mapping),
            "industrial_and_infrastructure_control.event_search",
        )


if __name__ == "__main__":
    unittest.main()
