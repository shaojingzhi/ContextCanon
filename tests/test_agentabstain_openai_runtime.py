from __future__ import annotations

import unittest
from unittest import mock

from contextcanon.tool_semantics import ToolSemantics
from experiments.agentabstain.openai_runtime import (
    AgentAbstainStaticToolSemanticsResolver,
    canonical_tool_name,
    official_server_env,
)
from experiments.agentabstain.official_gate import validate_gate_result
from experiments.agentabstain.run_agent import (
    _build_persisted_result,
    _configure_provider_environment,
    _format_exception,
    _structured_content,
)


class OpenAIRuntimeTests(unittest.TestCase):
    @staticmethod
    def _gate_result(side: str, *, executed_tools: list[str], **diagnostic_overrides: object) -> dict:
        diagnostics = {
            "observations_seen": 1, "claims_created": 1, "commit_attempted": True,
            "conflicts_detected": 0 if side == "act" else 1,
            "guard_decision": "ALLOW" if side == "act" else "REQUIRE_CLARIFICATION",
            "commit_dispatched": side == "act",
        }
        diagnostics.update(diagnostic_overrides)
        return {
            "task": "preview_008", "side": side, "tool_count": 1,
            "encoded_names_roundtrip": True, "exported": True,
            "expected_commit_tool": "phone_and_messages.send_phone_message",
            "executed_tools": executed_tools, "diagnostics": diagnostics,
        }

    def test_encoded_tool_names_decode_to_canonical_names(self) -> None:
        mapping = {"industrial_and_infrastructure_control__event_search": "industrial_and_infrastructure_control.event_search"}
        self.assertEqual(
            canonical_tool_name("industrial_and_infrastructure_control__event_search", mapping),
            "industrial_and_infrastructure_control.event_search",
        )

    def test_agentabstain_static_tool_semantics_are_compatibility_only(self) -> None:
        resolver = AgentAbstainStaticToolSemanticsResolver()
        self.assertEqual(
            resolver.classify("filesystem.read_file"),
            ToolSemantics.READ,
        )
        self.assertEqual(
            resolver.classify("phone_and_messages.send_phone_message"),
            ToolSemantics.SIDE_EFFECT,
        )
        self.assertEqual(
            resolver.classify("production.deploy_config"),
            ToolSemantics.UNKNOWN,
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
        self.assertEqual(validate_gate_result(self._gate_result(
            "act", executed_tools=["phone_and_messages.send_phone_message"])), [])

    def test_gate_validation_rejects_act_log_without_commit(self) -> None:
        failures = validate_gate_result(self._gate_result("act", executed_tools=[]))
        self.assertIn("expected commit tool was not present", "\n".join(failures))

    def test_gate_validation_rejects_act_regressions(self) -> None:
        result = {"task": "preview_008", "side": "act", "tool_count": 0,
                  "encoded_names_roundtrip": False, "exported": False,
                  "expected_commit_tool": "phone_and_messages.send_phone_message", "executed_tools": [],
                  "diagnostics": {"observations_seen": 0, "claims_created": 0,
                                   "commit_attempted": False, "conflicts_detected": 1, "guard_decision": "REQUIRE_CLARIFICATION",
                                   "commit_dispatched": False}}
        failures = validate_gate_result(result)
        self.assertGreaterEqual(len(failures), 7)

    def test_gate_validation_accepts_and_rejects_abstain(self) -> None:
        valid = self._gate_result("abstain", executed_tools=[])
        self.assertEqual(validate_gate_result(valid), [])
        invalid = dict(valid)
        invalid["diagnostics"] = dict(valid["diagnostics"], conflicts_detected=0, commit_dispatched=True)
        self.assertEqual(len(validate_gate_result(invalid)), 2)

    def test_gate_validation_rejects_blocked_commit_in_official_log(self) -> None:
        failures = validate_gate_result(self._gate_result(
            "abstain", executed_tools=["phone_and_messages.send_phone_message"]
        ))
        self.assertIn("blocked commit tool unexpectedly appeared", "\n".join(failures))

    def test_gate_validation_rejects_unattempted_commit(self) -> None:
        failures = validate_gate_result(self._gate_result(
            "abstain", executed_tools=[], commit_attempted=False
        ))
        self.assertIn("commit_attempted must be true", "\n".join(failures))

    def test_runner_exception_format_keeps_stage_without_secret_headers(self) -> None:
        text = _format_exception("model_request", RuntimeError("Connection error"))
        self.assertIn("model_request: RuntimeError: Connection error", text)
        self.assertNotIn("Authorization", text)

    def test_runner_reads_snake_case_mcp_structured_content(self) -> None:
        class Result:
            structured_content = {"state": {}, "execution_log": []}

        self.assertEqual(
            _structured_content(Result()),
            {"state": {}, "execution_log": []},
        )

    def test_deepseek_key_overrides_stale_openai_sdk_key_in_memory(self) -> None:
        import os

        with mock.patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "deepseek-test-key",
                "OPENAI_API_KEY": "stale-proxy-key",
            },
            clear=True,
        ):
            _configure_provider_environment()
            self.assertEqual(os.environ["OPENAI_API_KEY"], "deepseek-test-key")

    def test_provider_metadata_is_passed_to_official_result_builder(self) -> None:
        captured: dict[str, object] = {}

        def builder(**kwargs):
            captured.update(kwargs)
            return object()

        metadata = {
            "model": "deepseek-v4-pro",
            "provider": "openai-compatible",
            "condition": "guard",
            "semantic_extractor": "llm",
            "usage": {"total_tokens": 42},
            "commit_attempted": True,
            "commit_dispatched": False,
        }
        _build_persisted_result(
            builder,
            agent="agent",
            bundle="bundle",
            artifact_dir="artifact",
            final_output="output",
            export_payload={},
            run_error=None,
            provider_metadata=metadata,
        )
        self.assertEqual(captured["provider_metadata"], metadata)


if __name__ == "__main__":
    unittest.main()
