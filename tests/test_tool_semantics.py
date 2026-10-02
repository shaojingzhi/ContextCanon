from __future__ import annotations

import unittest

from contextcanon.tool_semantics import LLMToolSemanticsResolver, ToolSemantics


class FakeSemanticClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, model: str):
        self.prompts.append(prompt)
        return next(self.responses)


class ToolSemanticsTests(unittest.TestCase):
    def test_explicit_read_only_annotation_has_priority(self) -> None:
        client = FakeSemanticClient([])
        resolver = LLMToolSemanticsResolver(client)
        result = resolver.classify(
            "records.lookup",
            annotations={"readOnlyHint": True},
        )
        self.assertEqual(result, ToolSemantics.READ)
        self.assertEqual(client.prompts, [])

    def test_generic_side_effect_can_be_classified_without_static_name_map(self) -> None:
        client = FakeSemanticClient([{
            "semantics": "SIDE_EFFECT",
            "confidence": 0.96,
        }])
        resolver = LLMToolSemanticsResolver(client)
        result = resolver.classify(
            "production.deploy_config",
            description="Deploy a configuration to production.",
            input_schema={"type": "object"},
            annotations={"gold_answer": "must-not-leak"},
        )
        self.assertEqual(result, ToolSemantics.SIDE_EFFECT)
        self.assertIn("production.deploy_config", client.prompts[0])
        self.assertNotIn("gold_answer", client.prompts[0])
        self.assertNotIn("must-not-leak", client.prompts[0])

    def test_low_confidence_and_malformed_results_are_unknown(self) -> None:
        low_confidence = LLMToolSemanticsResolver(FakeSemanticClient([{
            "semantics": "SIDE_EFFECT",
            "confidence": 0.3,
        }]))
        malformed = LLMToolSemanticsResolver(FakeSemanticClient(["not-json"]))
        self.assertEqual(
            low_confidence.classify("production.deploy_config"),
            ToolSemantics.UNKNOWN,
        )
        self.assertEqual(
            malformed.classify("production.deploy_config"),
            ToolSemantics.UNKNOWN,
        )

    def test_discovery_prepare_resolves_each_metadata_identity_once(self) -> None:
        client = FakeSemanticClient([
            {"semantics": "READ", "confidence": 0.95},
            {"semantics": "SIDE_EFFECT", "confidence": 0.95},
        ])
        resolver = LLMToolSemanticsResolver(client)
        tools = {
            "records.read": {
                "description": "Read records",
                "input_schema": {"type": "object"},
                "annotations": {"readOnlyHint": False},
            },
            "records.write": {
                "description": "Write records",
                "input_schema": {"type": "object"},
                "annotations": {"destructiveHint": True},
            },
        }
        resolver.prepare(tools)
        resolver.prepare(tools)
        self.assertEqual(len(client.prompts), 1)
        self.assertEqual(resolver.cache_hits, 1)
        self.assertEqual(
            resolver.classify(
                "records.read",
                description="Read records",
                input_schema={"type": "object"},
                annotations={"readOnlyHint": False},
            ),
            ToolSemantics.READ,
        )


if __name__ == "__main__":
    unittest.main()
