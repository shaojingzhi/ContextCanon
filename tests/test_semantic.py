from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest import mock

from contextcanon.core import TemporalScope
from contextcanon.semantic import (
    ClaimCandidate,
    ExtractionContext,
    LLMStructuredExtractor,
    OpenAICompatibleExtractionClient,
    Provenance,
    normalize_candidate,
)
from experiments.agentabstain.adapter import AgentAbstainAdapter, RuntimeObservation


def _provenance() -> Provenance:
    return Provenance("obs-1", "tool.read", {}, "tool_result", "source excerpt")


class FakeSemanticClient:
    def __init__(self, callback):
        self.callback = callback

    def complete(self, prompt: str, *, model: str):
        return self.callback(prompt, model)


class SemanticExtractionTests(unittest.TestCase):
    def test_dates_normalize_without_forcing_a_semantic_dimension_key(self) -> None:
        responses = iter(
            [
                json.dumps({"claims": [{
                    "entity": "the Spring Gala", "property": "event date",
                    "value": "March 22, 2026", "value_type": "date",
                    "role": "OBSERVED", "confidence": 0.98,
                }]}),
                json.dumps({"claims": [{
                    "entity": "Spring Gala", "property": "event_date",
                    "value": "2026-03-23", "value_type": "date",
                    "role": "OBSERVED", "confidence": 0.97,
                }]}),
            ]
        )

        class FakeObservation:
            success = True
            tool_name = "event.lookup"
            tool_kind = "lookup"
            tool_parameters = {}
            call_index = 0
            tool_result = "unused"

        extractor = LLMStructuredExtractor(
            FakeSemanticClient(lambda prompt, model: next(responses))
        )
        first = extractor.extract(FakeObservation(), ExtractionContext("event date"))
        FakeObservation.call_index = 1
        second = extractor.extract(FakeObservation(), ExtractionContext("event date"))
        self.assertEqual(first[0].value, "2026-03-22")
        self.assertEqual(second[0].value, "2026-03-23")
        self.assertEqual(first[0].entity, second[0].entity)
        self.assertEqual(first[0].property, "event date")
        self.assertEqual(second[0].property, "event_date")
        self.assertNotEqual(first[0].value, second[0].value)

    def test_paraphrased_status_and_entity_are_supported_without_task_names(self) -> None:
        response = json.dumps({"claims": [{
            "entity": "Order W-8855135", "property": "lifecycle status",
            "value": "Delivered", "value_type": "enum",
            "role": "OBSERVED", "confidence": 0.9,
        }]})
        extractor = LLMStructuredExtractor(
            FakeSemanticClient(lambda prompt, model: response)
        )
        observation = type(
            "Observation", (), {
                "success": True, "tool_name": "records.lookup", "tool_kind": "lookup",
                "tool_parameters": {}, "tool_result": "The shipment arrived.", "call_index": 0,
            }
        )()
        claims = extractor.extract(observation)
        self.assertEqual(claims[0].entity, "Order W-8855135")
        self.assertEqual(claims[0].property, "lifecycle status")
        self.assertEqual(claims[0].value, "delivered")

    def test_malformed_or_low_confidence_output_fails_safely(self) -> None:
        observation = type(
            "Observation", (), {
                "success": True, "tool_name": "tool", "tool_kind": "lookup",
                "tool_parameters": {}, "tool_result": "nothing useful", "call_index": 0,
            }
        )()
        malformed = LLMStructuredExtractor(
            FakeSemanticClient(lambda prompt, model: "not json")
        )
        self.assertEqual(malformed.extract(observation), [])
        self.assertEqual(malformed.last_diagnostic, "extraction_error:JSONDecodeError")

        low_confidence = LLMStructuredExtractor(FakeSemanticClient(lambda prompt, model: json.dumps({"claims": [{
            "entity": "x", "property": "status", "value": "open",
            "value_type": "enum", "role": "OBSERVED", "confidence": 0.2,
        }]})))
        self.assertEqual(low_confidence.extract(observation), [])
        self.assertEqual(low_confidence.last_diagnostic, "no_acceptable_claims")

    def test_failed_observation_produces_no_claim(self) -> None:
        observation = type("Observation", (), {"success": False, "tool_name": "tool"})()
        extractor = LLMStructuredExtractor(
            FakeSemanticClient(lambda prompt, model: self.fail("must not call model"))
        )
        self.assertEqual(extractor.extract(observation), [])
        self.assertEqual(extractor.last_diagnostic, "observation_failed")

    def test_client_type_error_does_not_trigger_a_second_request(self) -> None:
        class RaisingClient:
            calls = 0

            def complete(self, prompt: str, *, model: str):
                self.calls += 1
                raise TypeError("internal client bug")

        client = RaisingClient()
        observation = type(
            "Observation", (), {
                "success": True, "tool_name": "tool", "tool_kind": "lookup",
                "tool_parameters": {}, "tool_result": "value", "call_index": 0,
            }
        )()
        extractor = LLMStructuredExtractor(client)
        self.assertEqual(extractor.extract(observation), [])
        self.assertEqual(client.calls, 1)

    def test_prompt_excludes_benchmark_metadata(self) -> None:
        captured: list[str] = []
        extractor = LLMStructuredExtractor(FakeSemanticClient(
            lambda prompt, model: captured.append(prompt) or {"claims": []}
        ))
        observation = type(
            "Observation", (), {
                "success": True, "tool_name": "tool", "tool_kind": "lookup",
                "tool_parameters": {"path": "a.txt", "gold_answer": "secret"},
                "tool_result": {"status": "open", "should_act": True}, "call_index": 0,
            }
        )()
        extractor.extract(observation, ExtractionContext("pair_id=secret"))
        prompt = captured[0].casefold()
        self.assertNotIn("gold_answer", prompt)
        self.assertNotIn("should_act", prompt)
        self.assertNotIn("pair_id=secret", prompt)

    def test_business_metadata_fields_are_not_recursively_deleted(self) -> None:
        captured: list[str] = []
        extractor = LLMStructuredExtractor(FakeSemanticClient(
            lambda prompt, model: captured.append(prompt) or {"claims": []}
        ))
        observation = type(
            "Observation", (), {
                "success": True,
                "tool_name": "catalog.lookup",
                "tool_kind": "lookup",
                "tool_parameters": {},
                "tool_result": {
                    "record": {
                        "category": "maintenance",
                        "metadata": {"owner": "operations"},
                        "task_id": "business-work-42",
                    }
                },
                "call_index": 0,
            }
        )()
        extractor.extract(observation)
        prompt = captured[0]
        self.assertIn('"category": "maintenance"', prompt)
        self.assertIn('"metadata"', prompt)
        self.assertIn('"task_id": "business-work-42"', prompt)

    def test_temporal_scope_is_preserved_as_semantic_metadata(self) -> None:
        extractor = LLMStructuredExtractor(FakeSemanticClient(lambda prompt, model: {"claims": [{
            "entity": "production authentication",
            "property": "target protocol",
            "value": "OAuth2",
            "value_type": "enum",
            "role": "INTENDED",
            "confidence": 0.95,
            "temporal_scope": "future",
        }]}))
        observation = type(
            "Observation", (), {
                "success": True, "tool_name": "adr.read", "tool_kind": "lookup",
                "tool_parameters": {}, "tool_result": "approved target", "call_index": 0,
            }
        )()
        candidate = extractor.extract(observation)[0]
        self.assertEqual(candidate.temporal_scope, TemporalScope.FUTURE.value)

    def test_provenance_is_runtime_supplied(self) -> None:
        extractor = LLMStructuredExtractor(FakeSemanticClient(lambda prompt, model: {"claims": [{
            "entity": "Spring Gala", "property": "event_date", "value": "03/22/2026",
            "value_type": "date", "role": "OBSERVED", "confidence": 0.9,
        }]}))
        observation = type(
            "Observation", (), {
                "success": True, "tool_name": "filesystem.read_file", "tool_kind": "lookup",
                "tool_parameters": {"path": "event-info.txt"},
                "tool_result": "March 22, 2026", "call_index": 4,
            }
        )()
        claim = extractor.extract(observation)[0]
        self.assertEqual(claim.provenance.tool_name, "filesystem.read_file")
        self.assertEqual(claim.provenance.tool_arguments, {"path": "event-info.txt"})
        self.assertEqual(claim.provenance.observation_id.startswith("observation-"), True)

    def test_generic_extractor_is_injected_into_deterministic_ledger(self) -> None:
        class FakeExtractor:
            def extract(self, observation, context=None):
                return [ClaimCandidate(
                    entity="generic entity", property="status", value="open",
                    value_type="enum", role="OBSERVED", confidence=0.9,
                    provenance=_provenance(),
                )]

        adapter = AgentAbstainAdapter(extractor=FakeExtractor())
        adapter.observe_tool_result(RuntimeObservation("arbitrary.tool", "lookup", {}, {"unrelated": True}, True, 0))
        self.assertEqual(
            [(claim.subject, claim.predicate, claim.value) for claim in adapter.ledger.claims],
            [("generic entity", "status", "open")],
        )
        self.assertEqual(adapter.ledger.claims[0].evidence.source_id, "tool.read")

    def test_legacy_ledger_keeps_deterministic_fixture_conflicts(self) -> None:
        class FakeExtractor:
            def extract(self, observation, context=None):
                return [ClaimCandidate(
                    entity="Spring Gala", property="event_date",
                    value=observation.tool_result, value_type="date", role="OBSERVED",
                    confidence=0.9, provenance=_provenance(),
                )]

        adapter = AgentAbstainAdapter(extractor=FakeExtractor())
        adapter.observe_tool_result(RuntimeObservation("tool.a", "lookup", {}, "March 22, 2026", True, 0))
        adapter.observe_tool_result(RuntimeObservation("tool.b", "lookup", {}, "2026-03-23", True, 1))
        conflicts = adapter.ledger.conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].competing_values, ("2026-03-22", "2026-03-23"))

    def test_client_builds_openai_compatible_json_request(self) -> None:
        client = OpenAICompatibleExtractionClient("secret")
        self.assertEqual(client.base_url, "https://api.deepseek.com")
        response = SimpleNamespace(
            json=lambda: {"choices": [{"message": {"content": '{"claims": []}'}}]},
            raise_for_status=lambda: None,
        )

        class FakeTimeout:
            def __init__(self, default, **parts):
                self.default = default
                for name, value in parts.items():
                    setattr(self, name, value)

        class FakeClient:
            init_kwargs = None
            post_kwargs = None

            def __init__(self, **kwargs):
                FakeClient.init_kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, **kwargs):
                FakeClient.post_kwargs = {"url": url, **kwargs}
                return response

        fake_httpx = SimpleNamespace(
            Client=FakeClient,
            Timeout=FakeTimeout,
            HTTPError=RuntimeError,
        )
        with mock.patch("contextcanon.semantic.httpx", fake_httpx):
            client.complete("extract facts", model="deepseek-v4-pro")
        body = json.loads(FakeClient.post_kwargs["content"])
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["temperature"], 0)
        self.assertNotIn("thinking", body)
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual(body["max_tokens"], 256)
        timeout = FakeClient.init_kwargs["timeout"]
        self.assertEqual(timeout.connect, 60.0)
        self.assertEqual(timeout.read, 60.0)
        self.assertFalse(FakeClient.init_kwargs["trust_env"])
        self.assertNotIn("secret", json.dumps(body))

    def test_client_uses_urllib_when_httpx_is_unavailable(self) -> None:
        client = OpenAICompatibleExtractionClient("secret", timeout=12.0)
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": '{"claims": []}'}}]}
        ).encode("utf-8")
        with mock.patch("contextcanon.semantic.httpx", None), mock.patch(
            "contextcanon.semantic.request.urlopen", return_value=response
        ) as urlopen:
            client.complete("extract facts", model="deepseek-v4-pro")
        body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 12.0)
        self.assertNotIn("secret", json.dumps(body))

    def test_client_rejects_non_positive_timeout(self) -> None:
        with self.assertRaises(ValueError):
            OpenAICompatibleExtractionClient("secret", timeout=0)

    def test_normalization_rejects_unsupported_role_and_type(self) -> None:
        candidate = ClaimCandidate("x", "p", "v", "string", "VERIFIED", 0.9, _provenance())
        self.assertIsNone(normalize_candidate(candidate))
        candidate = ClaimCandidate("x", "p", "v", "currency", "OBSERVED", 0.9, _provenance())
        self.assertIsNone(normalize_candidate(candidate))
        candidate = ClaimCandidate(
            "x", "p", "v", "string", "OBSERVED", 0.9, _provenance(),
            observed_at=123,  # type: ignore[arg-type]
        )
        self.assertIsNone(normalize_candidate(candidate))


if __name__ == "__main__":
    unittest.main()
