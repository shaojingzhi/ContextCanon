from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest import mock

from contextcanon_v2 import SummaryState
from experiments import v2_ramdocs


DATASET = Path("/tmp/RAMDocs_test.jsonl")


@unittest.skipUnless(DATASET.exists(), "cached RAMDocs dataset is unavailable")
class V2RamDocsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = v2_ramdocs.load_cases(DATASET)

    def test_exact_cache_and_mechanical_gold_counts(self) -> None:
        summary = v2_ramdocs.dataset_summary(self.cases)
        self.assertEqual(summary["case_count"], 500)
        self.assertEqual(summary["document_count"], 2766)
        self.assertEqual(
            summary["document_type_counts"],
            {"correct": 1918, "misinfo": 307, "noise": 541},
        )
        self.assertEqual(
            summary["combination_counts"],
            {
                "correct+misinfo": 241,
                "correct+noise": 334,
                "correct+misinfo+noise": 166,
            },
        )
        self.assertEqual(
            summary["summary_proxy_counts"],
            {"CLEAR": 256, "UNRESOLVED": 241, "INCOMPLETE": 3},
        )
        self.assertEqual(summary["conflict_eval_eligible_count"], 100)

    def test_evidence_and_fact_need_are_mechanical(self) -> None:
        case = self.cases[0]
        self.assertEqual(case.fact_needs[0].subject, case.question)
        self.assertEqual(case.fact_needs[0].dimension, "answer")
        self.assertEqual(case.evidence[0].evidence_id, "E1")
        self.assertEqual(case.evidence[0].location, "documents[0]")
        self.assertEqual(
            case.evidence[0].source_id,
            "ramdocs/row-0000/document-00",
        )

    def test_direct_prompt_has_equal_metadata_without_gold_leakage(self) -> None:
        prompt = v2_ramdocs.direct_prompt(self.cases[0])
        self.assertIn('"ref":"E1"', prompt)
        self.assertIn('"source":"ramdocs/row-0000/document-00"', prompt)
        self.assertIn('"location":"documents[0]"', prompt)
        self.assertIn(self.cases[0].question, prompt)
        self.assertNotIn('"type"', prompt)
        self.assertNotIn("gold_answers", prompt)
        self.assertNotIn("wrong_answers", prompt)

    def test_direct_adapter_calls_completion_once(self) -> None:
        calls = []

        def complete(prompt: str) -> str:
            calls.append(prompt)
            return json.dumps({
                "relevant_evidence_refs": ["E1"],
                "competing_evidence": False,
                "summary_state": "CLEAR",
            })

        prediction = v2_ramdocs.direct_prediction(self.cases[0], complete)
        self.assertEqual(len(calls), 1)
        self.assertEqual(prediction.relevant_evidence_ids, frozenset({"E1"}))

    def test_contextcanon_relevance_comes_from_claim_provenance(self) -> None:
        case = self.cases[0]
        calls = []

        def complete(prompt: str) -> str:
            calls.append(prompt)
            return json.dumps({
                "claims": [{
                    "subject": case.question,
                    "predicate": "answer",
                    "value": "3,559 people",
                    "value_type": "string",
                    "scope": {},
                    "modality": "DOCUMENTED",
                    "cardinality": "UNKNOWN",
                    "evidence_refs": ["E1", "E2"],
                }]
            })

        prediction = v2_ramdocs.contextcanon_prediction(case, complete)
        self.assertEqual(len(calls), 1)
        self.assertIn(case.question, calls[0])
        self.assertIn('"source":"ramdocs/row-0000/document-00"', calls[0])
        self.assertIn('"location":"documents[0]"', calls[0])
        self.assertNotIn('"type"', calls[0])
        self.assertNotIn("gold_answers", calls[0])
        self.assertNotIn("wrong_answers", calls[0])
        self.assertEqual(prediction.relevant_evidence_ids, frozenset({"E1", "E2"}))
        self.assertFalse(prediction.competing_evidence)
        self.assertEqual(prediction.summary_state, SummaryState.CLEAR)

    def test_perfect_predictions_score_every_metric(self) -> None:
        predictions = {
            case.row_index: v2_ramdocs.RamDocsPrediction(
                relevant_evidence_ids=case.gold_relevant_evidence_ids,
                competing_evidence=case.gold_competing_evidence,
                summary_state=case.gold_summary_proxy,
            )
            for case in self.cases
        }
        metrics = v2_ramdocs.score_predictions(self.cases, predictions)
        relevance = metrics["evidence_relevance"]
        self.assertEqual(relevance["case_count"], 500)
        self.assertTrue(all(
            relevance[name]["rate"] == 1.0
            for name in (
                "relevant_precision",
                "relevant_recall",
                "correct_recall",
                "misinfo_retention_rate",
                "noise_rejection_rate",
            )
        ))
        self.assertEqual(metrics["competing_evidence"]["eligible_case_count"], 100)
        self.assertEqual(metrics["competing_evidence"]["f1"], 1.0)
        all_summary = metrics["summary_proxy"]["all_cases_exploratory"]
        eligible_summary = metrics["summary_proxy"]["eligible_subset"]
        self.assertEqual(all_summary["accuracy"]["rate"], 1.0)
        self.assertEqual(eligible_summary["accuracy"]["rate"], 1.0)
        self.assertEqual(
            all_summary["per_class_counts"]["INCOMPLETE"]["gold"],
            3,
        )
        self.assertEqual(
            eligible_summary["per_class_counts"],
            {
                "CLEAR": {"gold": 43, "predicted": 43, "correct": 43},
                "UNRESOLVED": {"gold": 54, "predicted": 54, "correct": 54},
                "INCOMPLETE": {"gold": 3, "predicted": 3, "correct": 3},
            },
        )

    def test_ineligible_entity_ambiguity_does_not_affect_conflict_metrics(self) -> None:
        predictions = {
            case.row_index: v2_ramdocs.RamDocsPrediction(
                relevant_evidence_ids=case.gold_relevant_evidence_ids,
                competing_evidence=(
                    case.gold_competing_evidence
                    if case.conflict_eval_eligible else not case.gold_competing_evidence
                ),
                summary_state=(
                    case.gold_summary_proxy
                    if case.conflict_eval_eligible else SummaryState.INCOMPLETE
                ),
            )
            for case in self.cases
        }
        metrics = v2_ramdocs.score_predictions(self.cases, predictions)
        self.assertEqual(metrics["competing_evidence"]["eligible_case_count"], 100)
        self.assertEqual(metrics["competing_evidence"]["accuracy"]["rate"], 1.0)
        self.assertEqual(
            metrics["summary_proxy"]["eligible_subset"]["accuracy"]["rate"],
            1.0,
        )
        self.assertLess(
            metrics["summary_proxy"]["all_cases_exploratory"]["accuracy"]["rate"],
            1.0,
        )

    def test_empty_predictions_do_not_report_undefined_precision_as_perfect(self) -> None:
        predictions = {
            case.row_index: v2_ramdocs.RamDocsPrediction(
                relevant_evidence_ids=frozenset(),
                competing_evidence=False,
                summary_state=SummaryState.INCOMPLETE,
            )
            for case in self.cases
        }
        metrics = v2_ramdocs.score_predictions(self.cases, predictions)
        relevance = metrics["evidence_relevance"]
        self.assertEqual(relevance["relevant_precision"]["rate"], 0.0)
        self.assertEqual(relevance["relevant_recall"]["rate"], 0.0)
        self.assertEqual(relevance["noise_rejection_rate"]["rate"], 1.0)
        self.assertEqual(metrics["competing_evidence"]["precision"]["rate"], 0.0)
        self.assertEqual(metrics["competing_evidence"]["f1"], 0.0)

    def test_default_entrypoint_is_offline(self) -> None:
        output = io.StringIO()
        with mock.patch.object(
            v2_ramdocs,
            "OpenAICompatibleExtractionClient",
            side_effect=AssertionError("provider constructed"),
        ):
            with redirect_stdout(output):
                status = v2_ramdocs.main(["--dataset", str(DATASET)])
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertFalse(payload["live_execution"])
        self.assertEqual(payload["api_calls"], 0)

    def test_limit_selects_first_stable_rows(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = v2_ramdocs.main([
                "--dataset", str(DATASET), "--limit", "20"
            ])
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(payload["case_count"], 20)
        self.assertEqual([case.row_index for case in self.cases[:20]], list(range(20)))

    def test_live_paths_use_exactly_one_completion_per_case(self) -> None:
        cases = self.cases[:2]

        class Client:
            def __init__(self) -> None:
                self.prompts = []

            def complete(self, prompt: str, *, model: str):
                self.prompts.append((prompt, model))
                if prompt.startswith("Compile only semantic Claims"):
                    question = cases[len(self.prompts) - 1].question
                    content = json.dumps({
                        "claims": [{
                            "subject": question,
                            "predicate": "answer",
                            "value": "value",
                            "value_type": "string",
                            "scope": {},
                            "modality": "DOCUMENTED",
                            "cardinality": "UNKNOWN",
                            "evidence_refs": ["E1"],
                        }]
                    })
                else:
                    content = json.dumps({
                        "relevant_evidence_refs": ["E1"],
                        "competing_evidence": False,
                        "summary_state": "CLEAR",
                    })
                return mock.Mock(content=content)

        for system in ("contextcanon", "direct"):
            client = Client()
            with redirect_stdout(io.StringIO()):
                records, aggregate = v2_ramdocs.run_live(
                    cases, client, system=system, model="test-model"
                )
            self.assertEqual(len(client.prompts), 2)
            self.assertEqual(len(records), 2)
            self.assertEqual(aggregate["provider_call_count"], 2)
            self.assertEqual(aggregate["completed_case_count"], 2)
            self.assertEqual(aggregate["failed_case_count"], 0)
            self.assertEqual(aggregate["completion_rate"]["rate"], 1.0)

    def test_failed_completion_is_not_retried(self) -> None:
        class FailingClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, prompt: str, *, model: str):
                self.calls += 1
                raise TimeoutError("offline test timeout")

        client = FailingClient()
        with redirect_stdout(io.StringIO()):
            records, aggregate = v2_ramdocs.run_live(
                self.cases[:1], client, system="contextcanon", model="test-model"
            )
        self.assertEqual(client.calls, 1)
        self.assertEqual(aggregate["provider_call_count"], 1)
        self.assertEqual(aggregate["completed_case_count"], 0)
        self.assertEqual(aggregate["failed_case_count"], 1)
        self.assertEqual(aggregate["completion_rate"]["rate"], 0.0)
        self.assertIn("SemanticCompilationError", records[0]["error"])

    def test_failed_case_counts_against_end_to_end_denominators(self) -> None:
        cases = self.cases[:2]
        self.assertTrue(all(case.conflict_eval_eligible for case in cases))

        class PartiallyFailingClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, prompt: str, *, model: str):
                self.calls += 1
                if self.calls == 2:
                    raise TimeoutError("offline test timeout")
                case = cases[0]
                return mock.Mock(content=json.dumps({
                    "relevant_evidence_refs": sorted(case.gold_relevant_evidence_ids),
                    "competing_evidence": case.gold_competing_evidence,
                    "summary_state": case.gold_summary_proxy.value,
                }))

        client = PartiallyFailingClient()
        with redirect_stdout(io.StringIO()):
            records, aggregate = v2_ramdocs.run_live(
                cases, client, system="direct", model="test-model"
            )

        self.assertEqual(client.calls, 2)
        self.assertEqual(aggregate["provider_call_count"], 2)
        self.assertEqual(aggregate["attempted_case_count"], 2)
        self.assertEqual(aggregate["completed_case_count"], 1)
        self.assertEqual(aggregate["failed_case_count"], 1)
        self.assertEqual(aggregate["completion_rate"]["rate"], 0.5)
        self.assertEqual(aggregate["completed_only_metrics"]["case_count"], 1)
        self.assertEqual(
            aggregate["completed_only_metrics"]["evidence_relevance"]
            ["relevant_recall"]["rate"],
            1.0,
        )
        end_to_end = aggregate["end_to_end_metrics"]
        self.assertEqual(
            end_to_end["competing_evidence_eligible_accuracy"],
            {"count": 1, "total": 2, "rate": 0.5},
        )
        self.assertEqual(
            end_to_end["summary_proxy_all_attempted_accuracy"]["rate"], 0.5
        )
        self.assertEqual(
            end_to_end["summary_proxy_eligible_attempted_accuracy"]["rate"], 0.5
        )
        self.assertIsNotNone(records[1]["error"])


if __name__ == "__main__":
    unittest.main()
