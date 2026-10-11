from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from contextcanon.semantic import SemanticCompletion
from contextcanon_v2 import Cardinality, Claim, Modality, SemanticCompiler
from experiments import v2_ambigdocs as ambigdocs
from experiments.v2_ambigdocs import (
    MAX_OUTPUT_TOKENS,
    MODEL,
    TIMEOUT_SECONDS,
    build_client,
    direct_prompt,
    load_sample_manifest,
    main,
    map_row,
    parse_direct_response,
    project_contextcanon_claims,
    run_live,
    sample_qids,
    score_predictions,
)


def row(qid: int = 7) -> dict:
    return {
        "qid": qid,
        "ambiguous_entity": "Mercury",
        "question": "What is Mercury's composition?",
        "documents": [
            {"pid": "p1", "title": "Mercury (planet)", "text": "planet text", "answer": "rocky"},
            {"pid": "p2", "title": "Mercury (element)", "text": "element text", "answer": "liquid metal"},
        ],
    }


def claim_response(case) -> str:
    return json.dumps({
        "claims": [
            {
                "subject": target.entity,
                "predicate": case.question,
                "value": target.answer,
                "value_type": "string",
                "scope": {},
                "modality": "DOCUMENTED",
                "cardinality": "SINGLE",
                "evidence_refs": [target.evidence_id],
            }
            for target in case.gold
        ]
    })


def direct_response(case) -> str:
    return json.dumps({
        "associations": [
            {"entity": target.entity, "answer": target.answer, "evidence_ref": target.evidence_id}
            for target in case.gold
        ]
    })


class FakeClient:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, prompt: str, *, model: str) -> SemanticCompletion:
        self.calls.append((prompt, model))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SemanticCompletion(
            response,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            model=model,
        )


class AmbigDocsTests(unittest.TestCase):
    def test_row_maps_evidence_entities_and_pid(self) -> None:
        case = map_row(row())
        self.assertEqual([item.evidence_id for item in case.evidence], ["E1", "E2"])
        self.assertEqual(case.evidence[0].source_id, "ambigdocs:7:p1")
        self.assertEqual(case.evidence[0].location, "Mercury (planet)")
        self.assertEqual([need.subject for need in case.fact_needs], ["Mercury (planet)", "Mercury (element)"])
        self.assertEqual(
            [need.dimension for need in case.fact_needs],
            [row()["question"], row()["question"]],
        )
        self.assertEqual(case.gold[1].pid, "p2")

    def test_compiler_receives_original_question_through_fact_needs(self) -> None:
        case = map_row(row())
        prompts: list[str] = []
        compiler = SemanticCompiler(lambda prompt: prompts.append(prompt) or '{"claims":[]}')
        compiler.compile(case.evidence, case.fact_needs)
        self.assertEqual(len(prompts), 1)
        self.assertIn(row()["question"], prompts[0])

    def test_direct_prompt_has_same_target_and_no_gold_answers(self) -> None:
        prompt = direct_prompt(map_row(row()))
        self.assertIn(row()["question"], prompt)
        self.assertIn("Mercury (planet)", prompt)
        self.assertIn("p1", prompt)
        self.assertNotIn("Ambiguous entity:", prompt)
        self.assertNotIn("rocky", prompt)
        self.assertNotIn("liquid metal", prompt)

    def test_direct_schema_is_frozen_and_resolves_pid(self) -> None:
        case = map_row(row())
        parsed = parse_direct_response(
            '{"associations":[{"entity":"Mercury (planet)","answer":"rocky","evidence_ref":"E1"}]}',
            case,
        )
        self.assertEqual(parsed, ({"entity": "Mercury (planet)", "answer": "rocky", "pid": "p1"},))

    def test_contextcanon_claims_project_to_the_same_pid_identity(self) -> None:
        case = map_row(row())
        claim = Claim(
            subject="Mercury (planet)",
            predicate="answer",
            value="rocky",
            value_type="string",
            scope={},
            modality=Modality.OBSERVED,
            cardinality=Cardinality.SINGLE,
            evidence_ids=("E1",),
        )
        self.assertEqual(
            project_contextcanon_claims(case, (claim,)),
            ({"entity": "Mercury (planet)", "answer": "rocky", "pid": "p1"},),
        )

    def test_sample_is_deterministic_and_manifest_is_frozen(self) -> None:
        qids = [9, 2, 7, 4, 1, 8, 3, 6, 5]
        self.assertEqual(sample_qids(qids, size=4, seed=42), sample_qids(qids, size=4, seed=42))
        manifest = load_sample_manifest()
        self.assertEqual(len(manifest), 100)
        self.assertEqual(len(set(manifest)), 100)

    def test_scorer_perfect_and_wrong_predictions(self) -> None:
        case = map_row(row())
        perfect = {
            case.qid: {
                "success": True,
                "associations": [
                    {"subject": target.entity, "value": target.answer, "evidence_refs": [target.evidence_id]}
                    for target in case.gold
                ],
            }
        }
        perfect[case.qid]["associations"] = [
            {"entity": target.entity, "answer": target.answer, "pid": target.pid}
            for target in case.gold
        ]
        metrics = score_predictions((case,), perfect)
        self.assertEqual(metrics["completed_only_metrics"]["normalized_exact_answer_association"]["rate"], 1.0)
        self.assertEqual(metrics["end_to_end_metrics"]["answer_token_f1"]["rate"], 1.0)
        wrong = {
            case.qid: {
                "success": True,
                "associations": [{"entity": "Mercury (planet)", "answer": "liquid metal", "pid": "p2"}],
            }
        }
        wrong_metrics = score_predictions((case,), wrong)
        self.assertEqual(wrong_metrics["end_to_end_metrics"]["normalized_exact_answer_association"]["rate"], 0.0)

    def test_failed_case_counts_in_end_to_end_but_not_completed_only(self) -> None:
        case = map_row(row())
        predictions = {
            case.qid: {"success": False, "associations": []},
        }
        metrics = score_predictions((case,), predictions)
        self.assertEqual(metrics["schema_success_rate"]["rate"], 0.0)
        self.assertEqual(metrics["completed_only_metrics"]["case_count"], 0)
        self.assertEqual(metrics["end_to_end_metrics"]["normalized_exact_answer_association"]["rate"], 0.0)

    def test_frozen_client_configuration_is_non_thinking_flash(self) -> None:
        client = build_client("not-a-real-key", "https://api.deepseek.com")
        self.assertEqual(MODEL, "deepseek-flash")
        self.assertEqual(client.timeout, TIMEOUT_SECONDS)
        self.assertEqual(client.inference.max_output_tokens, MAX_OUTPUT_TOKENS)
        self.assertFalse(client.inference.thinking)
        self.assertIsNone(client.inference.reasoning_effort)

    def test_live_runner_interleaves_one_call_per_case_and_uses_frozen_scorer(self) -> None:
        cases = (map_row(row(7)), map_row(row(8)))
        client = FakeClient((
            claim_response(cases[0]),
            direct_response(cases[0]),
            direct_response(cases[1]),
            claim_response(cases[1]),
        ))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(ambigdocs, "score_predictions", wraps=score_predictions) as scorer:
                with redirect_stdout(io.StringIO()):
                    aggregate = run_live(
                        cases,
                        client,
                        contextcanon_path=root / "contextcanon.jsonl",
                        direct_path=root / "direct.jsonl",
                        aggregate_path=root / "aggregate.json",
                    )
            self.assertEqual(scorer.call_count, 2)
            self.assertEqual(len((root / "contextcanon.jsonl").read_text().splitlines()), 2)
            self.assertEqual(len((root / "direct.jsonl").read_text().splitlines()), 2)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual([model for _prompt, model in client.calls], [MODEL] * 4)
        self.assertEqual(
            ["Return only JSON matching" in prompt for prompt, _model in client.calls],
            [True, False, False, True],
        )
        self.assertEqual(aggregate["total_provider_calls"], 4)
        self.assertEqual(aggregate["systems"]["contextcanon"]["successful_cases"], 2)
        self.assertEqual(aggregate["systems"]["direct"]["successful_cases"], 2)
        self.assertEqual(
            aggregate["systems"]["contextcanon"]["metrics"]["end_to_end_metrics"]
            ["normalized_exact_answer_association"]["rate"],
            1.0,
        )

    def test_failed_completion_is_recorded_once_without_retry(self) -> None:
        case = map_row(row())
        client = FakeClient((TimeoutError("offline test timeout"), direct_response(case)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with redirect_stdout(io.StringIO()):
                aggregate = run_live(
                    (case,),
                    client,
                    contextcanon_path=root / "contextcanon.jsonl",
                    direct_path=root / "direct.jsonl",
                    aggregate_path=root / "aggregate.json",
                )
            failed = json.loads((root / "contextcanon.jsonl").read_text().strip())
        self.assertEqual(len(client.calls), 2)
        self.assertFalse(failed["success"])
        self.assertEqual(failed["associations"], [])
        self.assertEqual(aggregate["systems"]["contextcanon"]["provider_calls"], 1)
        self.assertEqual(aggregate["systems"]["contextcanon"]["failed_cases"], 1)

    def test_default_invocation_makes_zero_api_calls(self) -> None:
        cases = tuple(map_row(row(qid)) for qid in load_sample_manifest())
        with mock.patch.object(ambigdocs, "load_cases", return_value=cases):
            with mock.patch.object(ambigdocs, "build_client") as client_builder:
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(main([]), 0)
        client_builder.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["api_calls"], 0)


if __name__ == "__main__":
    unittest.main()
