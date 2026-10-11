from __future__ import annotations

import unittest

from contextcanon_v2 import Cardinality, Claim, Modality, SemanticCompiler
from experiments.v2_ambigdocs import (
    direct_prompt,
    load_sample_manifest,
    map_row,
    parse_direct_response,
    project_contextcanon_claims,
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


if __name__ == "__main__":
    unittest.main()
