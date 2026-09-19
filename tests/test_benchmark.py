import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from benchmarks.build_prompts import (
    SHARED_INSTRUCTIONS,
    STABILITY_CASE_IDS,
    build_prompt,
    export_prompts,
    load_cases,
)
from benchmarks.report import render_report
from benchmarks.scorer import (
    main as scorer_main,
    score_case,
    stability_rate,
    stability_variants,
    summarize,
    validate_prediction,
)
from contextcanon.assembly import assemble_context
from contextcanon.extraction import extract_demo_claims
from contextcanon.resolution import DefaultResolutionPolicy
from contextcanon.sources import load_sources
from contextcanon.verification import verify_claims


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_cases()

    def test_all_fifteen_cases_load_with_unique_ids(self) -> None:
        self.assertEqual(len(self.cases), 15)
        self.assertEqual(len({case.id for case in self.cases}), 15)
        for case in self.cases:
            truth = case.ground_truth
            self.assertIn(truth["status"], {"RESOLVED", "DIVERGED", "AMBIGUOUS", "UNVERIFIED"})
            self.assertIn(truth["current"], {"JWT", "OAuth2", None})
            self.assertIn(truth["target"], {"JWT", "OAuth2", None})

    def test_ground_truth_sources_exist_in_their_case(self) -> None:
        for case in self.cases:
            truth = case.ground_truth
            paths = {document.path for document in load_sources(case.path)}
            for key in ("valid_current_sources", "valid_target_sources", "required_conflict_sources"):
                for source in truth.get(key, []):
                    self.assertIn(source, paths, (case.id, source))

    def test_prompts_are_fair_except_for_one_context_section(self) -> None:
        case = next(item for item in self.cases if item.id == "diverged-01")
        raw = build_prompt(case, "raw")
        context = build_prompt(case, "contextcanon")
        self.assertEqual(raw.split("\n\nQuestion:")[0], context.split(
            "\n\nAdditional structured context produced by ContextCanon:"
        )[0])
        self.assertEqual(raw.count("--- "), context.split(
            "\n\nAdditional structured context produced by ContextCanon:"
        )[0].count("--- "))
        self.assertIn(SHARED_INSTRUCTIONS, raw)
        self.assertIn("Additional structured context produced by ContextCanon", context)
        self.assertNotIn("ground_truth", raw)
        self.assertEqual(context, build_prompt(case, "contextcanon"))
        self.assertIn("JSON null", SHARED_INSTRUCTIONS)
        self.assertNotIn('"JWT or OAuth2 or null"', SHARED_INSTRUCTIONS)

    def test_pipeline_representative_cases_matches_ground_truth(self) -> None:
        for case_id in ("consistent-01", "diverged-01", "runtime-conflict-01", "intent-conflict-01"):
            case = next(item for item in self.cases if item.id == case_id)
            documents = load_sources(case.path)
            claims = extract_demo_claims(documents)
            verify_claims(claims, documents)
            groups = {}
            for claim in claims:
                groups.setdefault((claim.subject, claim.predicate), []).append(claim)
            resolutions = [DefaultResolutionPolicy().resolve(group) for group in groups.values()]
            package = assemble_context(
                task=case.question,
                resolutions=resolutions,
                source_revision="test",
                created_at="2026-01-01T00:00:00Z",
                policy_name="default",
                policy_version="0.1",
            )
            statuses = {item.resolution_status.value for item in package.items}
            if case_id == "consistent-01":
                self.assertEqual(statuses, {"RESOLVED"})
            else:
                self.assertIn(case.ground_truth["status"], statuses | {resolution.status.value for resolution in package.unresolved_conflicts})

    def test_scoring_rules_and_source_normalization(self) -> None:
        case = next(item for item in self.cases if item.id == "runtime-conflict-01")
        prediction = {
            "current": None,
            "target": None,
            "status": "AMBIGUOUS",
            "evidence": {
                "current": [],
                "target": [],
                "conflict": ["./config\\jwt.yaml", "config/oauth.json"],
            },
        }
        self.assertEqual(score_case(case, prediction)["overall"], 4)
        self.assertEqual(validate_prediction(prediction)["evidence"]["conflict"][0], "config/jwt.yaml")

    def test_correct_value_and_source_pass_traceability(self) -> None:
        case = next(item for item in self.cases if item.id == "diverged-01")
        prediction = {
            "current": "JWT",
            "target": "OAuth2",
            "status": "DIVERGED",
            "evidence": {
                "current": ["config/auth.yaml"],
                "target": ["docs/adr/ADR-015.md"],
                "conflict": [],
            },
        }
        self.assertEqual(score_case(case, prediction)["traceability"], 1)

    def test_wrong_value_with_correct_source_fails_traceability(self) -> None:
        case = next(item for item in self.cases if item.id == "diverged-01")
        prediction = {
            "current": "OAuth2",
            "target": "OAuth2",
            "status": "DIVERGED",
            "evidence": {
                "current": ["config/auth.yaml"],
                "target": ["docs/adr/ADR-015.md"],
                "conflict": [],
            },
        }
        self.assertEqual(score_case(case, prediction)["traceability"], 0)

    def test_correct_value_with_wrong_source_fails_traceability(self) -> None:
        case = next(item for item in self.cases if item.id == "diverged-01")
        prediction = {
            "current": "JWT",
            "target": "OAuth2",
            "status": "DIVERGED",
            "evidence": {
                "current": ["wrong.yaml"],
                "target": ["docs/adr/ADR-015-oauth.md"],
                "conflict": [],
            },
        }
        self.assertEqual(score_case(case, prediction)["traceability"], 0)

    def test_ambiguous_case_missing_conflict_source_fails_traceability(self) -> None:
        case = next(item for item in self.cases if item.id == "runtime-conflict-01")
        prediction = {
            "current": None,
            "target": None,
            "status": "AMBIGUOUS",
            "evidence": {
                "current": [],
                "target": [],
                "conflict": ["config/jwt.yaml"],
            },
        }
        self.assertEqual(score_case(case, prediction)["traceability"], 0)

    def test_stability_prompt_export_writes_only_selected_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            export_prompts(output, self.cases)
            for case in self.cases:
                expected = 3 if case.id in STABILITY_CASE_IDS else 1
                self.assertEqual(
                    len(list((output / "raw").glob(f"{case.id}__*.txt"))),
                    expected,
                )
                self.assertEqual(
                    len(list((output / "contextcanon").glob(f"{case.id}__*.txt"))),
                    expected,
                )
            self.assertTrue((output / "raw" / "diverged-01__reverse.txt").exists())
            self.assertTrue((output / "contextcanon" / "diverged-01__permutation.txt").exists())
            for case in self.cases:
                self.assertEqual(
                    (output / "raw" / f"{case.id}__canonical.txt").stem,
                    f"{case.id}__canonical",
                )

    def test_main_scores_canonical_predictions_and_stability_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            for condition in ("raw", "contextcanon"):
                condition_dir = results / condition
                condition_dir.mkdir()
                for case in self.cases:
                    truth = case.ground_truth
                    prediction = {
                        "current": truth["current"],
                        "target": truth["target"],
                        "status": truth["status"],
                        "evidence": {
                            "current": truth.get("valid_current_sources", [])[:1],
                            "target": truth.get("valid_target_sources", [])[:1],
                            "conflict": truth.get("required_conflict_sources", []),
                        },
                    }
                    (condition_dir / f"{case.id}__canonical.json").write_text(
                        json.dumps(prediction), encoding="utf-8"
                    )
                    if case.id in STABILITY_CASE_IDS:
                        for ordering in ("reverse", "permutation"):
                            (condition_dir / f"{case.id}__{ordering}.json").write_text(
                                json.dumps(prediction), encoding="utf-8"
                            )
            previous_argv = sys.argv
            output = io.StringIO()
            try:
                sys.argv = ["scorer", str(results)]
                with contextlib.redirect_stdout(output):
                    self.assertEqual(scorer_main(), 0)
            finally:
                sys.argv = previous_argv
            report = json.loads(output.getvalue())
            for condition in ("raw", "contextcanon"):
                self.assertEqual(report[condition]["summary"]["cases_scored"], 15)
                self.assertEqual(report[condition]["summary"]["stability_rate"], 1.0)

    def test_wrong_source_loses_traceability(self) -> None:
        case = next(item for item in self.cases if item.id == "diverged-01")
        prediction = {
            "current": "JWT",
            "target": "OAuth2",
            "status": "DIVERGED",
            "evidence": {
                "current": ["wrong.yaml"],
                "target": ["wrong.md"],
                "conflict": [],
            },
        }
        self.assertEqual(score_case(case, prediction)["traceability"], 0)

    def test_stability_variants_are_deterministic(self) -> None:
        case = next(item for item in self.cases if item.id == "order-noise-03")
        self.assertEqual(
            stability_variants(case),
            [(case.id, "canonical"), (case.id, "reverse"), (case.id, "permutation")],
        )
        prediction = {
            "current": "JWT",
            "target": "OAuth2",
            "status": "DIVERGED",
            "evidence": {"current": [], "target": [], "conflict": []},
        }
        self.assertEqual(stability_rate({case.id: [prediction] * 3}), 1.0)

    def test_report_is_human_readable(self) -> None:
        summary = summarize([], stability=0.75)
        rendered = render_report(summary, summary)
        self.assertIn("Current Accuracy", rendered)
        self.assertIn("ContextCanon", rendered)
        self.assertIn("75.0%", rendered)


if __name__ == "__main__":
    unittest.main()
