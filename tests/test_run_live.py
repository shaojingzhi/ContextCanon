import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from benchmarks.build_prompts import load_cases
from benchmarks.run_live import (
    DEFAULT_BASE_URL,
    ModelResponse,
    OpenAICompatibleClient,
    TransportError,
    discover_prompts,
    run,
)
from benchmarks.scorer import load_predictions, main as scorer_main


VALID = {
    "current": "JWT",
    "target": "OAuth2",
    "status": "DIVERGED",
    "evidence": {"current": [], "target": [], "conflict": []},
}


class FakeModel:
    def __init__(self, responses=None, failures=0):
        self.responses = list(responses or [json.dumps(VALID)])
        self.failures = failures
        self.calls = []

    def __call__(self, prompt, model, temperature):
        self.calls.append((prompt, model, temperature))
        if self.failures:
            self.failures -= 1
            raise TransportError("temporary")
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return ModelResponse(response)


def write_prompts(root: Path) -> None:
    for condition in ("raw", "contextcanon"):
        directory = root / condition
        directory.mkdir(parents=True)
        (directory / "case-01__canonical.txt").write_text("canonical", encoding="utf-8")
        (directory / "case-01__reverse.txt").write_text("reverse", encoding="utf-8")
        (directory / "case-01__permutation.txt").write_text("permutation", encoding="utf-8")


class LiveRunnerTests(unittest.TestCase):
    def test_prompt_discovery_and_dry_run_count_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts"
            write_prompts(prompts)
            discovered = discover_prompts(prompts)
            self.assertEqual(len(discovered), 6)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(run(prompts, Path(directory) / "results", "model", "run-01", dry_run=True), 6)
            self.assertIn("expected requests: 6", output.getvalue())
            self.assertIn("raw/case-01__canonical.json", output.getvalue())

    def test_same_model_settings_are_used_for_both_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts"
            write_prompts(prompts)
            fake = FakeModel()
            results = Path(directory) / "results"
            run(prompts, results, "model-x", "run-01", condition="all", client=fake)
            self.assertEqual({(model, temperature) for _, model, temperature in fake.calls}, {("model-x", 0)})
            metadata = {
                tuple(
                    json.loads(path.read_text(encoding="utf-8"))[key]
                    for key in ("model", "temperature", "thinking", "reasoning_effort")
                )
                for path in (results / "run-01").glob("*/*.json")
            }
            self.assertEqual(metadata, {("model-x", 0, "enabled", "high")})

    def test_deepseek_request_enables_thinking_and_high_reasoning(self) -> None:
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": json.dumps(VALID)}}]}
        ).encode("utf-8")
        with mock.patch("benchmarks.run_live.request.urlopen", return_value=response) as urlopen:
            client = OpenAICompatibleClient("secret")
            client("prompt", "deepseek-v4-pro", 0)
        request_body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(request_body["thinking"], {"type": "enabled"})
        self.assertEqual(request_body["reasoning_effort"], "high")
        self.assertEqual(urlopen.call_args.args[0].full_url, f"{DEFAULT_BASE_URL}/chat/completions")

    def test_deepseek_api_key_is_read_without_persisting_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts"
            write_prompts(prompts)
            results = Path(directory) / "results"
            fake = FakeModel()
            with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "do-not-save"}, clear=True):
                with mock.patch("benchmarks.run_live.OpenAICompatibleClient", return_value=fake) as client:
                    run(prompts, results, "model", "run-01", condition="raw")
            client.assert_called_once_with("do-not-save", DEFAULT_BASE_URL)
            for path in (results / "run-01" / "raw").glob("*.json"):
                self.assertNotIn("do-not-save", path.read_text(encoding="utf-8"))

    def test_valid_result_has_expected_filename_and_scorer_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts"
            write_prompts(prompts)
            results = Path(directory) / "results"
            run(prompts, results, "model", "run-01", condition="raw", client=FakeModel())
            path = results / "run-01" / "raw" / "case-01__canonical.json"
            self.assertTrue(path.exists())
            self.assertTrue((path.parent / "case-01__reverse.json").exists())
            self.assertTrue((path.parent / "case-01__permutation.json").exists())
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["prediction"], VALID)
            self.assertEqual(load_predictions(results / "run-01", "raw")["case-01__canonical"], VALID)

    def test_invalid_json_is_saved_without_repair_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts"
            write_prompts(prompts)
            fake = FakeModel(["not json"])
            run(prompts, Path(directory) / "results", "model", "run-01", condition="raw", client=fake)
            record = json.loads(
                (Path(directory) / "results" / "run-01" / "raw" / "case-01__canonical.json").read_text()
            )
            self.assertEqual(record["raw_response"], "not json")
            self.assertIsNone(record["prediction"])
            self.assertTrue(record["parse_error"])
            self.assertEqual(len(fake.calls), 3)

    def test_transport_retry_is_bounded_but_schema_failure_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts"
            write_prompts(prompts)
            retrying = FakeModel(failures=2)
            run(prompts, Path(directory) / "results", "model", "run-01", condition="raw", client=retrying, retries=2)
            self.assertEqual(len(retrying.calls), 5)
            failing = FakeModel(["{}"])
            run(prompts, Path(directory) / "results", "model", "run-02", condition="raw", client=failing)
            self.assertEqual(len(failing.calls), 3)

    def test_run_id_directories_are_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts"
            write_prompts(prompts)
            results = Path(directory) / "results"
            run(prompts, results, "model", "run-01", condition="raw", client=FakeModel())
            with self.assertRaises(FileExistsError):
                run(prompts, results, "model", "run-01", condition="raw", client=FakeModel())

    def test_scorer_counts_invalid_live_record_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results" / "run-01"
            case_id = load_cases()[0].id
            for condition in ("raw", "contextcanon"):
                target = results / condition
                target.mkdir(parents=True)
                (target / f"{case_id}__canonical.json").write_text(
                    json.dumps({"prediction": None, "parse_error": "invalid JSON"}),
                    encoding="utf-8",
                )
            output = io.StringIO()
            previous_argv = sys.argv
            try:
                sys.argv = ["scorer", str(results)]
                with contextlib.redirect_stdout(output):
                    self.assertEqual(scorer_main(), 0)
            finally:
                sys.argv = previous_argv
            report = json.loads(output.getvalue())
            self.assertEqual(report["raw"]["summary"]["cases_scored"], 1)
            self.assertEqual(report["raw"]["summary"]["overall_points"], 0)


if __name__ == "__main__":
    unittest.main()
