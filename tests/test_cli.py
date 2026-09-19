import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from contextcanon.cli import main


class CliTests(unittest.TestCase):
    def test_no_arguments_prints_help_and_succeeds(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("doctor", output.getvalue())
        self.assertIn("build", output.getvalue())

    def test_doctor_reports_demo_divergence_with_provenance(self) -> None:
        root = Path(__file__).resolve().parents[1] / "examples" / "demo-auth"
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = main(["doctor", str(root)])

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        for expected in (
            "Property: auth.protocol",
            "JWT",
            "OAuth2",
            "DIVERGED",
            "INTENT_IMPLEMENTATION_DIVERGENCE",
            "config/auth.yaml",
            "config/auth.json",
            "docs/adr/ADR-015-oauth.md",
        ):
            self.assertIn(expected, rendered)
        self.assertLess(
            rendered.index("Current runtime:"),
            rendered.index("Architecture intent:"),
        )

    def test_doctor_defaults_to_current_directory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            (config / "auth.yaml").write_text(
                "auth:\n  provider: jwt\n",
                encoding="utf-8",
            )
            output = io.StringIO()

            with contextlib.chdir(root), contextlib.redirect_stdout(output):
                exit_code = main(["doctor"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Property: auth.protocol", output.getvalue())
        self.assertIn("config/auth.yaml [verified]", output.getvalue())

    def test_doctor_invalid_path_returns_usage_error_without_traceback(self) -> None:
        error_output = io.StringIO()

        with contextlib.redirect_stderr(error_output):
            exit_code = main(["doctor", "missing-directory"])

        rendered = error_output.getvalue()
        self.assertEqual(exit_code, 2)
        self.assertIn("is not a directory", rendered)
        self.assertNotIn("Traceback", rendered)

    def test_doctor_with_no_claims_reports_success(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.md").write_text(
                "# Notes\n\nNothing about authentication.\n",
                encoding="utf-8",
            )
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                exit_code = main(["doctor", str(root)])

        self.assertEqual(exit_code, 0)
        self.assertIn("No supported knowledge claims found.", output.getvalue())

    def test_doctor_renders_ambiguous_claims_without_selecting_a_winner(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            (config / "jwt.yaml").write_text(
                "auth:\n  provider: jwt\n",
                encoding="utf-8",
            )
            (config / "oauth.json").write_text(
                '{"auth": {"provider": "oauth2"}}\n',
                encoding="utf-8",
            )
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                exit_code = main(["doctor", str(root)])

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("AMBIGUOUS", rendered)
        self.assertIn("Conflicting claims:", rendered)
        self.assertIn("CONFLICTING_AUTHORITATIVE_EVIDENCE", rendered)
        self.assertNotIn("Current runtime:", rendered)

    def test_build_renders_demo_as_markdown_by_default(self) -> None:
        root = Path(__file__).resolve().parents[1] / "examples" / "demo-auth"
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = main(
                [
                    "build",
                    "How does authentication currently work?",
                    "--path",
                    str(root),
                ]
            )

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        for expected in ("Context Package", "JWT", "OAuth2", "DIVERGED"):
            self.assertIn(expected, rendered)

    def test_build_renders_valid_json(self) -> None:
        root = Path(__file__).resolve().parents[1] / "examples" / "demo-auth"
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = main(
                [
                    "build",
                    "authentication task",
                    "--path",
                    str(root),
                    "--format",
                    "json",
                ]
            )

        package = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(package["task"], "authentication task")
        self.assertEqual(len(package["items"]), 2)
        self.assertEqual(package["unresolved_conflicts"][0]["status"], "DIVERGED")

    def test_build_with_no_claims_produces_an_empty_package(self) -> None:
        with TemporaryDirectory() as directory:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(
                    [
                        "build",
                        "task",
                        "--path",
                        directory,
                        "--format",
                        "json",
                    ]
                )

        package = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(package["items"], [])
        self.assertEqual(package["unresolved_conflicts"], [])

    def test_build_invalid_path_returns_usage_error_without_traceback(self) -> None:
        error_output = io.StringIO()

        with contextlib.redirect_stderr(error_output):
            exit_code = main(
                ["build", "task", "--path", "missing-directory"]
            )

        rendered = error_output.getvalue()
        self.assertEqual(exit_code, 2)
        self.assertIn("contextcanon build: error", rendered)
        self.assertNotIn("Traceback", rendered)

    def test_build_unknown_format_is_rejected_by_argparse(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["build", "task", "--format", "html"])

        self.assertEqual(error.exception.code, 2)

    def test_unknown_command_is_not_reported_as_success(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["scan"])

        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
