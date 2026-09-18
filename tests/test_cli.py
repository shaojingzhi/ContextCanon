import contextlib
import io
import unittest

from contextcanon.cli import main


class CliPlaceholderTests(unittest.TestCase):
    def test_no_arguments_prints_help_and_succeeds(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("commands arrive in later milestones", output.getvalue())

    def test_future_command_is_not_reported_as_success(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["scan"])

        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
