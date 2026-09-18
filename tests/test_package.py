import importlib
from pathlib import Path
import tomllib
import unittest


class PackageScaffoldTests(unittest.TestCase):
    def test_package_imports(self) -> None:
        package = importlib.import_module("contextcanon")

        self.assertEqual(package.__name__, "contextcanon")

    def test_console_script_points_to_cli_main(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with (project_root / "pyproject.toml").open("rb") as file:
            pyproject = tomllib.load(file)

        self.assertEqual(
            pyproject["project"]["scripts"]["contextcanon"],
            "contextcanon.cli:main",
        )


if __name__ == "__main__":
    unittest.main()
