import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from contextcanon.core import SourceType
from contextcanon.sources import (
    ConfigLoader,
    MarkdownLoader,
    SourceDocument,
    load_sources,
)


class SourceLoaderTests(unittest.TestCase):
    def test_source_document_is_an_immutable_data_record(self) -> None:
        document = SourceDocument(
            path="README.md",
            source_type=SourceType.MARKDOWN,
            content="Current protocol: JWT",
        )

        with self.assertRaises(FrozenInstanceError):
            document.path = "OTHER.md"  # type: ignore[misc]

    def test_demo_sources_are_loaded_with_relative_posix_paths(self) -> None:
        root = Path(__file__).resolve().parents[1] / "examples" / "demo-auth"

        documents = load_sources(root)

        self.assertEqual(
            [document.path for document in documents],
            [
                "README.md",
                "config/auth.json",
                "config/auth.yaml",
                "docs/adr/ADR-015-oauth.md",
            ],
        )
        self.assertEqual(
            [document.source_type for document in documents],
            [
                SourceType.MARKDOWN,
                SourceType.JSON,
                SourceType.YAML,
                SourceType.ADR,
            ],
        )
        self.assertIsInstance(documents[0].content, str)
        self.assertEqual(documents[1].content, {"auth": {"provider": "jwt"}})
        self.assertEqual(documents[2].content, {"auth": {"provider": "jwt"}})

    def test_individual_loaders_parse_only_their_supported_formats(self) -> None:
        markdown_loader = MarkdownLoader()
        config_loader = ConfigLoader()

        self.assertTrue(markdown_loader.supports(Path("README.md")))
        self.assertFalse(markdown_loader.supports(Path("auth.yaml")))
        self.assertTrue(config_loader.supports(Path("auth.json")))
        self.assertTrue(config_loader.supports(Path("auth.yml")))
        self.assertFalse(config_loader.supports(Path("notes.txt")))

    def test_load_sources_ignores_unsupported_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text(
                "Current protocol: JWT\n",
                encoding="utf-8",
            )
            (root / "notes.txt").write_text(
                "Target protocol: OAuth2\n",
                encoding="utf-8",
            )

            documents = load_sources(root)

        self.assertEqual([document.path for document in documents], ["README.md"])

    def test_non_json_yaml_values_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "config.yaml"
            source.write_text("released: 2026-09-19\n", encoding="utf-8")

            with self.assertRaises(TypeError):
                ConfigLoader().load(source, root=root)

    def test_non_finite_numbers_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "config.yaml"
            source.write_text("ratio: .nan\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                ConfigLoader().load(source, root=root)

    def test_loaded_config_content_is_json_serializable(self) -> None:
        root = Path(__file__).resolve().parents[1] / "examples" / "demo-auth"

        for document in load_sources(root):
            json.dumps(document.content, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
