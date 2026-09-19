import json
from pathlib import Path
import unittest

from contextcanon.assembly import assemble_context
from contextcanon.extraction import extract_demo_claims
from contextcanon.renderers import JSONRenderer, MarkdownRenderer
from contextcanon.resolution import DefaultResolutionPolicy
from contextcanon.sources import load_sources
from contextcanon.verification import verify_claims


def _demo_package():
    root = Path(__file__).resolve().parents[1] / "examples" / "demo-auth"
    documents = load_sources(root)
    claims = extract_demo_claims(documents)
    verify_claims(claims, documents)
    resolution = DefaultResolutionPolicy().resolve(claims)
    return assemble_context(
        task="How does authentication currently work?",
        resolutions=[resolution],
        source_revision="abc123",
        created_at="2026-09-20T01:23:45Z",
        policy_name="default",
        policy_version="0.1",
    )


class ContextRendererTests(unittest.TestCase):
    def test_json_renderer_uses_context_package_serialization(self) -> None:
        package = _demo_package()
        renderer = JSONRenderer()

        first = renderer.render(package)
        second = renderer.render(package)
        parsed = json.loads(first)

        self.assertEqual(first, second)
        self.assertEqual(parsed, package.to_dict())
        self.assertEqual(parsed["id"], package.id)
        self.assertEqual(parsed["task"], package.task)
        self.assertEqual(len(parsed["items"]), 2)
        self.assertEqual(len(parsed["unresolved_conflicts"]), 1)

    def test_markdown_renderer_preserves_demo_semantics_and_provenance(self) -> None:
        rendered = MarkdownRenderer().render(_demo_package())

        for expected in (
            "# Context Package",
            "JWT",
            "OAuth2",
            "DIVERGED",
            "config/auth.yaml",
            "docs/adr/ADR-015-oauth.md",
            "INTENT_IMPLEMENTATION_DIVERGENCE",
            "## Unresolved Knowledge",
        ):
            self.assertIn(expected, rendered)


if __name__ == "__main__":
    unittest.main()

