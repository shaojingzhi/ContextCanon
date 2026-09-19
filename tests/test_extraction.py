from pathlib import Path
import unittest

from contextcanon.core import ClaimType, EvidenceRole, SourceType
from contextcanon.extraction import extract_demo_claims
from contextcanon.sources import SourceDocument, load_sources


class DemoExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1] / "examples" / "demo-auth"
        self.claims = extract_demo_claims(load_sources(root))

    def test_demo_identifies_runtime_and_architecture_intent(self) -> None:
        self.assertEqual(len(self.claims), 2)
        facts = {
            (claim.claim_type, claim.subject, claim.predicate, claim.value)
            for claim in self.claims
        }
        self.assertEqual(
            facts,
            {
                (ClaimType.RUNTIME_STATE, "auth", "protocol", "JWT"),
                (
                    ClaimType.ARCHITECTURE_INTENT,
                    "auth",
                    "protocol",
                    "OAuth2",
                ),
            },
        )

    def test_same_semantic_claim_aggregates_evidence_from_multiple_sources(self) -> None:
        runtime_claim = next(
            claim
            for claim in self.claims
            if claim.claim_type is ClaimType.RUNTIME_STATE
        )

        self.assertEqual(
            [evidence.source_id for evidence in runtime_claim.evidence],
            ["README.md", "config/auth.json", "config/auth.yaml"],
        )
        self.assertEqual(
            [evidence.role for evidence in runtime_claim.evidence],
            [
                EvidenceRole.DOCUMENTED,
                EvidenceRole.OBSERVED,
                EvidenceRole.OBSERVED,
            ],
        )
        self.assertEqual(len({item.id for item in runtime_claim.evidence}), 3)
        self.assertTrue(all(item.verified is None for item in runtime_claim.evidence))
        self.assertIsNone(runtime_claim.confidence)

    def test_claim_id_depends_on_semantics_not_provenance(self) -> None:
        first = SourceDocument(
            path="one/config.yaml",
            source_type=SourceType.YAML,
            content={"auth": {"provider": "jwt"}},
        )
        second = SourceDocument(
            path="two/config.json",
            source_type=SourceType.JSON,
            content={"auth": {"provider": "JWT"}},
        )

        separate_ids = {
            extract_demo_claims([document])[0].id
            for document in (first, second)
        }
        combined = extract_demo_claims([first, second])

        self.assertEqual(len(separate_ids), 1)
        self.assertEqual(len(combined), 1)
        self.assertEqual(len(combined[0].evidence), 2)

    def test_claim_value_comes_from_content_not_file_name(self) -> None:
        misleading_path = SourceDocument(
            path="config/jwt.yaml",
            source_type=SourceType.YAML,
            content={"auth": {"provider": "oauth2"}},
        )

        claims = extract_demo_claims([misleading_path])

        self.assertEqual(claims[0].value, "OAuth2")

    def test_provenance_includes_source_path_and_precise_location(self) -> None:
        intent_claim = next(
            claim
            for claim in self.claims
            if claim.claim_type is ClaimType.ARCHITECTURE_INTENT
        )

        self.assertEqual(len(intent_claim.evidence), 1)
        evidence = intent_claim.evidence[0]
        self.assertEqual(evidence.source_id, "docs/adr/ADR-015-oauth.md")
        self.assertEqual(evidence.source_type, SourceType.ADR)
        self.assertEqual(evidence.location, "line:5")
        self.assertEqual(evidence.role, EvidenceRole.INTENDED)

    def test_unaccepted_adr_does_not_create_architecture_intent(self) -> None:
        proposed = SourceDocument(
            path="docs/adr/ADR-999.md",
            source_type=SourceType.ADR,
            content="Status: Proposed\n\nTarget protocol: OAuth2\n",
        )

        self.assertEqual(extract_demo_claims([proposed]), [])


if __name__ == "__main__":
    unittest.main()
