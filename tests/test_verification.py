import json
from pathlib import Path
import unittest

from contextcanon.core import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceRole,
    SourceType,
)
from contextcanon.extraction import extract_demo_claims
from contextcanon.sources import SourceDocument, load_sources
from contextcanon.verification import (
    JsonPathVerifier,
    TextPresenceVerifier,
    VerificationResult,
    VerificationStatus,
    YamlPathVerifier,
    verify_claims,
)


def _claim_with_evidence(evidence: Evidence) -> Claim:
    return Claim(
        id="claim-auth-protocol",
        subject="auth",
        predicate="protocol",
        value="JWT",
        claim_type=ClaimType.RUNTIME_STATE,
        evidence=[evidence],
    )


class VerificationTests(unittest.TestCase):
    def test_yaml_path_verifier_checks_raw_evidence_value(self) -> None:
        evidence = Evidence(
            id="e-yaml",
            source_id="config/auth.yaml",
            source_type=SourceType.YAML,
            location="auth.provider",
            role=EvidenceRole.OBSERVED,
            content="jwt",
        )
        claim = _claim_with_evidence(evidence)
        document = SourceDocument(
            path=evidence.source_id,
            source_type=SourceType.YAML,
            content={"auth": {"provider": "jwt"}},
        )

        result = YamlPathVerifier().verify(evidence, document)

        self.assertEqual(result.status, VerificationStatus.VERIFIED)
        self.assertEqual(claim.value, "JWT")
        self.assertEqual(evidence.content, "jwt")

    def test_json_path_verifier_checks_nested_path(self) -> None:
        evidence = Evidence(
            id="e-json",
            source_id="config/auth.json",
            source_type=SourceType.JSON,
            location="auth.provider",
            role=EvidenceRole.OBSERVED,
            content="jwt",
        )
        document = SourceDocument(
            path=evidence.source_id,
            source_type=SourceType.JSON,
            content={"auth": {"provider": "jwt"}},
        )

        result = JsonPathVerifier().verify(evidence, document)

        self.assertEqual(result.status, VerificationStatus.VERIFIED)

    def test_structured_mismatch_is_an_explicit_failure(self) -> None:
        evidence = Evidence(
            id="e-yaml",
            source_id="config/auth.yaml",
            source_type=SourceType.YAML,
            location="auth.provider",
            role=EvidenceRole.OBSERVED,
            content="jwt",
        )
        claim = _claim_with_evidence(evidence)
        document = SourceDocument(
            path=evidence.source_id,
            source_type=SourceType.YAML,
            content={"auth": {"provider": "oauth2"}},
        )

        results = verify_claims([claim], [document])

        self.assertEqual(results[0].status, VerificationStatus.FAILED)
        self.assertIn("found 'oauth2'", results[0].message)
        self.assertEqual(evidence.verifier, "yaml-path")
        self.assertIs(evidence.verified, False)

    def test_missing_structured_path_is_an_explicit_failure(self) -> None:
        evidence = Evidence(
            id="e-json",
            source_id="config/auth.json",
            source_type=SourceType.JSON,
            location="auth.provider",
            role=EvidenceRole.OBSERVED,
            content="jwt",
        )
        document = SourceDocument(
            path=evidence.source_id,
            source_type=SourceType.JSON,
            content={"auth": {}},
        )

        result = JsonPathVerifier().verify(evidence, document)

        self.assertEqual(result.status, VerificationStatus.FAILED)
        self.assertIn("does not contain path", result.message)

    def test_text_presence_verifier_checks_exact_recorded_line(self) -> None:
        evidence = Evidence(
            id="e-readme",
            source_id="README.md",
            source_type=SourceType.MARKDOWN,
            location="line:2",
            role=EvidenceRole.DOCUMENTED,
            content="Current protocol: JWT",
        )
        document = SourceDocument(
            path=evidence.source_id,
            source_type=SourceType.MARKDOWN,
            content="# Auth\nCurrent protocol: JWT\n",
        )

        result = TextPresenceVerifier().verify(evidence, document)

        self.assertEqual(result.status, VerificationStatus.VERIFIED)

    def test_text_change_is_an_explicit_failure(self) -> None:
        evidence = Evidence(
            id="e-readme",
            source_id="README.md",
            source_type=SourceType.MARKDOWN,
            location="line:2",
            role=EvidenceRole.DOCUMENTED,
            content="Current protocol: JWT",
        )
        document = SourceDocument(
            path=evidence.source_id,
            source_type=SourceType.MARKDOWN,
            content="# Auth\nCurrent protocol: OAuth2\n",
        )

        result = TextPresenceVerifier().verify(evidence, document)

        self.assertEqual(result.status, VerificationStatus.FAILED)
        self.assertIn("Current protocol: OAuth2", result.message)

    def test_missing_source_is_reported_as_an_error(self) -> None:
        evidence = Evidence(
            id="e-yaml",
            source_id="config/missing.yaml",
            source_type=SourceType.YAML,
            location="auth.provider",
            role=EvidenceRole.OBSERVED,
            content="jwt",
        )
        claim = _claim_with_evidence(evidence)

        results = verify_claims([claim], [])

        self.assertEqual(results[0].status, VerificationStatus.ERROR)
        self.assertIn("was not loaded", results[0].message)
        self.assertIsNone(evidence.verified)

    def test_verifier_error_does_not_stop_remaining_evidence(self) -> None:
        class RaisingVerifier:
            name = "raising"

            def verify(
                self,
                evidence: Evidence,
                document: SourceDocument,
            ) -> VerificationResult:
                raise RuntimeError("broken verifier")

        yaml_evidence = Evidence(
            id="e-yaml",
            source_id="config/auth.yaml",
            source_type=SourceType.YAML,
            location="auth.provider",
            role=EvidenceRole.OBSERVED,
            content="jwt",
        )
        json_evidence = Evidence(
            id="e-json",
            source_id="config/auth.json",
            source_type=SourceType.JSON,
            location="auth.provider",
            role=EvidenceRole.OBSERVED,
            content="jwt",
        )
        claim = _claim_with_evidence(yaml_evidence)
        claim.evidence.append(json_evidence)
        documents = [
            SourceDocument(
                path=yaml_evidence.source_id,
                source_type=SourceType.YAML,
                content={"auth": {"provider": "jwt"}},
            ),
            SourceDocument(
                path=json_evidence.source_id,
                source_type=SourceType.JSON,
                content={"auth": {"provider": "jwt"}},
            ),
        ]

        results = verify_claims(
            [claim],
            documents,
            verifiers={
                SourceType.YAML: RaisingVerifier(),
                SourceType.JSON: JsonPathVerifier(),
            },
        )

        self.assertEqual(
            [result.status for result in results],
            [VerificationStatus.ERROR, VerificationStatus.VERIFIED],
        )
        self.assertIn("RuntimeError: broken verifier", results[0].message)
        self.assertIsNone(yaml_evidence.verified)
        self.assertIs(json_evidence.verified, True)

    def test_extracted_evidence_fails_against_modified_source(self) -> None:
        original = SourceDocument(
            path="config/auth.yaml",
            source_type=SourceType.YAML,
            content={"auth": {"provider": "jwt"}},
        )
        claims = extract_demo_claims([original])
        evidence = claims[0].evidence[0]
        modified = SourceDocument(
            path=original.path,
            source_type=original.source_type,
            content={"auth": {"provider": "oauth2"}},
        )

        results = verify_claims(claims, [modified])

        self.assertEqual(evidence.content, "jwt")
        self.assertEqual(results[0].status, VerificationStatus.FAILED)
        self.assertIs(evidence.verified, False)

    def test_demo_verifies_all_extracted_evidence_end_to_end(self) -> None:
        root = Path(__file__).resolve().parents[1] / "examples" / "demo-auth"
        documents = load_sources(root)
        claims = extract_demo_claims(documents)

        results = verify_claims(claims, documents)

        self.assertEqual(len(results), 4)
        self.assertTrue(
            all(result.status is VerificationStatus.VERIFIED for result in results)
        )
        config_evidence = [
            evidence
            for claim in claims
            for evidence in claim.evidence
            if evidence.source_type in {SourceType.YAML, SourceType.JSON}
        ]
        self.assertEqual(len(config_evidence), 2)
        self.assertTrue(
            all(evidence.verified is True for evidence in config_evidence)
        )

    def test_verification_result_serializes_to_json_primitives(self) -> None:
        evidence = Evidence(
            id="e-yaml",
            source_id="config/auth.yaml",
            source_type=SourceType.YAML,
            location="auth.provider",
            role=EvidenceRole.OBSERVED,
            content="jwt",
        )
        document = SourceDocument(
            path=evidence.source_id,
            source_type=SourceType.YAML,
            content={"auth": {"provider": "jwt"}},
        )

        serialized = YamlPathVerifier().verify(evidence, document).to_dict()

        self.assertEqual(serialized["status"], "VERIFIED")
        json.dumps(serialized, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
