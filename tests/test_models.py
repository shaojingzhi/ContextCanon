import json
import unittest

from contextcanon import (
    Claim,
    ClaimType,
    ContextItem,
    ContextPackage,
    Evidence,
    EvidenceRole,
    ReasonCode,
    Resolution,
    ResolutionStatus,
    SourceType,
)


class ModelSerializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = Evidence(
            id="e-config",
            source_id="config/auth.yaml",
            source_type=SourceType.YAML,
            location="auth.provider",
            role=EvidenceRole.OBSERVED,
            content={"provider": "jwt"},
            verified=True,
        )
        self.claim = Claim(
            id="c-runtime-auth",
            subject="auth",
            predicate="protocol",
            value="JWT",
            claim_type=ClaimType.RUNTIME_STATE,
            evidence=[self.evidence],
            confidence=1.0,
        )

    def test_package_imports_and_exports_models(self) -> None:
        self.assertEqual(self.claim.claim_type, ClaimType.RUNTIME_STATE)
        self.assertEqual(self.evidence.role, EvidenceRole.OBSERVED)

    def test_nested_models_serialize_to_json_compatible_dict(self) -> None:
        resolution = Resolution(
            status=ResolutionStatus.RESOLVED,
            selected_claims=[self.claim],
            reason_codes=[ReasonCode.VERIFIED_EVIDENCE],
            policy="default",
            policy_version="0.1",
        )
        package = ContextPackage(
            id="ctx-1",
            task="How does authentication work?",
            items=[
                ContextItem(
                    claim=self.claim,
                    status=ResolutionStatus.RESOLVED,
                    supporting_evidence=[self.evidence],
                    reason_codes=[ReasonCode.VERIFIED_EVIDENCE],
                )
            ],
            unresolved_conflicts=[resolution],
            source_revision="abc123",
            policy_name="default",
            policy_version="0.1",
            token_budget=500,
            created_at="2026-01-01T00:00:00+00:00",
        )

        serialized = package.to_dict()
        encoded = json.dumps(serialized, sort_keys=True)

        self.assertEqual(serialized["items"][0]["claim"]["value"], "JWT")
        self.assertEqual(serialized["items"][0]["status"], "RESOLVED")
        self.assertEqual(serialized["unresolved_conflicts"][0]["reason_codes"], [
            "VERIFIED_EVIDENCE"
        ])
        self.assertIn('"source_revision": "abc123"', encoded)

    def test_json_round_trip_restores_nested_model_types(self) -> None:
        package = ContextPackage(
            id="ctx-1",
            task="task",
            items=[ContextItem(claim=self.claim)],
        )

        restored = ContextPackage.from_json(package.to_json())

        self.assertEqual(restored.id, package.id)
        self.assertEqual(restored.items[0].claim.value, "JWT")
        self.assertIs(restored.items[0].claim.claim_type, ClaimType.RUNTIME_STATE)
        self.assertIs(restored.items[0].claim.evidence[0].source_type, SourceType.YAML)


if __name__ == "__main__":
    unittest.main()
