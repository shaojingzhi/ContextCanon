import json
import math
import unittest

from contextcanon.core import (
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
            timestamp="2026-01-01T00:00:00+00:00",
            verified=True,
        )
        self.claim = Claim(
            id="c-runtime-auth",
            subject="auth",
            predicate="protocol",
            value="JWT",
            claim_type=ClaimType.RUNTIME_STATE,
            evidence=[self.evidence],
            status=ResolutionStatus.RESOLVED,
            confidence=1.0,
        )

    def test_nested_models_serialize_to_json_primitives(self) -> None:
        conflicting_claim = Claim(
            id="c-intent-auth",
            subject="auth",
            predicate="protocol",
            value="OAuth2",
            claim_type=ClaimType.ARCHITECTURE_INTENT,
        )
        resolution = Resolution(
            status=ResolutionStatus.DIVERGED,
            selected_claims=[self.claim],
            conflicting_claims=[conflicting_claim],
            reason_codes=[ReasonCode.INTENT_IMPLEMENTATION_DIVERGENCE],
            policy="default",
            policy_version="0.1",
        )
        package = ContextPackage(
            id="ctx-1",
            task="How does authentication work?",
            source_revision="abc123",
            policy_name="default",
            policy_version="0.1",
            created_at="2026-01-01T00:00:00+00:00",
            items=[
                ContextItem(
                    claim=self.claim,
                    resolution_status=ResolutionStatus.RESOLVED,
                    supporting_evidence=[self.evidence],
                    reason_codes=[ReasonCode.VERIFIED_EVIDENCE],
                )
            ],
            unresolved_conflicts=[resolution],
            token_budget=500,
        )

        serialized = package.to_dict()
        encoded = json.dumps(serialized, sort_keys=True, allow_nan=False)

        self.assertIs(type(serialized["items"][0]["claim"]["claim_type"]), str)
        self.assertEqual(serialized["items"][0]["resolution_status"], "RESOLVED")
        self.assertEqual(
            serialized["unresolved_conflicts"][0]["reason_codes"],
            ["INTENT_IMPLEMENTATION_DIVERGENCE"],
        )
        self.assertEqual(
            serialized["unresolved_conflicts"][0]["conflicting_claims"][0][
                "value"
            ],
            "OAuth2",
        )
        self.assertIn('"source_revision": "abc123"', encoded)

    def test_json_objects_are_canonicalized_recursively(self) -> None:
        claim = Claim(
            id="c",
            subject="s",
            predicate="p",
            value={"z": 1, "a": {"y": 2, "b": 3}},
            claim_type=ClaimType.RUNTIME_STATE,
        )

        serialized = claim.to_dict()["value"]
        self.assertIsInstance(serialized, dict)
        self.assertEqual(list(serialized), ["a", "z"])
        nested = serialized["a"]
        self.assertIsInstance(nested, dict)
        self.assertEqual(list(nested), ["b", "y"])

    def test_unsupported_or_non_deterministic_values_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Claim(
                id="c",
                subject="s",
                predicate="p",
                value={"unordered"},
                claim_type=ClaimType.RUNTIME_STATE,
            )
        with self.assertRaises(TypeError):
            Evidence(
                id="e",
                source_id="s",
                source_type=SourceType.JSON,
                location="x",
                role=EvidenceRole.OBSERVED,
                content={1: "non-string key"},
            )
        with self.assertRaises(ValueError):
            Claim(
                id="c",
                subject="s",
                predicate="p",
                value=math.inf,
                claim_type=ClaimType.RUNTIME_STATE,
            )

    def test_canonical_enums_are_required(self) -> None:
        with self.assertRaises(TypeError):
            Claim(
                id="c",
                subject="s",
                predicate="p",
                value="v",
                claim_type="NOT_A_CLAIM_TYPE",  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            Evidence(
                id="e",
                source_id="s",
                source_type=SourceType.YAML,
                location="x",
                role="NOT_A_ROLE",  # type: ignore[arg-type]
            )

    def test_timestamps_use_one_canonical_representation(self) -> None:
        with self.assertRaises(TypeError):
            Evidence(
                id="e",
                source_id="s",
                source_type=SourceType.YAML,
                location="x",
                role=EvidenceRole.OBSERVED,
                timestamp=object(),  # type: ignore[arg-type]
            )

    def test_context_item_rejects_conflicting_statuses(self) -> None:
        with self.assertRaises(ValueError):
            ContextItem(
                claim=self.claim,
                resolution_status=ResolutionStatus.DIVERGED,
            )

    def test_context_item_evidence_must_belong_to_claim(self) -> None:
        unrelated_evidence = Evidence(
            id="e-other",
            source_id="README.md",
            source_type=SourceType.MARKDOWN,
            location="authentication",
            role=EvidenceRole.DOCUMENTED,
        )

        with self.assertRaises(ValueError):
            ContextItem(
                claim=self.claim,
                resolution_status=ResolutionStatus.RESOLVED,
                supporting_evidence=[unrelated_evidence],
            )

    def test_mutable_defaults_are_not_shared(self) -> None:
        first = Claim("first", "s", "p", "v", ClaimType.RUNTIME_STATE)
        second = Claim("second", "s", "p", "v", ClaimType.RUNTIME_STATE)

        first.evidence.append(self.evidence)

        self.assertEqual(second.evidence, [])

    def test_reproducibility_fields_are_required(self) -> None:
        with self.assertRaises(TypeError):
            ContextPackage(id="ctx", task="task")  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
