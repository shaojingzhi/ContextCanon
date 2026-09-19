from pathlib import Path
import unittest

from contextcanon.core import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceRole,
    ReasonCode,
    ResolutionStatus,
    SourceType,
)
from contextcanon.extraction import extract_demo_claims
from contextcanon.resolution import DefaultResolutionPolicy
from contextcanon.sources import load_sources
from contextcanon.verification import verify_claims


def _claim(
    claim_id: str,
    claim_type: ClaimType,
    value: str,
    *,
    role: EvidenceRole,
    verified: bool | None,
) -> Claim:
    source_type = (
        SourceType.ADR if role is EvidenceRole.INTENDED else SourceType.YAML
    )
    evidence = Evidence(
        id=f"e-{claim_id}",
        source_id=f"source-{claim_id}",
        source_type=source_type,
        location="auth.protocol",
        role=role,
        content=value,
        verified=verified,
    )
    return Claim(
        id=claim_id,
        subject="auth",
        predicate="protocol",
        value=value,
        claim_type=claim_type,
        evidence=[evidence],
    )


class DefaultResolutionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = DefaultResolutionPolicy()

    def test_demo_runtime_and_intent_are_diverged(self) -> None:
        root = Path(__file__).resolve().parents[1] / "examples" / "demo-auth"
        documents = load_sources(root)
        claims = extract_demo_claims(documents)
        verify_claims(claims, documents)

        resolution = self.policy.resolve(claims)

        self.assertEqual(resolution.status, ResolutionStatus.DIVERGED)
        self.assertEqual(
            [
                (claim.claim_type, claim.value)
                for claim in resolution.selected_claims
            ],
            [
                (ClaimType.ARCHITECTURE_INTENT, "OAuth2"),
                (ClaimType.RUNTIME_STATE, "JWT"),
            ],
        )
        self.assertEqual(resolution.conflicting_claims, [])
        self.assertIn(
            ReasonCode.INTENT_IMPLEMENTATION_DIVERGENCE,
            resolution.reason_codes,
        )
        self.assertEqual(resolution.policy, "default")
        self.assertEqual(resolution.policy_version, "0.1")

    def test_matching_verified_runtime_and_intent_are_resolved(self) -> None:
        claims = [
            _claim(
                "runtime-jwt",
                ClaimType.RUNTIME_STATE,
                "JWT",
                role=EvidenceRole.OBSERVED,
                verified=True,
            ),
            _claim(
                "intent-jwt",
                ClaimType.ARCHITECTURE_INTENT,
                "JWT",
                role=EvidenceRole.INTENDED,
                verified=True,
            ),
        ]

        resolution = self.policy.resolve(claims)

        self.assertEqual(resolution.status, ResolutionStatus.RESOLVED)
        self.assertEqual(len(resolution.selected_claims), 2)
        self.assertNotIn(
            ReasonCode.INTENT_IMPLEMENTATION_DIVERGENCE,
            resolution.reason_codes,
        )

    def test_conflicting_verified_runtime_claims_are_ambiguous(self) -> None:
        claims = [
            _claim(
                "runtime-jwt",
                ClaimType.RUNTIME_STATE,
                "JWT",
                role=EvidenceRole.OBSERVED,
                verified=True,
            ),
            _claim(
                "runtime-oauth",
                ClaimType.RUNTIME_STATE,
                "OAuth2",
                role=EvidenceRole.OBSERVED,
                verified=True,
            ),
        ]

        resolution = self.policy.resolve(claims)

        self.assertEqual(resolution.status, ResolutionStatus.AMBIGUOUS)
        self.assertEqual(resolution.selected_claims, [])
        self.assertEqual(
            {claim.value for claim in resolution.conflicting_claims},
            {"JWT", "OAuth2"},
        )
        self.assertIn(
            ReasonCode.CONFLICTING_AUTHORITATIVE_EVIDENCE,
            resolution.reason_codes,
        )

    def test_unverified_runtime_claim_remains_unverified(self) -> None:
        claim = _claim(
            "runtime-jwt",
            ClaimType.RUNTIME_STATE,
            "JWT",
            role=EvidenceRole.OBSERVED,
            verified=None,
        )

        resolution = self.policy.resolve([claim])

        self.assertEqual(resolution.status, ResolutionStatus.UNVERIFIED)
        self.assertEqual(resolution.selected_claims, [claim])
        self.assertIn(
            ReasonCode.INSUFFICIENT_EVIDENCE,
            resolution.reason_codes,
        )

    def test_single_verified_runtime_claim_is_resolved(self) -> None:
        claim = _claim(
            "runtime-jwt",
            ClaimType.RUNTIME_STATE,
            "JWT",
            role=EvidenceRole.OBSERVED,
            verified=True,
        )

        resolution = self.policy.resolve([claim])

        self.assertEqual(resolution.status, ResolutionStatus.RESOLVED)
        self.assertEqual(resolution.selected_claims, [claim])
        self.assertIn(
            ReasonCode.VERIFIED_EVIDENCE,
            resolution.reason_codes,
        )

    def test_observed_runtime_role_outranks_documented_role(self) -> None:
        observed = _claim(
            "runtime-observed",
            ClaimType.RUNTIME_STATE,
            "JWT",
            role=EvidenceRole.OBSERVED,
            verified=None,
        )
        documented = _claim(
            "runtime-documented",
            ClaimType.RUNTIME_STATE,
            "OAuth2",
            role=EvidenceRole.DOCUMENTED,
            verified=None,
        )

        resolution = self.policy.resolve([documented, observed])

        self.assertEqual(resolution.status, ResolutionStatus.UNVERIFIED)
        self.assertEqual(resolution.selected_claims, [observed])
        self.assertIn(
            ReasonCode.HIGHER_ROLE_PRIORITY,
            resolution.reason_codes,
        )

    def test_failed_evidence_is_not_successful_support(self) -> None:
        claim = _claim(
            "runtime-jwt",
            ClaimType.RUNTIME_STATE,
            "JWT",
            role=EvidenceRole.OBSERVED,
            verified=False,
        )

        resolution = self.policy.resolve([claim])

        self.assertEqual(resolution.status, ResolutionStatus.UNVERIFIED)
        self.assertEqual(resolution.selected_claims, [])
        self.assertNotIn(
            ReasonCode.VERIFIED_EVIDENCE,
            resolution.reason_codes,
        )

    def test_intended_role_outranks_verified_observed_intent_evidence(self) -> None:
        intended = _claim(
            "intent-oauth",
            ClaimType.ARCHITECTURE_INTENT,
            "OAuth2",
            role=EvidenceRole.INTENDED,
            verified=None,
        )
        observed = _claim(
            "intent-jwt",
            ClaimType.ARCHITECTURE_INTENT,
            "JWT",
            role=EvidenceRole.OBSERVED,
            verified=True,
        )

        resolution = self.policy.resolve([observed, intended])

        self.assertEqual(resolution.status, ResolutionStatus.UNVERIFIED)
        self.assertEqual(resolution.selected_claims, [intended])
        self.assertIn(
            ReasonCode.HIGHER_ROLE_PRIORITY,
            resolution.reason_codes,
        )

    def test_equally_preferred_unverified_runtime_claims_are_ambiguous(self) -> None:
        claims = [
            _claim(
                "runtime-jwt",
                ClaimType.RUNTIME_STATE,
                "JWT",
                role=EvidenceRole.OBSERVED,
                verified=None,
            ),
            _claim(
                "runtime-oauth",
                ClaimType.RUNTIME_STATE,
                "OAuth2",
                role=EvidenceRole.OBSERVED,
                verified=None,
            ),
        ]

        resolution = self.policy.resolve(claims)

        self.assertEqual(resolution.status, ResolutionStatus.AMBIGUOUS)
        self.assertEqual(len(resolution.conflicting_claims), 2)
        self.assertIn(
            ReasonCode.CONFLICTING_AUTHORITATIVE_EVIDENCE,
            resolution.reason_codes,
        )

    def test_equally_authoritative_intent_claims_are_ambiguous(self) -> None:
        claims = [
            _claim(
                "intent-jwt",
                ClaimType.ARCHITECTURE_INTENT,
                "JWT",
                role=EvidenceRole.INTENDED,
                verified=True,
            ),
            _claim(
                "intent-oauth",
                ClaimType.ARCHITECTURE_INTENT,
                "OAuth2",
                role=EvidenceRole.INTENDED,
                verified=True,
            ),
        ]

        resolution = self.policy.resolve(claims)

        self.assertEqual(resolution.status, ResolutionStatus.AMBIGUOUS)
        self.assertEqual(len(resolution.conflicting_claims), 2)

    def test_resolution_is_independent_of_claim_input_order(self) -> None:
        claims = [
            _claim(
                "runtime-jwt",
                ClaimType.RUNTIME_STATE,
                "JWT",
                role=EvidenceRole.OBSERVED,
                verified=True,
            ),
            _claim(
                "intent-oauth",
                ClaimType.ARCHITECTURE_INTENT,
                "OAuth2",
                role=EvidenceRole.INTENDED,
                verified=True,
            ),
        ]

        forward = self.policy.resolve(claims)
        reverse = self.policy.resolve(list(reversed(claims)))

        self.assertEqual(forward.status, reverse.status)
        self.assertEqual(
            [claim.id for claim in forward.selected_claims],
            [claim.id for claim in reverse.selected_claims],
        )
        self.assertEqual(
            [claim.id for claim in forward.conflicting_claims],
            [claim.id for claim in reverse.conflicting_claims],
        )
        self.assertEqual(forward.reason_codes, reverse.reason_codes)


if __name__ == "__main__":
    unittest.main()
