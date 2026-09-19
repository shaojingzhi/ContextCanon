from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from contextcanon.assembly import assemble_context, source_revision
from contextcanon.core import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceRole,
    ReasonCode,
    Resolution,
    ResolutionStatus,
    SourceType,
)
from contextcanon.extraction import extract_demo_claims
from contextcanon.resolution import DefaultResolutionPolicy
from contextcanon.sources import load_sources
from contextcanon.verification import verify_claims


def _demo_resolution() -> Resolution:
    root = Path(__file__).resolve().parents[1] / "examples" / "demo-auth"
    documents = load_sources(root)
    claims = extract_demo_claims(documents)
    verify_claims(claims, documents)
    return DefaultResolutionPolicy().resolve(claims)


def _resolved_claim(claim_id: str, subject: str) -> Resolution:
    claim = Claim(
        id=claim_id,
        subject=subject,
        predicate="enabled",
        value=True,
        claim_type=ClaimType.RUNTIME_STATE,
    )
    return Resolution(
        status=ResolutionStatus.RESOLVED,
        selected_claims=[claim],
        reason_codes=[ReasonCode.VERIFIED_EVIDENCE],
        policy="default",
        policy_version="0.1",
    )


class ContextAssemblyTests(unittest.TestCase):
    def test_demo_assembly_preserves_selected_and_unresolved_knowledge(self) -> None:
        package = assemble_context(
            task="How does authentication currently work?",
            resolutions=[_demo_resolution()],
            source_revision="abc123",
            created_at="2026-09-20T01:23:45Z",
            policy_name="default",
            policy_version="0.1",
        )

        self.assertEqual(package.task, "How does authentication currently work?")
        self.assertEqual(package.source_revision, "abc123")
        self.assertEqual(package.policy_name, "default")
        self.assertEqual(package.policy_version, "0.1")
        self.assertEqual(
            [(item.claim.claim_type, item.claim.value) for item in package.items],
            [
                (ClaimType.ARCHITECTURE_INTENT, "OAuth2"),
                (ClaimType.RUNTIME_STATE, "JWT"),
            ],
        )
        self.assertTrue(
            all(
                item.resolution_status is ResolutionStatus.DIVERGED
                for item in package.items
            )
        )
        self.assertEqual(len(package.unresolved_conflicts), 1)
        self.assertEqual(
            package.unresolved_conflicts[0].status,
            ResolutionStatus.DIVERGED,
        )

    def test_failed_evidence_is_excluded_without_mutating_claim(self) -> None:
        verified = Evidence(
            id="verified",
            source_id="config/current.yaml",
            source_type=SourceType.YAML,
            location="auth.provider",
            role=EvidenceRole.OBSERVED,
            content="jwt",
            verified=True,
        )
        failed = Evidence(
            id="failed",
            source_id="config/stale.yaml",
            source_type=SourceType.YAML,
            location="auth.provider",
            role=EvidenceRole.OBSERVED,
            content="jwt",
            verified=False,
        )
        unverified = Evidence(
            id="unverified",
            source_id="docs/auth.md",
            source_type=SourceType.MARKDOWN,
            location="line:1",
            role=EvidenceRole.DOCUMENTED,
            content="Current protocol: JWT",
            verified=None,
        )
        claim = Claim(
            id="runtime-jwt",
            subject="auth",
            predicate="protocol",
            value="JWT",
            claim_type=ClaimType.RUNTIME_STATE,
            evidence=[failed, unverified, verified],
        )
        resolution = Resolution(
            status=ResolutionStatus.RESOLVED,
            selected_claims=[claim],
            reason_codes=[ReasonCode.VERIFIED_EVIDENCE],
        )

        package = assemble_context(
            task="task",
            resolutions=[resolution],
            source_revision="abc123",
            created_at="2026-09-20T01:23:45Z",
            policy_name="default",
            policy_version="0.1",
        )

        self.assertEqual(
            package.items[0].supporting_evidence,
            [verified, unverified],
        )
        self.assertEqual(claim.evidence, [failed, unverified, verified])

    def test_package_id_is_stable_and_excludes_created_at(self) -> None:
        resolution = _demo_resolution()
        arguments = {
            "task": "authentication task",
            "resolutions": [resolution],
            "source_revision": "abc123",
            "policy_name": "default",
            "policy_version": "0.1",
        }

        first = assemble_context(
            **arguments,
            created_at="2026-09-20T01:00:00Z",
        )
        later = assemble_context(
            **arguments,
            created_at="2026-09-20T02:00:00Z",
        )
        different_task = assemble_context(
            **{**arguments, "task": "different task"},
            created_at="2026-09-20T01:00:00Z",
        )
        different_revision = assemble_context(
            **{**arguments, "source_revision": "def456"},
            created_at="2026-09-20T01:00:00Z",
        )
        different_policy = assemble_context(
            **{**arguments, "policy_version": "0.2"},
            created_at="2026-09-20T01:00:00Z",
        )
        with_budget = assemble_context(
            **arguments,
            created_at="2026-09-20T01:00:00Z",
            token_budget=500,
        )
        resolved_state = assemble_context(
            **{
                **arguments,
                "resolutions": [
                    Resolution(
                        status=ResolutionStatus.RESOLVED,
                        selected_claims=list(resolution.selected_claims),
                        reason_codes=[ReasonCode.VERIFIED_EVIDENCE],
                    )
                ],
            },
            created_at="2026-09-20T01:00:00Z",
        )

        self.assertEqual(first.id, later.id)
        self.assertNotEqual(first.id, different_task.id)
        self.assertNotEqual(first.id, different_revision.id)
        self.assertNotEqual(first.id, different_policy.id)
        self.assertNotEqual(first.id, with_budget.id)
        self.assertNotEqual(first.id, resolved_state.id)

    def test_resolution_input_order_does_not_change_items_or_id(self) -> None:
        first_resolution = _resolved_claim("claim-z", "zeta")
        second_resolution = _resolved_claim("claim-a", "alpha")
        arguments = {
            "task": "task",
            "source_revision": "abc123",
            "created_at": "2026-09-20T01:23:45Z",
            "policy_name": "default",
            "policy_version": "0.1",
        }

        forward = assemble_context(
            **arguments,
            resolutions=[first_resolution, second_resolution],
        )
        reverse = assemble_context(
            **arguments,
            resolutions=[second_resolution, first_resolution],
        )

        self.assertEqual(
            [item.claim.id for item in forward.items],
            ["claim-a", "claim-z"],
        )
        self.assertEqual(
            [item.claim.id for item in forward.items],
            [item.claim.id for item in reverse.items],
        )
        self.assertEqual(forward.id, reverse.id)

    def test_empty_package_is_valid(self) -> None:
        package = assemble_context(
            task="task",
            resolutions=[],
            source_revision="unknown",
            created_at="2026-09-20T01:23:45Z",
            policy_name="default",
            policy_version="0.1",
        )

        self.assertEqual(package.items, [])
        self.assertEqual(package.unresolved_conflicts, [])
        self.assertTrue(package.id.startswith("ctx-"))

    def test_non_git_directory_has_unknown_source_revision(self) -> None:
        with TemporaryDirectory() as directory:
            revision = source_revision(Path(directory))

        self.assertEqual(revision, "unknown")


if __name__ == "__main__":
    unittest.main()

