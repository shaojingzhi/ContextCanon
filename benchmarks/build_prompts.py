"""Build deterministic Raw Context and ContextCanon benchmark prompts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from contextcanon.assembly import assemble_context, source_revision
from contextcanon.extraction import extract_demo_claims
from contextcanon.renderers import MarkdownRenderer
from contextcanon.resolution import DefaultResolutionPolicy
from contextcanon.sources import load_sources
from contextcanon.verification import verify_claims


SHARED_INSTRUCTIONS = """Use only supplied repository information.
Do not guess. Distinguish current implementation from future architecture intent.
Represent conflicting or insufficient knowledge explicitly.
Cite repository paths supporting the answer.
Return only valid JSON using this schema:
{
  "current": "JWT or OAuth2 or null",
  "target": "JWT or OAuth2 or null",
  "status": "RESOLVED | DIVERGED | AMBIGUOUS | UNVERIFIED",
  "evidence": {"current": [], "target": [], "conflict": []}
}
"""


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    question: str
    ground_truth: dict[str, object]
    path: Path


def _case_root() -> Path:
    return Path(__file__).parent / "cases"


def load_cases(root: Path | None = None) -> list[BenchmarkCase]:
    case_root = _case_root() if root is None else root
    cases: list[BenchmarkCase] = []
    for metadata_path in sorted(case_root.glob("*/case.yaml")):
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid benchmark metadata: {metadata_path}")
        case_id = metadata.get("id")
        question = metadata.get("question")
        ground_truth = metadata.get("ground_truth")
        if not isinstance(case_id, str) or not isinstance(question, str):
            raise ValueError(f"missing case id/question: {metadata_path}")
        if not isinstance(ground_truth, dict):
            raise ValueError(f"missing ground truth: {metadata_path}")
        cases.append(
            BenchmarkCase(
                id=case_id,
                question=" ".join(question.split()),
                ground_truth=ground_truth,
                path=metadata_path.parent,
            )
        )
    if len(cases) != 15:
        raise ValueError(f"expected 15 benchmark cases, found {len(cases)}")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("benchmark case IDs must be unique")
    return cases


def _raw_material(case: BenchmarkCase, ordering: str = "canonical") -> str:
    documents = load_sources(case.path)
    entries = [
        (document.path, document.content)
        for document in documents
        if document.path != "case.yaml"
    ]
    if ordering == "reverse":
        entries.reverse()
    elif ordering == "permutation":
        entries = entries[::2] + entries[1::2]
    blocks = []
    for path, content in entries:
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True)
        blocks.append(f"--- {path} ---\n{content.rstrip()}")
    return "\n\n".join(blocks)


def _context_rendering(case: BenchmarkCase) -> str:
    documents = load_sources(case.path)
    claims = extract_demo_claims(documents)
    verify_claims(claims, documents)
    policy = DefaultResolutionPolicy()
    groups: dict[tuple[str, str], list] = {}
    for claim in claims:
        groups.setdefault((claim.subject, claim.predicate), []).append(claim)
    resolutions = [policy.resolve(groups[key]) for key in sorted(groups)]
    package = assemble_context(
        task=case.question,
        resolutions=resolutions,
        source_revision=source_revision(case.path),
        created_at="1970-01-01T00:00:00Z",
        policy_name=policy.name,
        policy_version=policy.version,
    )
    return MarkdownRenderer().render(package)


def build_prompt(case: BenchmarkCase, condition: str, ordering: str = "canonical") -> str:
    if condition not in {"raw", "contextcanon"}:
        raise ValueError("condition must be raw or contextcanon")
    prompt = f"{SHARED_INSTRUCTIONS}\n\nRepository source materials:\n{_raw_material(case, ordering)}"
    if condition == "contextcanon":
        prompt += (
            "\n\nAdditional structured context produced by ContextCanon:\n"
            + _context_rendering(case)
        )
    return f"{prompt}\n\nQuestion:\n{case.question}\n"


def build_all_prompts(cases: Iterable[BenchmarkCase]) -> dict[str, dict[str, str]]:
    return {
        case.id: {
            "raw": build_prompt(case, "raw"),
            "contextcanon": build_prompt(case, "contextcanon"),
        }
        for case in cases
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export M5.5 benchmark prompts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for case in load_cases():
        for condition in ("raw", "contextcanon"):
            target = args.output / condition / f"{case.id}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(build_prompt(case, condition), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

