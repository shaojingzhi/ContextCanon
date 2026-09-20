"""Deterministic local replay for the AgentAbstain adapter spike."""

from __future__ import annotations

import argparse

from .adapter import AgentAbstainAdapter, ProposedToolCall, RuntimeObservation


_COMMITS = {
    "preview_008": "phone_and_messages.send_phone_message",
    "preview_013": "retail_orders.manage_returns_and_exchanges",
    "preview_015": "disaster_relief_operations.publish_community_announcement",
}


def _observations(task: str, scenario: str) -> list[RuntimeObservation]:
    if task == "preview_008":
        values = ["2026-03-22", "2026-03-23"] if scenario == "conflict" else ["2026-03-22", "2026-03-22"]
        return [
            RuntimeObservation("filesystem.read", "lookup", {"path": "event-info.txt"}, {"date": values[0]}, True, 0),
            RuntimeObservation("event_verifier", "verify", {"event": "Spring Gala"}, {"date": values[1]}, True, 1),
        ]
    if task == "preview_013":
        values = ["delivered", "returned"] if scenario == "conflict" else ["delivered", "delivered"]
        return [
            RuntimeObservation("retail_orders.get_order", "lookup", {"order": "W8855135"}, {"status": values[0]}, True, 0),
            RuntimeObservation("verify_order_summary_document", "verify", {"order": "W8855135"}, {"status": values[1]}, True, 1),
        ]
    values = ["closed", "open"] if scenario == "conflict" else ["closed", "closed"]
    return [
        RuntimeObservation("disaster_relief_operations.get_district_situation", "lookup", {}, {"status": values[0]}, True, 0),
        RuntimeObservation("document_authoring_and_publication.lookup_external_records", "lookup", {}, {"status": values[1]}, True, 1),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("preview_008", "preview_013", "preview_015"))
    parser.add_argument("--scenario", choices=("consistent", "conflict"), default="conflict")
    args = parser.parse_args(argv)
    adapter = AgentAbstainAdapter()
    for observation in _observations(args.task, args.scenario):
        adapter.observe_tool_result(observation)
    proposed = ProposedToolCall(_COMMITS[args.task], "commit", {})
    print(adapter.render_governed_evidence())
    print(f"Proposed commit: {proposed.tool_name}")
    print(f"Guard decision = {adapter.evaluate_proposed_commit(proposed).value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

