"""Run exported M5.5 prompts against one model and save rich result records."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol
from urllib import error, request

from .scorer import validate_prediction


TEMPERATURE = 0
THINKING_MODE = "enabled"
REASONING_EFFORT = "high"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_RETRIES = 2
CONDITIONS = ("raw", "contextcanon")
ORDERINGS = ("canonical", "reverse", "permutation")


class ModelClient(Protocol):
    def __call__(self, prompt: str, model: str, temperature: int) -> "ModelResponse":
        ...


@dataclass(frozen=True)
class ModelResponse:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class TransportError(RuntimeError):
    """A provider failure that is safe to retry."""


class OpenAICompatibleClient:
    """Small stdlib client for an OpenAI-compatible chat completions endpoint."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def __call__(self, prompt: str, model: str, temperature: int) -> ModelResponse:
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "thinking": {"type": THINKING_MODE},
                "reasoning_effort": REASONING_EFFORT,
            }
        ).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500 or exc.code in {408, 409}:
                raise TransportError(f"provider HTTP {exc.code}") from exc
            raise RuntimeError(f"provider HTTP {exc.code}") from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            raise TransportError(str(exc)) from exc
        try:
            text = body["choices"][0]["message"]["content"]
            usage = body.get("usage") or {}
            if not isinstance(text, str):
                raise TypeError("provider response content is not text")
            return ModelResponse(
                text=text,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("provider response has no chat completion") from exc


def discover_prompts(prompts: Path, condition: str = "all") -> list[tuple[str, str, Path]]:
    conditions = CONDITIONS if condition == "all" else (condition,)
    discovered: list[tuple[str, str, Path]] = []
    for selected in conditions:
        for path in sorted((prompts / selected).glob("*.txt")):
            parts = path.stem.rsplit("__", 1)
            if len(parts) != 2 or parts[1] not in ORDERINGS:
                raise ValueError(f"invalid prompt filename: {path.name}")
            discovered.append((selected, parts[0], path))
    if not discovered:
        raise ValueError(f"no exported prompts found under {prompts}")
    return discovered


def expected_request_count(prompts: Iterable[tuple[str, str, Path]]) -> int:
    return sum(1 for _ in prompts)


def _result_path(results: Path, run_id: str, condition: str, case_id: str, ordering: str) -> Path:
    return results / run_id / condition / f"{case_id}__{ordering}.json"


def _invoke_with_retries(client: ModelClient, prompt: str, model: str, retries: int) -> ModelResponse:
    attempts = retries + 1
    for attempt in range(attempts):
        try:
            return client(prompt, model, TEMPERATURE)
        except TransportError:
            if attempt == attempts - 1:
                raise
    raise AssertionError("unreachable")


def _run_one(
    client: ModelClient,
    prompt_path: Path,
    condition: str,
    case_id: str,
    ordering: str,
    model: str,
    retries: int,
) -> dict[str, object]:
    prompt = prompt_path.read_text(encoding="utf-8")
    started = time.perf_counter()
    try:
        response = _invoke_with_retries(client, prompt, model, retries)
    except Exception as exc:
        return {
            "case_id": case_id,
            "condition": condition,
            "ordering": ordering,
            "model": model,
            "temperature": TEMPERATURE,
            "thinking": THINKING_MODE,
            "reasoning_effort": REASONING_EFFORT,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "input_tokens": None,
            "output_tokens": None,
            "raw_response": None,
            "prediction": None,
            "parse_error": f"transport error: {exc}",
        }
    record: dict[str, object] = {
        "case_id": case_id,
        "condition": condition,
        "ordering": ordering,
        "model": model,
        "temperature": TEMPERATURE,
        "thinking": THINKING_MODE,
        "reasoning_effort": REASONING_EFFORT,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "raw_response": response.text,
        "prediction": None,
        "parse_error": None,
    }
    try:
        prediction = json.loads(response.text)
        validate_prediction(prediction)
    except (json.JSONDecodeError, ValueError) as exc:
        record["parse_error"] = str(exc)
    else:
        record["prediction"] = prediction
    return record


def run(
    prompts: Path,
    results: Path,
    model: str,
    run_id: str,
    condition: str = "all",
    client: ModelClient | None = None,
    retries: int = DEFAULT_RETRIES,
    dry_run: bool = False,
) -> int:
    discovered = discover_prompts(prompts, condition)
    print(f"expected requests: {len(discovered)}")
    run_root = results / run_id
    if run_root.exists():
        raise FileExistsError(f"result run already exists: {run_root}")
    if dry_run:
        for selected, case_id, path in discovered:
            ordering = path.stem.rsplit("__", 1)[1]
            print(_result_path(results, run_id, selected, case_id, ordering))
        return len(discovered)
    if client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for a live run")
        client = OpenAICompatibleClient(api_key, os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    run_root.mkdir(parents=True)
    for selected, case_id, path in discovered:
        ordering = path.stem.rsplit("__", 1)[1]
        target = _result_path(results, run_id, selected, case_id, ordering)
        target.parent.mkdir(parents=True, exist_ok=True)
        record = _run_one(client, path, selected, case_id, ordering, model, retries)
        record["run_id"] = run_id
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(discovered)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exported M5.5 prompts against one model")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", choices=(*CONDITIONS, "all"), default="all")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(
        prompts=args.prompts,
        results=args.results,
        model=args.model,
        run_id=args.run_id,
        condition=args.condition,
        retries=args.retries,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
