"""Task-local limits for synchronous semantic inference."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import TypeVar


T = TypeVar("T")


class SemanticDeadlineExceeded(TimeoutError):
    """A semantic operation exceeded its whole-operation deadline."""


@dataclass(slots=True)
class SemanticBudget:
    """One fail-safe budget shared by all semantic stages in a task."""

    max_requests: int = 12
    max_total_seconds: float = 30.0
    per_request_deadline_seconds: float = 5.0
    clock: Callable[[], float] = monotonic
    requests_started: int = 0
    requests_completed: int = 0
    requests_timed_out: int = 0
    requests_skipped_budget: int = 0
    total_semantic_wall_ms: int = 0
    time_by_stage_ms: dict[str, int] = field(default_factory=dict)
    requests_by_stage: dict[str, int] = field(default_factory=dict)
    governance_incomplete: bool = False
    _started_at: float = field(init=False)

    def __post_init__(self) -> None:
        if self.max_requests < 0:
            raise ValueError("max_requests must not be negative")
        if self.max_total_seconds <= 0 or self.per_request_deadline_seconds <= 0:
            raise ValueError("semantic deadlines must be positive")
        self._started_at = self.clock()

    @property
    def remaining_request_budget(self) -> int:
        return max(0, self.max_requests - self.requests_started)

    @property
    def remaining_time_budget(self) -> float:
        return max(0.0, self.max_total_seconds - (self.clock() - self._started_at))

    def invoke(self, stage: str, operation: Callable[[], T]) -> T:
        if self.remaining_request_budget <= 0 or self.remaining_time_budget <= 0:
            self.requests_skipped_budget += 1
            self.governance_incomplete = True
            raise SemanticDeadlineExceeded("semantic budget exhausted")
        deadline = min(self.per_request_deadline_seconds, self.remaining_time_budget)
        if deadline <= 0:
            self.requests_skipped_budget += 1
            self.governance_incomplete = True
            raise SemanticDeadlineExceeded("semantic time budget exhausted")
        self.requests_started += 1
        self.requests_by_stage[stage] = self.requests_by_stage.get(stage, 0) + 1
        started = self.clock()
        try:
            with _operation_deadline(deadline):
                value = operation()
        except SemanticDeadlineExceeded:
            self.requests_timed_out += 1
            self.governance_incomplete = True
            raise
        except Exception:
            self.governance_incomplete = True
            raise
        else:
            self.requests_completed += 1
            return value
        finally:
            elapsed_ms = max(0, round((self.clock() - started) * 1000))
            self.total_semantic_wall_ms += elapsed_ms
            self.time_by_stage_ms[stage] = self.time_by_stage_ms.get(stage, 0) + elapsed_ms

    def diagnostics(self) -> dict[str, object]:
        return {
            "semantic_requests_total": self.requests_started,
            "semantic_requests_by_stage": dict(sorted(self.requests_by_stage.items())),
            "semantic_time_ms_total": self.total_semantic_wall_ms,
            "semantic_time_ms_by_stage": dict(sorted(self.time_by_stage_ms.items())),
            "semantic_timeouts": self.requests_timed_out,
            "semantic_budget_exhausted": self.requests_skipped_budget,
            "remaining_request_budget": self.remaining_request_budget,
            "remaining_time_budget": self.remaining_time_budget,
            "governance_incomplete": self.governance_incomplete,
        }


class BudgetedSemanticClient:
    """Stage-labelled adapter over the existing synchronous model boundary."""

    def __init__(self, client: object, budget: SemanticBudget, stage: str) -> None:
        self.client = client
        self.budget = budget
        self.stage = stage

    def complete(self, prompt: str, *, model: str) -> object:
        return self.budget.invoke(
            self.stage,
            lambda: self.client.complete(prompt, model=model),  # type: ignore[attr-defined]
        )


class _operation_deadline:
    """Interrupt a main-thread synchronous operation without leaving a worker behind."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.previous_handler: object | None = None
        self.previous_timer: tuple[float, float] | None = None
        self.enabled = False

    def __enter__(self) -> None:
        # setitimer interrupts the operation itself; unlike a worker-thread
        # timeout, it cannot leave a blocked semantic request running behind.
        if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
            raise SemanticDeadlineExceeded("hard semantic deadline unavailable outside main thread")
        self.previous_handler = signal.getsignal(signal.SIGALRM)

        def _raise_deadline(_signum: int, _frame: object) -> None:
            raise SemanticDeadlineExceeded("semantic operation deadline exceeded")

        signal.signal(signal.SIGALRM, _raise_deadline)
        self.previous_timer = signal.setitimer(signal.ITIMER_REAL, self.seconds)
        self.enabled = True

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
            assert self.previous_handler is not None
            signal.signal(signal.SIGALRM, self.previous_handler)
            if self.previous_timer and self.previous_timer != (0.0, 0.0):
                signal.setitimer(signal.ITIMER_REAL, *self.previous_timer)
        return False


__all__ = ["BudgetedSemanticClient", "SemanticBudget", "SemanticDeadlineExceeded"]
