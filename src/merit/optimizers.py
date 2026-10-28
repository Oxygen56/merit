"""Reference optimizers used by the deterministic toy experiment."""

from __future__ import annotations

import random
from typing import Callable, Protocol

from .budget import Budget
from .model import AgentArtifact, HistoryEntry, OptimizeResult


EvaluateCandidate = Callable[[AgentArtifact], float]


class Optimizer(Protocol):
    name: str

    def optimize(
        self,
        start: AgentArtifact,
        initial_loss: float,
        budget: Budget,
        seed: int,
        evaluate: EvaluateCandidate,
    ) -> OptimizeResult:
        """Produce a successor without access to held-out tasks."""


def _accept_history(
    history: list[HistoryEntry],
    iteration: int,
    candidate: AgentArtifact,
    loss: float,
    accepted: bool,
    budget: Budget,
) -> None:
    history.append(
        HistoryEntry(
            iteration=iteration,
            candidate_artifact_id=candidate.artifact_id,
            candidate=candidate,
            dev_rmse=loss,
            accepted=accepted,
            remaining_budget=budget.remaining,
        )
    )


class RandomSearchOptimizer:
    name = "random_search"

    def __init__(self, step: float = 0.8) -> None:
        self.step = step

    def optimize(
        self,
        start: AgentArtifact,
        initial_loss: float,
        budget: Budget,
        seed: int,
        evaluate: EvaluateCandidate,
    ) -> OptimizeResult:
        rng = random.Random(seed)
        current = start
        current_loss = initial_loss
        history: list[HistoryEntry] = []
        iteration = 0
        while budget.remaining:
            progress = iteration / budget.limit
            scale = self.step * max(0.08, 1.0 - progress)
            values = tuple(v + rng.gauss(0.0, scale) for v in current.values)
            candidate = current.with_values(values)
            loss = evaluate(candidate)
            accepted = loss < current_loss
            if accepted:
                current, current_loss = candidate, loss
            _accept_history(history, iteration, candidate, loss, accepted, budget)
            iteration += 1
        return OptimizeResult(current, current_loss, tuple(history))


class CoordinateSearchOptimizer:
    name = "coordinate_search"

    def __init__(self, step: float = 1.25, decay: float = 0.92) -> None:
        self.step = step
        self.decay = decay

    def optimize(
        self,
        start: AgentArtifact,
        initial_loss: float,
        budget: Budget,
        seed: int,
        evaluate: EvaluateCandidate,
    ) -> OptimizeResult:
        del seed  # This deterministic optimizer still receives the paired seed.
        current = start
        current_loss = initial_loss
        history: list[HistoryEntry] = []
        dimension = len(start.values)
        iteration = 0
        while budget.remaining:
            coordinate = (iteration // 2) % dimension
            direction = 1.0 if iteration % 2 == 0 else -1.0
            cycle = iteration // (2 * dimension)
            magnitude = self.step * (self.decay**cycle)
            values = list(current.values)
            values[coordinate] += direction * magnitude
            candidate = current.with_values(tuple(values))
            loss = evaluate(candidate)
            accepted = loss < current_loss
            if accepted:
                current, current_loss = candidate, loss
            _accept_history(history, iteration, candidate, loss, accepted, budget)
            iteration += 1
        return OptimizeResult(current, current_loss, tuple(history))


class NoChangeOptimizer:
    """A control that preserves the initial artifact and spends no candidates."""

    name = "no_change"

    def optimize(
        self,
        start: AgentArtifact,
        initial_loss: float,
        budget: Budget,
        seed: int,
        evaluate: EvaluateCandidate,
    ) -> OptimizeResult:
        del budget, seed, evaluate
        return OptimizeResult(start, initial_loss, ())


class RandomEditOptimizer:
    """Apply random edits without using scores to select among them."""

    name = "random_edit"

    def __init__(self, step: float = 0.8) -> None:
        self.step = step

    def optimize(
        self,
        start: AgentArtifact,
        initial_loss: float,
        budget: Budget,
        seed: int,
        evaluate: EvaluateCandidate,
    ) -> OptimizeResult:
        rng = random.Random(seed)
        current = start
        current_loss = initial_loss
        history: list[HistoryEntry] = []
        iteration = 0
        while budget.remaining:
            progress = iteration / budget.limit
            scale = self.step * max(0.08, 1.0 - progress)
            values = tuple(v + rng.gauss(0.0, scale) for v in current.values)
            candidate = current.with_values(values)
            current_loss = evaluate(candidate)
            current = candidate
            _accept_history(history, iteration, candidate, current_loss, True, budget)
            iteration += 1
        return OptimizeResult(current, current_loss, tuple(history))


class MemoryResetOptimizer:
    """Reset the proposal state to the common start after every scored candidate."""

    name = "memory_reset"

    def __init__(self, step: float = 0.8) -> None:
        self.step = step

    def optimize(
        self,
        start: AgentArtifact,
        initial_loss: float,
        budget: Budget,
        seed: int,
        evaluate: EvaluateCandidate,
    ) -> OptimizeResult:
        rng = random.Random(seed)
        current = start
        current_loss = initial_loss
        history: list[HistoryEntry] = []
        iteration = 0
        while budget.remaining:
            progress = iteration / budget.limit
            scale = self.step * max(0.08, 1.0 - progress)
            values = tuple(v + rng.gauss(0.0, scale) for v in start.values)
            candidate = start.with_values(values)
            loss = evaluate(candidate)
            accepted = loss < current_loss
            if accepted:
                current, current_loss = candidate, loss
            _accept_history(history, iteration, candidate, loss, accepted, budget)
            iteration += 1
        return OptimizeResult(current, current_loss, tuple(history))
