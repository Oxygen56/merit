"""Adapters for plugging ordinary Python improvement methods into Merit."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from .budget import Budget
from .model import AgentArtifact, HistoryEntry, OptimizeResult
from .optimizers import EvaluateCandidate


@dataclass(frozen=True)
class ProposalContext:
    """The only state exposed to a Python proposal callback."""

    current: AgentArtifact
    current_dev_rmse: float
    iteration: int
    remaining_budget: int
    history: tuple[HistoryEntry, ...]
    rng: random.Random = field(compare=False, repr=False)


ProposeCallback = Callable[[ProposalContext], AgentArtifact]
AcceptCallback = Callable[[float, float], bool]


def _greedy_accept(current_loss: float, candidate_loss: float) -> bool:
    return candidate_loss < current_loss


class PythonOptimizerAdapter:
    """Run a proposal callback with shared budget, scoring and evidence handling."""

    def __init__(self, name: str, propose: ProposeCallback, accept: AcceptCallback | None = None) -> None:
        if not name.strip():
            raise ValueError("Optimizer name must not be empty")
        if not callable(propose):
            raise TypeError("propose must be callable")
        if accept is not None and not callable(accept):
            raise TypeError("accept must be callable")
        self.name = name
        self._propose = propose
        self._accept = accept or _greedy_accept

    def optimize(
        self,
        start: AgentArtifact,
        initial_loss: float,
        budget: Budget,
        seed: int,
        evaluate: EvaluateCandidate,
    ) -> OptimizeResult:
        current = start
        current_loss = initial_loss
        history: list[HistoryEntry] = []
        rng = random.Random(seed)
        iteration = 0
        while budget.remaining:
            context = ProposalContext(
                current=current,
                current_dev_rmse=current_loss,
                iteration=iteration,
                remaining_budget=budget.remaining,
                history=tuple(history),
                rng=rng,
            )
            candidate = self._propose(context)
            if not isinstance(candidate, AgentArtifact):
                raise TypeError("Proposal callback must return an AgentArtifact")
            if candidate.parent != current.artifact_id:
                raise ValueError("Proposal must be a direct child of the current artifact; use current.with_values()")
            loss = evaluate(candidate)
            accepted = self._accept(current_loss, loss)
            if accepted:
                current, current_loss = candidate, loss
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
            iteration += 1
        return OptimizeResult(current, current_loss, tuple(history))
