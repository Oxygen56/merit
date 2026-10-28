"""Task, artifact, execution and cost contracts for future real-agent adapters."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .budget import Budget


AGENT_PROTOCOL_VERSION = "merit-agent-contract-v0.1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, init=False)
class AgentSpec:
    """An immutable, JSON-configurable agent artifact with content lineage."""

    instructions: str
    config_json: str
    parent: str | None
    artifact_id: str = field(init=False)

    def __init__(
        self,
        instructions: str,
        config: Mapping[str, Any] | None = None,
        parent: str | None = None,
    ) -> None:
        if not isinstance(instructions, str):
            raise TypeError("Agent instructions must be a string")
        if config is None:
            config = {}
        if not isinstance(config, Mapping) or any(not isinstance(key, str) for key in config):
            raise TypeError("Agent config must be a mapping with string keys")
        config_json = _canonical_json(dict(config))
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "config_json", config_json)
        object.__setattr__(self, "parent", parent)
        artifact_id = _digest(
            {
                "instructions": instructions,
                "config": json.loads(config_json),
                "parent": parent,
                "version": 1,
            }
        )
        object.__setattr__(self, "artifact_id", artifact_id)

    @property
    def config(self) -> dict[str, Any]:
        """Return a fresh JSON value so callers cannot mutate the artifact."""

        return json.loads(self.config_json)

    def evolve(
        self,
        *,
        instructions: str | None = None,
        config_updates: Mapping[str, Any] | None = None,
    ) -> "AgentSpec":
        config = self.config
        config.update(config_updates or {})
        return AgentSpec(
            instructions=self.instructions if instructions is None else instructions,
            config=config,
            parent=self.artifact_id,
        )


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    prompt: str
    expected: str

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("Agent task_id must not be empty")
        if not isinstance(self.prompt, str) or not isinstance(self.expected, str):
            raise TypeError("Agent task prompt and expected output must be strings")


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    tokens_reported: bool = True
    cost_reported: bool = True
    cost_is_estimate: bool = False

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("Token counts must be non-negative")
        if not math.isfinite(self.cost_usd) or self.cost_usd < 0:
            raise ValueError("Cost must be finite and non-negative")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("Latency must be finite and non-negative")

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_ms=self.latency_ms + other.latency_ms,
            tokens_reported=self.tokens_reported and other.tokens_reported,
            cost_reported=self.cost_reported and other.cost_reported,
            cost_is_estimate=self.cost_is_estimate or other.cost_is_estimate,
        )


@dataclass(frozen=True)
class AgentResponse:
    output: str
    usage: Usage = Usage()


@dataclass(frozen=True)
class AgentTaskResult:
    task_id: str
    output: str
    correct: bool
    usage: Usage


@dataclass(frozen=True)
class AgentEvaluation:
    accuracy: float
    usage: Usage
    results: tuple[AgentTaskResult, ...]


class AgentExecutor(Protocol):
    """The boundary a real model, workflow or agent runtime must implement."""

    name: str

    def invoke(self, agent: AgentSpec, task: AgentTask) -> AgentResponse:
        """Execute one task without revealing other held-out tasks."""


@dataclass(frozen=True)
class AgentProposal:
    """A candidate artifact plus the independent cost of producing it."""

    artifact: AgentSpec
    usage: Usage = Usage()


@dataclass(frozen=True)
class AgentHistoryEntry:
    iteration: int
    candidate: AgentSpec
    evaluation: AgentEvaluation
    accepted: bool
    remaining_budget: int
    proposer_usage: Usage = Usage()
    selection_evaluation: AgentEvaluation | None = None


@dataclass(frozen=True)
class AgentProposalContext:
    current: AgentSpec
    current_evaluation: AgentEvaluation
    train_tasks: tuple[AgentTask, ...]
    iteration: int
    remaining_budget: int
    history: tuple[AgentHistoryEntry, ...]
    rng: random.Random = field(compare=False, repr=False)


AgentProposeCallback = Callable[[AgentProposalContext], AgentSpec | AgentProposal]
AgentAcceptCallback = Callable[[float, float], bool]


@dataclass(frozen=True)
class AgentOptimizationResult:
    successor: AgentSpec
    evaluation: AgentEvaluation
    history: tuple[AgentHistoryEntry, ...]
    details: dict[str, Any] = field(default_factory=dict)
    additional_proposer_usage: Usage = Usage()
    initial_selection_evaluation: AgentEvaluation | None = None
    successor_selection_evaluation: AgentEvaluation | None = None


class AgentOptimizerAdapter:
    """Adapt a proposal callback with separate evaluated-agent and proposer costs."""

    def __init__(
        self,
        name: str,
        propose: AgentProposeCallback,
        accept: AgentAcceptCallback | None = None,
        proposer_source: str = "python",
    ) -> None:
        if not name.strip():
            raise ValueError("Optimizer name must not be empty")
        if not callable(propose):
            raise TypeError("propose must be callable")
        if accept is not None and not callable(accept):
            raise TypeError("accept must be callable")
        self.name = name
        self._propose = propose
        self._accept = accept or (lambda current, candidate: candidate > current)
        self.proposer_source = proposer_source

    def optimize(
        self,
        start: AgentSpec,
        train_tasks: tuple[AgentTask, ...],
        initial_evaluation: AgentEvaluation,
        budget: Budget,
        seed: int,
        evaluate: Callable[[AgentSpec], AgentEvaluation],
    ) -> AgentOptimizationResult:
        current = start
        current_evaluation = initial_evaluation
        history: list[AgentHistoryEntry] = []
        rng = random.Random(seed)
        iteration = 0
        while budget.remaining:
            context = AgentProposalContext(
                current=current,
                current_evaluation=current_evaluation,
                train_tasks=train_tasks,
                iteration=iteration,
                remaining_budget=budget.remaining,
                history=tuple(history),
                rng=rng,
            )
            proposal = self._propose(context)
            if isinstance(proposal, AgentProposal):
                candidate = proposal.artifact
                proposer_usage = proposal.usage
            elif isinstance(proposal, AgentSpec):
                candidate = proposal
                proposer_usage = Usage()
            else:
                raise TypeError("Agent proposal callback must return an AgentSpec or AgentProposal")
            if candidate.parent != current.artifact_id:
                raise ValueError("Agent proposal must be a direct child of the current artifact")
            evaluation = evaluate(candidate)
            accepted = self._accept(current_evaluation.accuracy, evaluation.accuracy)
            if accepted:
                current, current_evaluation = candidate, evaluation
            history.append(
                AgentHistoryEntry(
                    iteration=iteration,
                    candidate=candidate,
                    evaluation=evaluation,
                    accepted=accepted,
                    remaining_budget=budget.remaining,
                    proposer_usage=proposer_usage,
                )
            )
            iteration += 1
        return AgentOptimizationResult(current, current_evaluation, tuple(history))


@dataclass(frozen=True)
class AgentContractManifest:
    experiment: str
    executor: str
    optimizer: str
    seed: int
    candidate_budget: int
    start_artifact_id: str
    train_task_digest: str
    heldout_task_digest: str
    protocol_version: str = AGENT_PROTOCOL_VERSION
    proposer_source: str = "python"
    validation_task_digest: str | None = None

    def __post_init__(self) -> None:
        if self.candidate_budget < 1:
            raise ValueError("candidate_budget must be positive")

    @property
    def digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class AgentContractExperiment:
    manifest: AgentContractManifest
    start: AgentSpec
    successor: AgentSpec
    train_tasks: tuple[AgentTask, ...]
    heldout_tasks: tuple[AgentTask, ...]
    initial_dev: AgentEvaluation
    successor_dev: AgentEvaluation
    baseline_heldout: AgentEvaluation
    successor_heldout: AgentEvaluation
    history: tuple[AgentHistoryEntry, ...]
    summary: dict[str, Any]
    initial_validation: AgentEvaluation | None = None
    successor_validation: AgentEvaluation | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tasks_digest(tasks: Sequence[AgentTask]) -> str:
    return _digest([asdict(task) for task in tasks])


def evaluate_agent(
    executor: AgentExecutor,
    agent: AgentSpec,
    tasks: Sequence[AgentTask],
) -> AgentEvaluation:
    if not tasks:
        raise ValueError("Agent evaluation requires at least one task")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("Agent task IDs must be unique within a split")
    results = []
    total_usage = Usage()
    for task in tasks:
        response = executor.invoke(agent, task)
        if not isinstance(response, AgentResponse):
            raise TypeError("Agent executor must return an AgentResponse")
        usage = response.usage
        results.append(
            AgentTaskResult(
                task_id=task.task_id,
                output=response.output,
                correct=response.output == task.expected,
                usage=usage,
            )
        )
        total_usage = total_usage + usage
    accuracy = sum(result.correct for result in results) / len(results)
    return AgentEvaluation(accuracy, total_usage, tuple(results))


def _total_usage(evaluations: Sequence[AgentEvaluation]) -> Usage:
    total = Usage()
    for evaluation in evaluations:
        total = total + evaluation.usage
    return total


def run_agent_contract(
    manifest: AgentContractManifest,
    start: AgentSpec,
    train_tasks: Sequence[AgentTask],
    heldout_tasks: Sequence[AgentTask],
    optimizer: AgentOptimizerAdapter,
    executor: AgentExecutor,
    validation_tasks: Sequence[AgentTask] = (),
) -> AgentContractExperiment:
    """Optimize on train tasks, select candidates separately, then evaluate the successor."""

    train_tasks = tuple(train_tasks)
    heldout_tasks = tuple(heldout_tasks)
    validation_tasks = tuple(validation_tasks)
    if manifest.optimizer != optimizer.name or manifest.executor != executor.name:
        raise ValueError("Manifest names do not match the optimizer and executor")
    if manifest.proposer_source != optimizer.proposer_source:
        raise ValueError("Manifest proposer source does not match the registered optimizer")
    if manifest.start_artifact_id != start.artifact_id:
        raise ValueError("Manifest start artifact does not match the supplied start")
    if manifest.train_task_digest != tasks_digest(train_tasks):
        raise ValueError("Manifest train task digest does not match the supplied tasks")
    if manifest.heldout_task_digest != tasks_digest(heldout_tasks):
        raise ValueError("Manifest held-out task digest does not match the supplied tasks")
    validation_digest = tasks_digest(validation_tasks) if validation_tasks else None
    if manifest.validation_task_digest != validation_digest:
        raise ValueError("Manifest validation task digest does not match the supplied tasks")
    if hasattr(optimizer, "validation_tasks") and tuple(optimizer.validation_tasks) != validation_tasks:
        raise ValueError("Optimizer validation tasks do not match the manifest")
    if {task.task_id for task in train_tasks}.intersection(task.task_id for task in heldout_tasks):
        raise ValueError("Train and held-out task IDs must be disjoint")
    if {task.prompt for task in train_tasks}.intersection(task.prompt for task in heldout_tasks):
        raise ValueError("Train and held-out task inputs must be disjoint")
    for name, tasks in (("train", train_tasks), ("held-out", heldout_tasks)):
        if {task.task_id for task in validation_tasks}.intersection(task.task_id for task in tasks):
            raise ValueError(f"Validation and {name} task IDs must be disjoint")
        if {task.prompt for task in validation_tasks}.intersection(task.prompt for task in tasks):
            raise ValueError(f"Validation and {name} task inputs must be disjoint")

    initial_dev = evaluate_agent(executor, start, train_tasks)
    baseline_heldout = evaluate_agent(executor, start, heldout_tasks)
    budget = Budget(manifest.candidate_budget)

    def evaluate(candidate: AgentSpec) -> AgentEvaluation:
        budget.consume()
        return evaluate_agent(executor, candidate, train_tasks)

    result = optimizer.optimize(start, train_tasks, initial_dev, budget, manifest.seed, evaluate)
    successor_heldout = evaluate_agent(executor, result.successor, heldout_tasks)
    dev_evaluations = [initial_dev, *(entry.evaluation for entry in result.history)]
    selection_evaluations = [
        evaluation
        for evaluation in (
            result.initial_selection_evaluation,
            *(entry.selection_evaluation for entry in result.history),
        )
        if evaluation is not None
    ]
    heldout_evaluations = [baseline_heldout, successor_heldout]
    dev_usage = _total_usage([*dev_evaluations, *selection_evaluations])
    heldout_usage = _total_usage(heldout_evaluations)
    total_usage = dev_usage + heldout_usage
    proposer_usage = result.additional_proposer_usage
    for entry in result.history:
        proposer_usage = proposer_usage + entry.proposer_usage
    combined_usage = total_usage + proposer_usage
    summary = {
        "metric": "exact_match_accuracy",
        "initial_dev_accuracy": initial_dev.accuracy,
        "successor_dev_accuracy": result.evaluation.accuracy,
        "baseline_heldout_accuracy": baseline_heldout.accuracy,
        "successor_heldout_accuracy": successor_heldout.accuracy,
        "heldout_accuracy_improvement": successor_heldout.accuracy - baseline_heldout.accuracy,
        "candidate_evaluations": budget.used,
        "accepted_candidates": sum(entry.accepted for entry in result.history),
        "proposal_calls": len(result.history),
        "proposer_source": optimizer.proposer_source,
        "optimizer_details": result.details,
        "dev_agent_invocations": (
            len(dev_evaluations) * len(train_tasks)
            + sum(len(evaluation.results) for evaluation in selection_evaluations)
        ),
        "heldout_agent_invocations": len(heldout_evaluations) * len(heldout_tasks),
        "dev_input_tokens": dev_usage.input_tokens,
        "dev_output_tokens": dev_usage.output_tokens,
        "heldout_input_tokens": heldout_usage.input_tokens,
        "heldout_output_tokens": heldout_usage.output_tokens,
        "total_input_tokens": total_usage.input_tokens,
        "total_output_tokens": total_usage.output_tokens,
        "total_cost_usd": total_usage.cost_usd,
        "total_latency_ms": total_usage.latency_ms,
        "total_tokens_reported": total_usage.tokens_reported,
        "total_cost_reported": total_usage.cost_reported,
        "total_cost_is_estimate": total_usage.cost_is_estimate,
        "proposer_input_tokens": proposer_usage.input_tokens,
        "proposer_output_tokens": proposer_usage.output_tokens,
        "proposer_cost_usd": proposer_usage.cost_usd,
        "proposer_latency_ms": proposer_usage.latency_ms,
        "proposer_tokens_reported": proposer_usage.tokens_reported,
        "proposer_cost_reported": proposer_usage.cost_reported,
        "proposer_cost_is_estimate": proposer_usage.cost_is_estimate,
        "combined_total_input_tokens": combined_usage.input_tokens,
        "combined_total_output_tokens": combined_usage.output_tokens,
        "combined_total_cost_usd": combined_usage.cost_usd,
        "combined_total_latency_ms": combined_usage.latency_ms,
        "combined_total_tokens_reported": combined_usage.tokens_reported,
        "combined_total_cost_reported": combined_usage.cost_reported,
        "accounting": "total_* covers the evaluated agent; proposer_* and combined_total_* are separate",
        "scope": "Agent contract validation; claims depend on the registered executor",
    }
    if result.initial_selection_evaluation is not None:
        summary["initial_validation_accuracy"] = result.initial_selection_evaluation.accuracy
    if result.successor_selection_evaluation is not None:
        summary["successor_validation_accuracy"] = result.successor_selection_evaluation.accuracy
    return AgentContractExperiment(
        manifest=manifest,
        start=start,
        successor=result.successor,
        train_tasks=train_tasks,
        heldout_tasks=heldout_tasks,
        initial_dev=initial_dev,
        successor_dev=result.evaluation,
        baseline_heldout=baseline_heldout,
        successor_heldout=successor_heldout,
        history=result.history,
        summary=summary,
        initial_validation=result.initial_selection_evaluation,
        successor_validation=result.successor_selection_evaluation,
    )
