"""Paired execution with shared starts, paired tasks and hard budget caps."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Callable, Sequence

from .budget import Budget
from .model import (
    AgentArtifact,
    ArmResult,
    Experiment,
    ExperimentManifest,
    PairedRun,
    Task,
)
from .optimizers import Optimizer
from .stats import bootstrap_mean_ci, sign_flip_pvalue


StartFactory = Callable[[int], AgentArtifact]
TaskFactory = Callable[[int, int, int, int], tuple[Sequence[Task], Sequence[Task]]]


def rmse(agent: AgentArtifact, tasks: Sequence[Task]) -> float:
    if not tasks:
        raise ValueError("RMSE requires at least one task")
    squared_errors = []
    for task in tasks:
        if len(task.features) != len(agent.values):
            raise ValueError("Task feature dimension does not match agent dimension")
        prediction = sum(value * feature for value, feature in zip(agent.values, task.features))
        squared_errors.append((prediction - task.target) ** 2)
    return statistics.fmean(squared_errors) ** 0.5


def tasks_digest(tasks: Sequence[Task]) -> str:
    payload = [(task.features, task.target) for task in tasks]
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PairRunner:
    def __init__(
        self,
        baseline: Optimizer,
        challenger: Optimizer,
        start_factory: StartFactory,
        task_factory: TaskFactory,
        controls: Sequence[Optimizer] = (),
    ) -> None:
        self.baseline = baseline
        self.challenger = challenger
        self.start_factory = start_factory
        self.task_factory = task_factory
        self.controls = tuple(controls)

    def _run_arm(
        self,
        optimizer: Optimizer,
        start: AgentArtifact,
        train_tasks: Sequence[Task],
        heldout_tasks: Sequence[Task],
        initial_dev_rmse: float,
        initial_heldout_rmse: float,
        budget_limit: int,
        seed: int,
    ) -> ArmResult:
        budget = Budget(budget_limit)

        def evaluate(candidate: AgentArtifact) -> float:
            # Optimizers cannot evaluate a candidate without paying budget.
            budget.consume()
            return rmse(candidate, train_tasks)

        result = optimizer.optimize(start, initial_dev_rmse, budget, seed, evaluate)
        # Recompute final metrics instead of trusting optimizer-reported scores.
        final_dev_rmse = rmse(result.successor, train_tasks)
        heldout_rmse = rmse(result.successor, heldout_tasks)
        return ArmResult(
            optimizer=optimizer.name,
            start_artifact_id=start.artifact_id,
            successor=result.successor,
            initial_dev_rmse=initial_dev_rmse,
            final_dev_rmse=final_dev_rmse,
            initial_heldout_rmse=initial_heldout_rmse,
            heldout_rmse=heldout_rmse,
            candidate_evaluations=budget.used,
            accepted_candidates=sum(entry.accepted for entry in result.history),
            history=result.history,
        )

    def run(self, manifest: ExperimentManifest) -> Experiment:
        if manifest.baseline != self.baseline.name or manifest.challenger != self.challenger.name:
            raise ValueError("Manifest optimizer names do not match the registered optimizers")
        if manifest.controls != tuple(control.name for control in self.controls):
            raise ValueError("Manifest controls do not match the registered controls")

        runs = []
        for seed in manifest.seeds:
            start = self.start_factory(seed)
            train_tasks, heldout_tasks = self.task_factory(
                seed, manifest.train_size, manifest.test_size, manifest.feature_dim
            )
            initial_dev_rmse = rmse(start, train_tasks)
            initial_heldout_rmse = rmse(start, heldout_tasks)
            baseline = self._run_arm(
                self.baseline,
                start,
                train_tasks,
                heldout_tasks,
                initial_dev_rmse,
                initial_heldout_rmse,
                manifest.candidate_budget,
                seed,
            )
            challenger = self._run_arm(
                self.challenger,
                start,
                train_tasks,
                heldout_tasks,
                initial_dev_rmse,
                initial_heldout_rmse,
                manifest.candidate_budget,
                seed,
            )
            controls = tuple(
                self._run_arm(
                    control,
                    start,
                    train_tasks,
                    heldout_tasks,
                    initial_dev_rmse,
                    initial_heldout_rmse,
                    manifest.candidate_budget,
                    seed,
                )
                for control in self.controls
            )
            improvement = baseline.heldout_rmse - challenger.heldout_rmse
            runs.append(
                PairedRun(
                    seed=seed,
                    start_artifact_id=start.artifact_id,
                    train_task_digest=tasks_digest(train_tasks),
                    heldout_task_digest=tasks_digest(heldout_tasks),
                    baseline=baseline,
                    challenger=challenger,
                    heldout_rmse_improvement=improvement,
                    controls=controls,
                )
            )

        differences = [run.heldout_rmse_improvement for run in runs]
        ci_lower, ci_upper = bootstrap_mean_ci(differences)
        control_summary = {}
        for index, control in enumerate(self.controls):
            arms = [run.controls[index] for run in runs]
            control_summary[control.name] = {
                "mean_heldout_rmse": statistics.fmean(arm.heldout_rmse for arm in arms),
                "mean_candidate_evaluations": statistics.fmean(arm.candidate_evaluations for arm in arms),
            }
        summary = {
            "run_count": len(runs),
            "metric": "heldout_rmse",
            "improvement_definition": "baseline_rmse - challenger_rmse; positive favors challenger",
            "mean_baseline_heldout_rmse": statistics.fmean(run.baseline.heldout_rmse for run in runs),
            "mean_challenger_heldout_rmse": statistics.fmean(run.challenger.heldout_rmse for run in runs),
            "mean_heldout_rmse_improvement": statistics.fmean(differences),
            "paired_bootstrap_95_ci": [ci_lower, ci_upper],
            "one_sided_sign_flip_p_value": sign_flip_pvalue(differences),
            "candidate_budget_per_optimization_arm": manifest.candidate_budget,
            "controls": control_summary,
            "scope": "Deterministic toy reference experiment; not an LLM or RSI benchmark result",
        }
        return Experiment(manifest=manifest, runs=tuple(runs), summary=summary)
