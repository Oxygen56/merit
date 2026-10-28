"""Multi-seed paired comparisons and ablation controls for agent improvers."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .agent_contract import (
    AgentContractExperiment,
    AgentContractManifest,
    AgentExecutor,
    AgentOptimizerAdapter,
    AgentSpec,
    AgentTask,
    run_agent_contract,
    tasks_digest,
)
from .stats import bootstrap_mean_ci, sign_flip_pvalue


AGENT_PAIR_PROTOCOL_VERSION = "merit-agent-paired-v0.1"


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AgentTaskBundle:
    seed: int
    start: AgentSpec
    train_tasks: tuple[AgentTask, ...]
    heldout_tasks: tuple[AgentTask, ...]
    validation_tasks: tuple[AgentTask, ...] = ()


@dataclass(frozen=True)
class AgentSeedSpec:
    seed: int
    start_artifact_id: str
    train_task_digest: str
    heldout_task_digest: str
    validation_task_digest: str | None = None


@dataclass(frozen=True)
class AgentPairedManifest:
    experiment: str
    executor: str
    baseline: str
    challenger: str
    candidate_budget: int
    runs: tuple[AgentSeedSpec, ...]
    protocol_version: str = AGENT_PAIR_PROTOCOL_VERSION
    controls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.baseline == self.challenger:
            raise ValueError("Baseline and challenger names must be distinct")
        if self.candidate_budget < 1:
            raise ValueError("candidate_budget must be positive")
        if not self.runs:
            raise ValueError("At least one paired seed is required")
        if len({run.seed for run in self.runs}) != len(self.runs):
            raise ValueError("Paired seeds must be unique")
        if len(set(self.controls)) != len(self.controls):
            raise ValueError("Agent control names must be unique")
        if {self.baseline, self.challenger}.intersection(self.controls):
            raise ValueError("Agent controls must be distinct from baseline and challenger")

    @property
    def digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class AgentPairedRun:
    seed: int
    start_artifact_id: str
    train_task_digest: str
    heldout_task_digest: str
    baseline: AgentContractExperiment
    challenger: AgentContractExperiment
    heldout_accuracy_improvement: float
    controls: tuple[AgentContractExperiment, ...] = ()
    validation_task_digest: str | None = None


@dataclass(frozen=True)
class AgentPairedExperiment:
    manifest: AgentPairedManifest
    runs: tuple[AgentPairedRun, ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_agent_paired_manifest(
    experiment: str,
    executor: str,
    baseline: str,
    challenger: str,
    candidate_budget: int,
    bundles: Sequence[AgentTaskBundle],
    controls: Sequence[str] = (),
) -> AgentPairedManifest:
    return AgentPairedManifest(
        experiment=experiment,
        executor=executor,
        baseline=baseline,
        challenger=challenger,
        candidate_budget=candidate_budget,
        runs=tuple(
            AgentSeedSpec(
                seed=bundle.seed,
                start_artifact_id=bundle.start.artifact_id,
                train_task_digest=tasks_digest(bundle.train_tasks),
                heldout_task_digest=tasks_digest(bundle.heldout_tasks),
                validation_task_digest=(
                    tasks_digest(bundle.validation_tasks) if bundle.validation_tasks else None
                ),
            )
            for bundle in bundles
        ),
        controls=tuple(controls),
    )


def run_agent_paired(
    manifest: AgentPairedManifest,
    bundles: Sequence[AgentTaskBundle],
    baseline: AgentOptimizerAdapter,
    challenger: AgentOptimizerAdapter,
    executor: AgentExecutor,
    controls: Sequence[AgentOptimizerAdapter] = (),
) -> AgentPairedExperiment:
    if manifest.executor != executor.name:
        raise ValueError("Manifest executor does not match the registered executor")
    if manifest.baseline != baseline.name or manifest.challenger != challenger.name:
        raise ValueError("Manifest optimizer names do not match the registered optimizers")
    controls = tuple(controls)
    if manifest.controls != tuple(control.name for control in controls):
        raise ValueError("Manifest controls do not match the registered controls")
    bundles = tuple(bundles)
    expected_specs = tuple(
        AgentSeedSpec(
            seed=bundle.seed,
            start_artifact_id=bundle.start.artifact_id,
            train_task_digest=tasks_digest(bundle.train_tasks),
            heldout_task_digest=tasks_digest(bundle.heldout_tasks),
            validation_task_digest=tasks_digest(bundle.validation_tasks) if bundle.validation_tasks else None,
        )
        for bundle in bundles
    )
    if manifest.runs != expected_specs:
        raise ValueError("Manifest seed specifications do not match the supplied task bundles")

    runs = []
    for bundle in bundles:

        def run_arm(optimizer: AgentOptimizerAdapter) -> AgentContractExperiment:
            arm_manifest = AgentContractManifest(
                experiment=f"{manifest.experiment}:{bundle.seed}:{optimizer.name}",
                executor=executor.name,
                optimizer=optimizer.name,
                seed=bundle.seed,
                candidate_budget=manifest.candidate_budget,
                start_artifact_id=bundle.start.artifact_id,
                train_task_digest=tasks_digest(bundle.train_tasks),
                heldout_task_digest=tasks_digest(bundle.heldout_tasks),
                proposer_source=optimizer.proposer_source,
                validation_task_digest=(
                    tasks_digest(bundle.validation_tasks) if bundle.validation_tasks else None
                ),
            )
            return run_agent_contract(
                arm_manifest,
                bundle.start,
                bundle.train_tasks,
                bundle.heldout_tasks,
                optimizer,
                executor,
                bundle.validation_tasks,
            )

        baseline_result = run_arm(baseline)
        challenger_result = run_arm(challenger)
        control_results = tuple(run_arm(control) for control in controls)
        if baseline_result.baseline_heldout.accuracy != challenger_result.baseline_heldout.accuracy:
            raise RuntimeError("Paired arms produced different no-change baseline scores")
        if any(
            control.baseline_heldout.accuracy != baseline_result.baseline_heldout.accuracy
            for control in control_results
        ):
            raise RuntimeError("Control arms produced different no-change baseline scores")
        improvement = (
            challenger_result.successor_heldout.accuracy - baseline_result.successor_heldout.accuracy
        )
        runs.append(
            AgentPairedRun(
                seed=bundle.seed,
                start_artifact_id=bundle.start.artifact_id,
                train_task_digest=tasks_digest(bundle.train_tasks),
                heldout_task_digest=tasks_digest(bundle.heldout_tasks),
                baseline=baseline_result,
                challenger=challenger_result,
                heldout_accuracy_improvement=improvement,
                controls=control_results,
                validation_task_digest=(
                    tasks_digest(bundle.validation_tasks) if bundle.validation_tasks else None
                ),
            )
        )

    differences = [run.heldout_accuracy_improvement for run in runs]
    ci_lower, ci_upper = bootstrap_mean_ci(differences)
    control_summary = {}
    for index, control in enumerate(controls):
        arms = [run.controls[index] for run in runs]
        control_summary[control.name] = {
            "mean_heldout_accuracy": statistics.fmean(arm.successor_heldout.accuracy for arm in arms),
            "mean_candidate_evaluations": statistics.fmean(arm.summary["candidate_evaluations"] for arm in arms),
            "total_proposer_cost_usd": sum(arm.summary["proposer_cost_usd"] for arm in arms),
            "total_cost_usd": sum(arm.summary["total_cost_usd"] for arm in arms),
        }
    summary = {
        "run_count": len(runs),
        "metric": "heldout_exact_match_accuracy",
        "improvement_definition": "challenger_accuracy - baseline_accuracy; positive favors challenger",
        "mean_baseline_heldout_accuracy": statistics.fmean(
            run.baseline.successor_heldout.accuracy for run in runs
        ),
        "mean_challenger_heldout_accuracy": statistics.fmean(
            run.challenger.successor_heldout.accuracy for run in runs
        ),
        "mean_heldout_accuracy_improvement": statistics.fmean(differences),
        "paired_bootstrap_95_ci": [ci_lower, ci_upper],
        "one_sided_sign_flip_p_value": sign_flip_pvalue(differences),
        "candidate_budget_per_arm": manifest.candidate_budget,
        "baseline_candidate_evaluations": sum(run.baseline.summary["candidate_evaluations"] for run in runs),
        "challenger_candidate_evaluations": sum(run.challenger.summary["candidate_evaluations"] for run in runs),
        "baseline_total_cost_usd": sum(run.baseline.summary["total_cost_usd"] for run in runs),
        "challenger_total_cost_usd": sum(run.challenger.summary["total_cost_usd"] for run in runs),
        "baseline_proposer_cost_usd": sum(run.baseline.summary["proposer_cost_usd"] for run in runs),
        "challenger_proposer_cost_usd": sum(run.challenger.summary["proposer_cost_usd"] for run in runs),
        "baseline_combined_cost_usd": sum(run.baseline.summary["combined_total_cost_usd"] for run in runs),
        "challenger_combined_cost_usd": sum(run.challenger.summary["combined_total_cost_usd"] for run in runs),
        "baseline_total_input_tokens": sum(run.baseline.summary["total_input_tokens"] for run in runs),
        "baseline_total_output_tokens": sum(run.baseline.summary["total_output_tokens"] for run in runs),
        "challenger_total_input_tokens": sum(run.challenger.summary["total_input_tokens"] for run in runs),
        "challenger_total_output_tokens": sum(run.challenger.summary["total_output_tokens"] for run in runs),
        "baseline_total_cost_reported": all(run.baseline.summary["total_cost_reported"] for run in runs),
        "challenger_total_cost_reported": all(run.challenger.summary["total_cost_reported"] for run in runs),
        "baseline_total_tokens_reported": all(run.baseline.summary["total_tokens_reported"] for run in runs),
        "challenger_total_tokens_reported": all(run.challenger.summary["total_tokens_reported"] for run in runs),
        "controls": control_summary,
        "scope": "Paired agent-improvement results are specific to the registered executor and task bundles",
    }
    return AgentPairedExperiment(manifest, tuple(runs), summary)
