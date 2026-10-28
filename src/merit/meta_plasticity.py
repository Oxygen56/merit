"""Paired early/late improver evaluation with trajectory-level uncertainty."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .agent_contract import (
    AgentAcceptCallback,
    AgentContractExperiment,
    AgentContractManifest,
    AgentEvaluation,
    AgentExecutor,
    AgentOptimizerAdapter,
    AgentProposeCallback,
    AgentSpec,
    AgentTask,
    Usage,
    evaluate_agent,
    run_agent_contract,
    tasks_digest,
)
from .agent_pair import AgentSeedSpec, AgentTaskBundle
from .stats import bootstrap_mean_ci, sign_flip_pvalue
from .transfer import TaskFamily


PLASTICITY_PROTOCOL_VERSION = "merit-meta-plasticity-v0.1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _usage_sum(evaluations: Sequence[AgentEvaluation]) -> Usage:
    total = Usage()
    for evaluation in evaluations:
        total = total + evaluation.usage
    return total


@dataclass(frozen=True)
class ImproverSnapshot:
    """An improver artifact and its then-current agent; they are not interchangeable."""

    trajectory_id: str
    phase: str
    improver: AgentSpec
    current_agent: AgentSpec

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("Snapshot trajectory_id must not be empty")
        if self.phase not in {"early", "late"}:
            raise ValueError("Snapshot phase must be early or late")

    @property
    def snapshot_id(self) -> str:
        return _digest(
            {
                "trajectory_id": self.trajectory_id,
                "phase": self.phase,
                "improver_artifact_id": self.improver.artifact_id,
                "current_agent_artifact_id": self.current_agent.artifact_id,
            }
        )


@dataclass(frozen=True)
class ImproverTrajectory:
    trajectory_id: str
    early: ImproverSnapshot
    late: ImproverSnapshot

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("Trajectory ID must not be empty")
        if self.early.trajectory_id != self.trajectory_id or self.late.trajectory_id != self.trajectory_id:
            raise ValueError("Snapshots must belong to their registered trajectory")
        if self.early.phase != "early" or self.late.phase != "late":
            raise ValueError("A trajectory requires one early and one late snapshot")


class PlasticityProposer:
    """Register one proposer configuration; only the snapshot artifact varies by phase."""

    def __init__(
        self,
        name: str,
        source: str,
        factory: Callable[[AgentSpec], AgentProposeCallback],
        *,
        config: Mapping[str, Any] | None = None,
        accept: AgentAcceptCallback | None = None,
        accept_policy: str | None = None,
    ) -> None:
        if not name.strip() or not source.strip():
            raise ValueError("Proposer name and source must not be empty")
        if not callable(factory):
            raise TypeError("Proposer factory must be callable")
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("Proposer config must be a mapping")
        self.name = name
        self.source = source
        self.config_json = _canonical_json(dict(config or {}))
        self._factory = factory
        self._accept = accept
        self.accept_policy = accept_policy or ("greedy_accuracy" if accept is None else "custom")

    @property
    def config(self) -> dict[str, Any]:
        return json.loads(self.config_json)

    def make_optimizer(self, snapshot: ImproverSnapshot) -> AgentOptimizerAdapter:
        return AgentOptimizerAdapter(
            snapshot.phase,
            self._factory(snapshot.improver),
            accept=self._accept,
            proposer_source=self.source,
        )


@dataclass(frozen=True)
class PlasticityTrajectorySpec:
    trajectory_id: str
    early_snapshot_id: str
    late_snapshot_id: str
    early_improver_artifact_id: str
    late_improver_artifact_id: str
    early_current_agent_artifact_id: str
    late_current_agent_artifact_id: str


@dataclass(frozen=True)
class PlasticityFamilySpec:
    name: str
    role: str
    task_digest: str


@dataclass(frozen=True)
class MetaPlasticityManifest:
    experiment: str
    executor: str
    proposer_name: str
    proposer_source: str
    proposer_config_json: str
    accept_policy: str
    candidate_budget: int
    dev_threshold: float
    trajectories: tuple[PlasticityTrajectorySpec, ...]
    followup_runs: tuple[AgentSeedSpec, ...]
    families: tuple[PlasticityFamilySpec, ...] = ()
    protocol_version: str = PLASTICITY_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.candidate_budget < 1:
            raise ValueError("Follow-up candidate budget must be positive")
        if not math.isfinite(self.dev_threshold) or not 0 <= self.dev_threshold <= 1:
            raise ValueError("Development threshold must be between zero and one")
        if not self.trajectories or not self.followup_runs:
            raise ValueError("Meta-Plasticity requires trajectories and follow-up seeds")
        if len({item.trajectory_id for item in self.trajectories}) != len(self.trajectories):
            raise ValueError("Trajectory IDs must be unique")
        if len({item.seed for item in self.followup_runs}) != len(self.followup_runs):
            raise ValueError("Follow-up seeds must be unique")
        config = json.loads(self.proposer_config_json)
        if not isinstance(config, dict):
            raise ValueError("Proposer config must serialize to a JSON object")

    @property
    def digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class PlasticityFamilyComparison:
    family: str
    role: str
    tasks: tuple[AgentTask, ...]
    common_start: AgentEvaluation
    early_successor: AgentEvaluation
    late_successor: AgentEvaluation


@dataclass(frozen=True)
class MetaPlasticityRun:
    trajectory_id: str
    seed: int
    early_snapshot_id: str
    late_snapshot_id: str
    start_artifact_id: str
    train_task_digest: str
    heldout_task_digest: str
    early_current_agent: AgentEvaluation
    late_current_agent: AgentEvaluation
    early_followup: AgentContractExperiment
    late_followup: AgentContractExperiment
    family_comparisons: tuple[PlasticityFamilyComparison, ...] = ()

    @property
    def current_agent_delta(self) -> float:
        return self.late_current_agent.accuracy - self.early_current_agent.accuracy

    @property
    def successor_delta(self) -> float:
        return self.late_followup.successor_heldout.accuracy - self.early_followup.successor_heldout.accuracy


@dataclass(frozen=True)
class MetaPlasticityExperiment:
    manifest: MetaPlasticityManifest
    trajectories: tuple[ImproverTrajectory, ...]
    runs: tuple[MetaPlasticityRun, ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _trajectory_spec(trajectory: ImproverTrajectory) -> PlasticityTrajectorySpec:
    return PlasticityTrajectorySpec(
        trajectory_id=trajectory.trajectory_id,
        early_snapshot_id=trajectory.early.snapshot_id,
        late_snapshot_id=trajectory.late.snapshot_id,
        early_improver_artifact_id=trajectory.early.improver.artifact_id,
        late_improver_artifact_id=trajectory.late.improver.artifact_id,
        early_current_agent_artifact_id=trajectory.early.current_agent.artifact_id,
        late_current_agent_artifact_id=trajectory.late.current_agent.artifact_id,
    )


def _bundle_spec(bundle: AgentTaskBundle) -> AgentSeedSpec:
    return AgentSeedSpec(
        seed=bundle.seed,
        start_artifact_id=bundle.start.artifact_id,
        train_task_digest=tasks_digest(bundle.train_tasks),
        heldout_task_digest=tasks_digest(bundle.heldout_tasks),
    )


def _family_spec(family: TaskFamily) -> PlasticityFamilySpec:
    return PlasticityFamilySpec(family.name, family.role, family.digest)


def make_meta_plasticity_manifest(
    experiment: str,
    executor: str,
    proposer: PlasticityProposer,
    candidate_budget: int,
    dev_threshold: float,
    trajectories: Sequence[ImproverTrajectory],
    bundles: Sequence[AgentTaskBundle],
    families: Sequence[TaskFamily] = (),
) -> MetaPlasticityManifest:
    return MetaPlasticityManifest(
        experiment=experiment,
        executor=executor,
        proposer_name=proposer.name,
        proposer_source=proposer.source,
        proposer_config_json=proposer.config_json,
        accept_policy=proposer.accept_policy,
        candidate_budget=candidate_budget,
        dev_threshold=dev_threshold,
        trajectories=tuple(_trajectory_spec(item) for item in trajectories),
        followup_runs=tuple(_bundle_spec(item) for item in bundles),
        families=tuple(_family_spec(item) for item in families),
    )


def candidate_evaluations_to_threshold(
    followup: AgentContractExperiment,
    threshold: float,
) -> int | None:
    """Use only development feedback; None means right-censored at the budget."""

    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Development threshold must be between zero and one")
    if followup.initial_dev.accuracy >= threshold:
        return 0
    current = followup.initial_dev
    for entry in followup.history:
        if entry.accepted:
            current = entry.evaluation
        if current.accuracy >= threshold:
            return entry.iteration + 1
    return None


def _family_summary(
    runs: Sequence[MetaPlasticityRun],
) -> tuple[dict[str, Any], float | None, float | None, float | None]:
    names: list[str] = []
    for run in runs:
        for comparison in run.family_comparisons:
            if comparison.family not in names:
                names.append(comparison.family)
    result = {}
    transfer_deltas = []
    early_source_deltas = []
    late_source_deltas = []
    for name in names:
        comparisons = [item for run in runs for item in run.family_comparisons if item.family == name]
        early_delta = _mean([item.early_successor.accuracy - item.common_start.accuracy for item in comparisons])
        late_delta = _mean([item.late_successor.accuracy - item.common_start.accuracy for item in comparisons])
        successor_delta = _mean([item.late_successor.accuracy - item.early_successor.accuracy for item in comparisons])
        role = comparisons[0].role
        result[name] = {
            "role": role,
            "run_count": len(comparisons),
            "early_mean_accuracy": _mean([item.early_successor.accuracy for item in comparisons]),
            "late_mean_accuracy": _mean([item.late_successor.accuracy for item in comparisons]),
            "early_mean_improvement_from_common_start": early_delta,
            "late_mean_improvement_from_common_start": late_delta,
            "mean_successor_delta_late_minus_early": successor_delta,
        }
        if role == "transfer":
            transfer_deltas.extend(item.late_successor.accuracy - item.early_successor.accuracy for item in comparisons)
        if role == "source":
            early_source_deltas.extend(item.early_successor.accuracy - item.common_start.accuracy for item in comparisons)
            late_source_deltas.extend(item.late_successor.accuracy - item.common_start.accuracy for item in comparisons)
    return (
        result,
        _mean(transfer_deltas),
        min(early_source_deltas) if early_source_deltas else None,
        min(late_source_deltas) if late_source_deltas else None,
    )


def _run_metrics(run: MetaPlasticityRun, threshold: float) -> dict[str, Any]:
    early_threshold = candidate_evaluations_to_threshold(run.early_followup, threshold)
    late_threshold = candidate_evaluations_to_threshold(run.late_followup, threshold)
    early_family_cost = sum(item.early_successor.usage.cost_usd for item in run.family_comparisons)
    late_family_cost = sum(item.late_successor.usage.cost_usd for item in run.family_comparisons)
    return {
        "trajectory_id": run.trajectory_id,
        "seed": run.seed,
        "current_agent_delta_late_minus_early": run.current_agent_delta,
        "successor_delta_late_minus_early": run.successor_delta,
        "early_successor_accuracy": run.early_followup.successor_heldout.accuracy,
        "late_successor_accuracy": run.late_followup.successor_heldout.accuracy,
        "early_evaluations_to_dev_threshold": early_threshold,
        "late_evaluations_to_dev_threshold": late_threshold,
        "early_candidate_evaluations": run.early_followup.summary["candidate_evaluations"],
        "late_candidate_evaluations": run.late_followup.summary["candidate_evaluations"],
        "early_proposal_calls": run.early_followup.summary["proposal_calls"],
        "late_proposal_calls": run.late_followup.summary["proposal_calls"],
        "early_method_cost_usd": run.early_followup.summary["combined_total_cost_usd"] + early_family_cost,
        "late_method_cost_usd": run.late_followup.summary["combined_total_cost_usd"] + late_family_cost,
    }


def run_meta_plasticity(
    manifest: MetaPlasticityManifest,
    trajectories: Sequence[ImproverTrajectory],
    bundles: Sequence[AgentTaskBundle],
    proposer: PlasticityProposer,
    executor: AgentExecutor,
    families: Sequence[TaskFamily] = (),
) -> MetaPlasticityExperiment:
    """Compare early and late improvers from identical follow-up starting points."""

    trajectories = tuple(trajectories)
    bundles = tuple(bundles)
    families = tuple(families)
    if manifest.executor != executor.name:
        raise ValueError("Manifest executor does not match the registered executor")
    if manifest.trajectories != tuple(_trajectory_spec(item) for item in trajectories):
        raise ValueError("Manifest snapshots do not match the supplied trajectories")
    if manifest.followup_runs != tuple(_bundle_spec(item) for item in bundles):
        raise ValueError("Manifest follow-up specifications do not match the supplied bundles")
    if manifest.families != tuple(_family_spec(item) for item in families):
        raise ValueError("Manifest task families do not match the supplied families")
    if (
        manifest.proposer_name != proposer.name
        or manifest.proposer_source != proposer.source
        or manifest.proposer_config_json != proposer.config_json
        or manifest.accept_policy != proposer.accept_policy
    ):
        raise ValueError("Manifest proposer configuration does not match the registered proposer")
    if any(family.role not in {"source", "transfer"} for family in families):
        raise ValueError("Meta-Plasticity target tasks belong in follow-up held-out bundles")
    family_task_ids = [task.task_id for family in families for task in family.tasks]
    if len(set(family_task_ids)) != len(family_task_ids):
        raise ValueError("Task IDs must be unique across follow-up task families")
    for bundle in bundles:
        if set(family_task_ids).intersection(task.task_id for task in (*bundle.train_tasks, *bundle.heldout_tasks)):
            raise ValueError("Follow-up task families must be disjoint from train and held-out tasks")

    runs = []
    for trajectory in trajectories:
        for bundle in bundles:
            early_current = evaluate_agent(executor, trajectory.early.current_agent, bundle.heldout_tasks)
            late_current = evaluate_agent(executor, trajectory.late.current_agent, bundle.heldout_tasks)

            def run_followup(snapshot: ImproverSnapshot) -> AgentContractExperiment:
                optimizer = proposer.make_optimizer(snapshot)
                followup_manifest = AgentContractManifest(
                    experiment=f"{manifest.experiment}:{trajectory.trajectory_id}:{bundle.seed}:{snapshot.phase}",
                    executor=executor.name,
                    optimizer=optimizer.name,
                    seed=bundle.seed,
                    candidate_budget=manifest.candidate_budget,
                    start_artifact_id=bundle.start.artifact_id,
                    train_task_digest=tasks_digest(bundle.train_tasks),
                    heldout_task_digest=tasks_digest(bundle.heldout_tasks),
                    proposer_source=optimizer.proposer_source,
                )
                return run_agent_contract(
                    followup_manifest,
                    bundle.start,
                    bundle.train_tasks,
                    bundle.heldout_tasks,
                    optimizer,
                    executor,
                )

            early_followup = run_followup(trajectory.early)
            late_followup = run_followup(trajectory.late)
            if early_followup.baseline_heldout.accuracy != late_followup.baseline_heldout.accuracy:
                raise RuntimeError("Early and late follow-up arms produced different common-start scores")
            family_comparisons = []
            for family in families:
                common_start = evaluate_agent(executor, bundle.start, family.tasks)
                early_successor = evaluate_agent(executor, early_followup.successor, family.tasks)
                late_successor = evaluate_agent(executor, late_followup.successor, family.tasks)
                family_comparisons.append(
                    PlasticityFamilyComparison(
                        family.name,
                        family.role,
                        family.tasks,
                        common_start,
                        early_successor,
                        late_successor,
                    )
                )
            runs.append(
                MetaPlasticityRun(
                    trajectory_id=trajectory.trajectory_id,
                    seed=bundle.seed,
                    early_snapshot_id=trajectory.early.snapshot_id,
                    late_snapshot_id=trajectory.late.snapshot_id,
                    start_artifact_id=bundle.start.artifact_id,
                    train_task_digest=tasks_digest(bundle.train_tasks),
                    heldout_task_digest=tasks_digest(bundle.heldout_tasks),
                    early_current_agent=early_current,
                    late_current_agent=late_current,
                    early_followup=early_followup,
                    late_followup=late_followup,
                    family_comparisons=tuple(family_comparisons),
                )
            )

    run_metrics = [_run_metrics(run, manifest.dev_threshold) for run in runs]
    per_trajectory = []
    for trajectory in trajectories:
        metrics = [item for item in run_metrics if item["trajectory_id"] == trajectory.trajectory_id]
        early_reached = [item["early_evaluations_to_dev_threshold"] for item in metrics if item["early_evaluations_to_dev_threshold"] is not None]
        late_reached = [item["late_evaluations_to_dev_threshold"] for item in metrics if item["late_evaluations_to_dev_threshold"] is not None]
        per_trajectory.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "seed_count": len(metrics),
                "mean_current_agent_delta": _mean([item["current_agent_delta_late_minus_early"] for item in metrics]),
                "mean_successor_delta": _mean([item["successor_delta_late_minus_early"] for item in metrics]),
                "early_threshold_reached_rate": _mean([item["early_evaluations_to_dev_threshold"] is not None for item in metrics]),
                "late_threshold_reached_rate": _mean([item["late_evaluations_to_dev_threshold"] is not None for item in metrics]),
                "early_mean_evaluations_to_threshold_when_reached": _mean(early_reached),
                "late_mean_evaluations_to_threshold_when_reached": _mean(late_reached),
            }
        )
    trajectory_deltas = [item["mean_successor_delta"] for item in per_trajectory]
    ci_lower, ci_upper = bootstrap_mean_ci(trajectory_deltas)
    early_reached = [item["early_evaluations_to_dev_threshold"] for item in run_metrics if item["early_evaluations_to_dev_threshold"] is not None]
    late_reached = [item["late_evaluations_to_dev_threshold"] for item in run_metrics if item["late_evaluations_to_dev_threshold"] is not None]
    family_summary, transfer_delta, early_source_delta, late_source_delta = _family_summary(runs)

    early_family_usage = _usage_sum([item.early_successor for run in runs for item in run.family_comparisons])
    late_family_usage = _usage_sum([item.late_successor for run in runs for item in run.family_comparisons])
    shared_family_usage = _usage_sum([item.common_start for run in runs for item in run.family_comparisons])
    early_current_usage = _usage_sum([run.early_current_agent for run in runs])
    late_current_usage = _usage_sum([run.late_current_agent for run in runs])
    early_method_cost = sum(item["early_method_cost_usd"] for item in run_metrics)
    late_method_cost = sum(item["late_method_cost_usd"] for item in run_metrics)
    total_cost = (
        early_method_cost
        + late_method_cost
        + shared_family_usage.cost_usd
        + early_current_usage.cost_usd
        + late_current_usage.cost_usd
    )
    summary = {
        "metric": "heldout_exact_match_accuracy",
        "trajectory_count": len(trajectories),
        "seed_count_per_trajectory": len(bundles),
        "paired_run_count": len(runs),
        "candidate_budget_per_phase": manifest.candidate_budget,
        "dev_threshold": manifest.dev_threshold,
        "mean_early_current_agent_accuracy": _mean([run.early_current_agent.accuracy for run in runs]),
        "mean_late_current_agent_accuracy": _mean([run.late_current_agent.accuracy for run in runs]),
        "mean_current_agent_delta_late_minus_early": _mean([run.current_agent_delta for run in runs]),
        "mean_early_successor_accuracy": _mean([run.early_followup.successor_heldout.accuracy for run in runs]),
        "mean_late_successor_accuracy": _mean([run.late_followup.successor_heldout.accuracy for run in runs]),
        "mean_successor_delta_late_minus_early": _mean([run.successor_delta for run in runs]),
        "trajectory_bootstrap_95_ci": [ci_lower, ci_upper],
        "one_sided_trajectory_sign_flip_p_value": sign_flip_pvalue(trajectory_deltas),
        "uncertainty_unit": "trajectory; seeds are paired within a trajectory and are not independent replicates",
        "early_threshold_reached_rate": _mean([item["early_evaluations_to_dev_threshold"] is not None for item in run_metrics]),
        "late_threshold_reached_rate": _mean([item["late_evaluations_to_dev_threshold"] is not None for item in run_metrics]),
        "early_mean_evaluations_to_threshold_when_reached": _mean(early_reached),
        "late_mean_evaluations_to_threshold_when_reached": _mean(late_reached),
        "threshold_definition": "First accepted development evaluation reaching the threshold; unreached runs are right-censored",
        "early_mean_candidate_evaluations": _mean([item["early_candidate_evaluations"] for item in run_metrics]),
        "late_mean_candidate_evaluations": _mean([item["late_candidate_evaluations"] for item in run_metrics]),
        "early_mean_proposal_calls": _mean([item["early_proposal_calls"] for item in run_metrics]),
        "late_mean_proposal_calls": _mean([item["late_proposal_calls"] for item in run_metrics]),
        "early_method_cost_usd": early_method_cost,
        "late_method_cost_usd": late_method_cost,
        "shared_family_start_cost_usd": shared_family_usage.cost_usd,
        "early_current_agent_evaluation_cost_usd": early_current_usage.cost_usd,
        "late_current_agent_evaluation_cost_usd": late_current_usage.cost_usd,
        "total_observed_cost_usd": total_cost,
        "cost_accounting": "Method cost includes follow-up evaluated-agent and proposer costs plus family successor evaluation; shared-start and snapshot costs are separate",
        "early_family_successor_cost_usd": early_family_usage.cost_usd,
        "late_family_successor_cost_usd": late_family_usage.cost_usd,
        "mean_transfer_successor_delta_late_minus_early": transfer_delta,
        "early_worst_source_improvement": early_source_delta,
        "late_worst_source_improvement": late_source_delta,
        "family_summary": family_summary,
        "per_trajectory": per_trajectory,
        "per_run": run_metrics,
        "scope": "Paired follow-up plasticity is specific to the supplied snapshots, executor, tasks and proposer configuration",
    }
    return MetaPlasticityExperiment(manifest, trajectories, tuple(runs), summary)
