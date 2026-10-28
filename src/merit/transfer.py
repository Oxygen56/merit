"""Cross-family transfer and old-capability regression evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .agent_contract import (
    AgentContractExperiment,
    AgentEvaluation,
    AgentExecutor,
    AgentTask,
    Usage,
    evaluate_agent,
    tasks_digest,
)


TRANSFER_PROTOCOL_VERSION = "merit-transfer-v0.1"


@dataclass(frozen=True)
class TaskFamily:
    name: str
    role: str
    tasks: tuple[AgentTask, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Task family name must not be empty")
        if self.role not in {"source", "target", "transfer"}:
            raise ValueError("Task family role must be source, target or transfer")
        if not self.tasks:
            raise ValueError("Task family must contain at least one task")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Task IDs must be unique within a family")

    @property
    def digest(self) -> str:
        return tasks_digest(self.tasks)


@dataclass(frozen=True)
class TransferFamilySpec:
    name: str
    role: str
    task_digest: str


@dataclass(frozen=True)
class TransferManifest:
    experiment: str
    base_manifest_digest: str
    families: tuple[TransferFamilySpec, ...]
    protocol_version: str = TRANSFER_PROTOCOL_VERSION

    @property
    def digest(self) -> str:
        import hashlib
        import json

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FamilyComparison:
    family: str
    role: str
    tasks: tuple[AgentTask, ...]
    baseline: AgentEvaluation
    successor: AgentEvaluation
    reused_base_evaluation: bool

    @property
    def improvement(self) -> float:
        return self.successor.accuracy - self.baseline.accuracy


@dataclass(frozen=True)
class TransferExperiment:
    manifest: TransferManifest
    base: AgentContractExperiment
    comparisons: tuple[FamilyComparison, ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_transfer_manifest(
    experiment: str,
    base: AgentContractExperiment,
    families: Sequence[TaskFamily],
) -> TransferManifest:
    return TransferManifest(
        experiment=experiment,
        base_manifest_digest=base.manifest.digest,
        families=tuple(
            TransferFamilySpec(name=family.name, role=family.role, task_digest=family.digest)
            for family in families
        ),
    )


def _usage_sum(comparisons: Sequence[FamilyComparison], include_reused: bool) -> Usage:
    total = Usage()
    for comparison in comparisons:
        if include_reused or not comparison.reused_base_evaluation:
            total = total + comparison.baseline.usage + comparison.successor.usage
    return total


def run_transfer_evaluation(
    manifest: TransferManifest,
    base: AgentContractExperiment,
    executor: AgentExecutor,
    families: Sequence[TaskFamily],
) -> TransferExperiment:
    """Evaluate both artifacts on each family, reusing the base target evaluation."""

    families = tuple(families)
    if not families:
        raise ValueError("Transfer evaluation requires at least one task family")
    if manifest.base_manifest_digest != base.manifest.digest:
        raise ValueError("Transfer manifest does not match the base experiment")
    expected_specs = tuple(
        TransferFamilySpec(name=family.name, role=family.role, task_digest=family.digest)
        for family in families
    )
    if manifest.families != expected_specs:
        raise ValueError("Transfer manifest families do not match the supplied families")
    if len({family.name for family in families}) != len(families):
        raise ValueError("Task family names must be unique")
    all_task_ids = [task.task_id for family in families for task in family.tasks]
    if len(set(all_task_ids)) != len(all_task_ids):
        raise ValueError("Task IDs must be unique across task families")
    if sum(family.role == "target" for family in families) != 1:
        raise ValueError("Transfer evaluation requires exactly one target family")
    if not any(family.role == "source" for family in families):
        raise ValueError("Transfer evaluation requires a source regression family")
    if not any(family.role == "transfer" for family in families):
        raise ValueError("Transfer evaluation requires a transfer family")

    comparisons = []
    for family in families:
        reused = family.role == "target" and family.digest == base.manifest.heldout_task_digest
        if family.role == "target" and not reused:
            raise ValueError("Target family must match the base held-out task digest")
        if reused:
            baseline = base.baseline_heldout
            successor = base.successor_heldout
        else:
            baseline = evaluate_agent(executor, base.start, family.tasks)
            successor = evaluate_agent(executor, base.successor, family.tasks)
        comparisons.append(
            FamilyComparison(
                family=family.name,
                role=family.role,
                tasks=family.tasks,
                baseline=baseline,
                successor=successor,
                reused_base_evaluation=reused,
            )
        )

    target = next(comparison for comparison in comparisons if comparison.role == "target")
    sources = [comparison for comparison in comparisons if comparison.role == "source"]
    transfers = [comparison for comparison in comparisons if comparison.role == "transfer"]
    additional_usage = _usage_sum(comparisons, include_reused=False)
    total_usage = _usage_sum(comparisons, include_reused=True)
    summary = {
        "metric": "exact_match_accuracy",
        "target_baseline_accuracy": target.baseline.accuracy,
        "target_successor_accuracy": target.successor.accuracy,
        "target_improvement": target.improvement,
        "min_source_baseline_accuracy": min(comparison.baseline.accuracy for comparison in sources),
        "min_source_successor_accuracy": min(comparison.successor.accuracy for comparison in sources),
        "worst_source_accuracy_delta": min(comparison.improvement for comparison in sources),
        "mean_transfer_baseline_accuracy": sum(comparison.baseline.accuracy for comparison in transfers) / len(transfers),
        "mean_transfer_successor_accuracy": sum(comparison.successor.accuracy for comparison in transfers) / len(transfers),
        "mean_transfer_improvement": sum(comparison.improvement for comparison in transfers) / len(transfers),
        "additional_agent_invocations": sum(
            2 * len(comparison.tasks) for comparison in comparisons if not comparison.reused_base_evaluation
        ),
        "additional_input_tokens": additional_usage.input_tokens,
        "additional_output_tokens": additional_usage.output_tokens,
        "additional_cost_usd": additional_usage.cost_usd,
        "total_matrix_input_tokens": total_usage.input_tokens,
        "total_matrix_output_tokens": total_usage.output_tokens,
        "total_matrix_cost_usd": total_usage.cost_usd,
        "interpretation": "Accuracy delta is successor minus baseline; negative source delta indicates regression",
        "scope": "Task-family transfer and regression results are specific to the registered executor and tasks",
    }
    return TransferExperiment(manifest, base, tuple(comparisons), summary)
