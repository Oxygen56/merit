"""Core data contracts for Merit paired improvement experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


PROTOCOL_VERSION = "merit-paired-v0.1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class AgentArtifact:
    """An immutable agent snapshot with content-addressed lineage."""

    values: tuple[float, ...]
    parent: str | None = None
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(float(v) for v in self.values))
        if not self.values:
            raise ValueError("AgentArtifact.values must not be empty")
        payload = {"parent": self.parent, "values": self.values, "version": 1}
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        object.__setattr__(self, "artifact_id", digest)

    def with_values(self, values: tuple[float, ...]) -> "AgentArtifact":
        return AgentArtifact(values=values, parent=self.artifact_id)


@dataclass(frozen=True)
class Task:
    features: tuple[float, ...]
    target: float


@dataclass(frozen=True)
class ExperimentManifest:
    experiment: str
    baseline: str
    challenger: str
    seeds: tuple[int, ...]
    candidate_budget: int
    train_size: int
    test_size: int
    feature_dim: int
    controls: tuple[str, ...] = ()
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("At least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique")
        if self.candidate_budget < 1:
            raise ValueError("candidate_budget must be positive")
        if min(self.train_size, self.test_size, self.feature_dim) < 1:
            raise ValueError("Task sizes and feature_dim must be positive")
        if len(set(self.controls)) != len(self.controls):
            raise ValueError("Control names must be unique")
        if {self.baseline, self.challenger}.intersection(self.controls):
            raise ValueError("Controls must be distinct from baseline and challenger")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HistoryEntry:
    iteration: int
    candidate_artifact_id: str
    candidate: AgentArtifact
    dev_rmse: float
    accepted: bool
    remaining_budget: int


@dataclass(frozen=True)
class OptimizeResult:
    successor: AgentArtifact
    final_dev_rmse: float
    history: tuple[HistoryEntry, ...]


@dataclass(frozen=True)
class ArmResult:
    optimizer: str
    start_artifact_id: str
    successor: AgentArtifact
    initial_dev_rmse: float
    final_dev_rmse: float
    initial_heldout_rmse: float
    heldout_rmse: float
    candidate_evaluations: int
    accepted_candidates: int
    history: tuple[HistoryEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedRun:
    seed: int
    start_artifact_id: str
    train_task_digest: str
    heldout_task_digest: str
    baseline: ArmResult
    challenger: ArmResult
    heldout_rmse_improvement: float
    controls: tuple[ArmResult, ...] = ()


@dataclass(frozen=True)
class Experiment:
    manifest: ExperimentManifest
    runs: tuple[PairedRun, ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
