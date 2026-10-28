"""A deterministic, dependency-free reference experiment."""

from __future__ import annotations

import random

from .model import AgentArtifact, Experiment, ExperimentManifest, Task
from .optimizers import (
    CoordinateSearchOptimizer,
    MemoryResetOptimizer,
    NoChangeOptimizer,
    RandomEditOptimizer,
    RandomSearchOptimizer,
)
from .runner import PairRunner


BASE_WEIGHTS = (1.4, -2.1, 0.8, 2.6)


def default_manifest(seed_count: int = 12, candidate_budget: int = 64) -> ExperimentManifest:
    return ExperimentManifest(
        experiment="toy-random-vs-coordinate",
        baseline=RandomSearchOptimizer.name,
        challenger=CoordinateSearchOptimizer.name,
        seeds=tuple(range(1001, 1001 + seed_count)),
        candidate_budget=candidate_budget,
        train_size=40,
        test_size=80,
        feature_dim=len(BASE_WEIGHTS),
        controls=(NoChangeOptimizer.name, RandomEditOptimizer.name, MemoryResetOptimizer.name),
    )


def make_start(seed: int) -> AgentArtifact:
    del seed
    return AgentArtifact(values=(0.0,) * len(BASE_WEIGHTS))


def make_tasks(seed: int, train_size: int, test_size: int, feature_dim: int) -> tuple[list[Task], list[Task]]:
    if feature_dim != len(BASE_WEIGHTS):
        raise ValueError("The toy task generator expects four features")
    rng = random.Random(seed)
    target_weights = tuple(weight + rng.uniform(-0.15, 0.15) for weight in BASE_WEIGHTS)

    def generate(count: int) -> list[Task]:
        tasks = []
        for _ in range(count):
            features = tuple(rng.uniform(-1.0, 1.0) for _ in range(feature_dim))
            target = sum(weight * feature for weight, feature in zip(target_weights, features))
            target += rng.gauss(0.0, 0.03)
            tasks.append(Task(features=features, target=target))
        return tasks

    return generate(train_size), generate(test_size)


def run_toy_experiment(manifest: ExperimentManifest) -> Experiment:
    runner = PairRunner(
        baseline=RandomSearchOptimizer(),
        challenger=CoordinateSearchOptimizer(),
        start_factory=make_start,
        task_factory=make_tasks,
        controls=(NoChangeOptimizer(), RandomEditOptimizer(), MemoryResetOptimizer()),
    )
    return runner.run(manifest)
