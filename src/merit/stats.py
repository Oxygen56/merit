"""Paired descriptive statistics with no third-party dependencies."""

from __future__ import annotations

import itertools
import math
import random
import statistics
from typing import Sequence


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(
    values: Sequence[float], *, confidence: float = 0.95, resamples: int = 5000, seed: int = 20261027
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired per-seed differences."""
    if not values:
        raise ValueError("At least one paired difference is required")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    return _percentile(means, alpha), _percentile(means, 1.0 - alpha)


def sign_flip_pvalue(values: Sequence[float], *, resamples: int = 5000, seed: int = 20261027) -> float:
    """One-sided paired sign-flip p-value for the alternative mean(delta) > 0."""
    if not values:
        raise ValueError("At least one paired difference is required")
    observed = statistics.fmean(values)
    n = len(values)
    if n <= 16:
        extreme = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            total += 1
            statistic = statistics.fmean(sign * value for sign, value in zip(signs, values))
            if statistic >= observed - 1e-15:
                extreme += 1
        return extreme / total
    rng = random.Random(seed)
    extreme = 0
    for _ in range(resamples):
        statistic = statistics.fmean((1.0 if rng.random() < 0.5 else -1.0) * value for value in values)
        if statistic >= observed:
            extreme += 1
    return (extreme + 1) / (resamples + 1)
