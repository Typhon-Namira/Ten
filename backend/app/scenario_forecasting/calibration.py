"""Leakage-safe reliability summaries from completed scenario outcomes."""

from __future__ import annotations

from collections import defaultdict

from .models import ForwardMarketScenario, ScenarioOutcome


def calibration_reliability(
    completed: tuple[tuple[ForwardMarketScenario, ScenarioOutcome], ...],
    *,
    timeframe: str,
    minimum_sample: int = 20,
) -> tuple[str, float | None]:
    samples = [
        (scenario, outcome)
        for scenario, outcome in completed
        if scenario.timeframe == timeframe
        and outcome.completed_at >= scenario.expiry
    ]
    if len(samples) < minimum_sample:
        return "calibration_pending", None
    buckets: dict[str, list[float]] = defaultdict(list)
    for _scenario, outcome in samples:
        buckets[outcome.calibration_bucket].append(outcome.directional_accuracy)
    relevant = buckets.get(
        f"{int(samples[-1][0].raw_directional_confidence * 10) * 10:02d}",
        [],
    )
    values = relevant or [outcome.directional_accuracy for _, outcome in samples]
    return "calibrated", sum(values) / len(values)
