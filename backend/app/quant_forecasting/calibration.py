"""Calibration metrics for completed shadow outcomes."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from math import log
from uuid import NAMESPACE_URL, uuid5

from .models import CalibrationBucket, CalibrationObservation, CalibrationReport, CalibrationStatus, ForecastOutcome, HorizonPrediction, OutcomeStatus


class CalibrationReporter:
    def build_segmented(
        self,
        model_name: str,
        model_version: str,
        observations: list[CalibrationObservation],
        *,
        generated_at: datetime | None = None,
    ) -> tuple[CalibrationReport, ...]:
        """Produce aggregate plus horizon/session/regime/confidence/quality slices."""
        now = generated_at or datetime.now(UTC)
        reports = [
            self.build(
                model_name,
                model_version,
                [(item.prediction, item.outcome) for item in observations],
                generated_at=now,
            )
        ]
        dimensions: dict[str, Callable[[CalibrationObservation], str]] = {
            "horizon": lambda item: item.prediction.horizon.horizon_id,
            "session": lambda item: item.session,
            "regime": lambda item: item.regime,
            "confidence_band": lambda item: item.confidence_band,
            "data_quality_status": lambda item: item.data_quality_status,
        }
        for dimension, selector in dimensions.items():
            for value in sorted({selector(item) for item in observations}):
                scoped = [
                    (item.prediction, item.outcome)
                    for item in observations
                    if selector(item) == value
                ]
                reports.append(
                    self.build(
                        model_name,
                        model_version,
                        scoped,
                        generated_at=now,
                        dimension=dimension,
                        dimension_value=value,
                    )
                )
        return tuple(reports)

    def build(
        self,
        model_name: str,
        model_version: str,
        samples: list[tuple[HorizonPrediction, ForecastOutcome]],
        *,
        generated_at: datetime | None = None,
        dimension: str = "all",
        dimension_value: str = "all",
    ) -> CalibrationReport:
        now = generated_at or datetime.now(UTC)
        valid = [(prediction, outcome) for prediction, outcome in samples if outcome.status == OutcomeStatus.VALID]
        report_id = uuid5(NAMESPACE_URL, f"ten:calibration:{model_name}:{model_version}:{now.isoformat()}:{dimension}:{dimension_value}")
        if not valid:
            return CalibrationReport(
                report_id=report_id,
                model_name=model_name,
                model_version=model_version,
                generated_at=now,
                sample_count=0,
                status=CalibrationStatus.UNAVAILABLE,
                filters={dimension: dimension_value},
            )
        scored: list[tuple[float, int, str]] = []
        brier_terms: list[float] = []
        log_terms: list[float] = []
        for prediction, outcome in valid:
            realized = outcome.realized_direction or "neutral"
            probabilities = {
                "buy": prediction.buy_probability,
                "sell": prediction.sell_probability,
                "neutral": prediction.neutral_probability,
            }
            brier_terms.append(sum((probability - (1.0 if label == realized else 0.0)) ** 2 for label, probability in probabilities.items()) / 3)
            log_terms.append(-log(max(probabilities.get(realized, 0.0), 1e-15)))
            predicted_label, confidence = max(probabilities.items(), key=lambda item: item[1])
            scored.append((confidence, int(predicted_label == realized), prediction.horizon.horizon_id))
        buckets: list[CalibrationBucket] = []
        weighted_error = 0.0
        for horizon in sorted({item[2] for item in scored}):
            scoped = [item for item in scored if item[2] == horizon]
            for index in range(10):
                low, high = index / 10, (index + 1) / 10
                members = [item for item in scoped if low <= item[0] <= high if index == 9 or item[0] < high]
                if not members:
                    continue
                mean_confidence = sum(item[0] for item in members) / len(members)
                observed = sum(item[1] for item in members) / len(members)
                weighted_error += abs(mean_confidence - observed) * len(members) / len(scored)
                buckets.append(
                    CalibrationBucket(
                        horizon_id=horizon,
                        dimension=dimension,
                        dimension_value=dimension_value,
                        probability_low=low,
                        probability_high=high,
                        count=len(members),
                        mean_confidence=mean_confidence,
                        observed_frequency=observed,
                    )
                )
        return CalibrationReport(
            report_id=report_id,
            model_name=model_name,
            model_version=model_version,
            generated_at=now,
            sample_count=len(valid),
            brier_score=sum(brier_terms) / len(brier_terms),
            log_loss=sum(log_terms) / len(log_terms),
            expected_calibration_error=weighted_error,
            buckets=tuple(buckets),
            status=CalibrationStatus.UNCALIBRATED,
            filters={dimension: dimension_value},
        )
