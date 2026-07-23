"""Measured comparison, probability calibration, and production-readiness reports."""

from __future__ import annotations

from datetime import UTC, datetime
from math import log
from statistics import mean
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from .config import GuardrailPolicyConfig
from .models import (
    DetailedSignalOutcome,
    PerformanceReport,
    ProbabilityCalibrationReport,
    ProductionReadinessReport,
)


class PerformanceReporter:
    def build(
        self,
        *,
        period_start: datetime,
        period_end: datetime,
        system_outcomes: dict[str, tuple[DetailedSignalOutcome, ...]],
        dimensions: dict[str, dict[str, tuple[DetailedSignalOutcome, ...]]] | None = None,
        generated_at: datetime | None = None,
    ) -> PerformanceReport:
        now = generated_at or datetime.now(UTC)
        comparison = {name: self._metrics(values) for name, values in system_outcomes.items()}
        dimension_metrics = {
            dimension: {key: self._metrics(values) for key, values in buckets.items()}
            for dimension, buckets in (dimensions or {}).items()
        }
        sample_count = max((len(values) for values in system_outcomes.values()), default=0)
        return PerformanceReport(
            report_id=uuid5(NAMESPACE_URL, f"ten:performance:{period_start.isoformat()}:{period_end.isoformat()}"),
            period_start=period_start,
            period_end=period_end,
            comparison=comparison,
            dimensions=dimension_metrics,
            sample_count=sample_count,
            generated_at=now,
        )

    @staticmethod
    def _metrics(outcomes: tuple[DetailedSignalOutcome, ...]) -> dict[str, float | int | None]:
        complete = [item for item in outcomes if item.evaluation_horizon_complete]
        returns = [item.slippage_adjusted_result for item in complete if item.slippage_adjusted_result is not None]
        wins = [value for value in returns if value > 0]
        losses = [value for value in returns if value < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        cumulative = 0.0
        peak = 0.0
        drawdown = 0.0
        for value in returns:
            cumulative += value
            peak = max(peak, cumulative)
            drawdown = max(drawdown, peak - cumulative)
        return {
            "analysis_cycles": len(outcomes),
            "completed_sample_size": len(complete),
            "win_rate": len(wins) / len(returns) if returns else None,
            "loss_rate": len(losses) / len(returns) if returns else None,
            "expected_value": mean(returns) if returns else None,
            "average_risk_to_reward": mean([item.realized_risk_to_reward for item in complete if item.realized_risk_to_reward is not None]) if any(item.realized_risk_to_reward is not None for item in complete) else None,
            "maximum_drawdown": drawdown if returns else None,
            "profit_factor": gross_profit / gross_loss if gross_loss else None,
            "average_mfe": mean([item.maximum_favorable_excursion for item in complete if item.maximum_favorable_excursion is not None]) if complete else None,
            "average_mae": mean([item.maximum_adverse_excursion for item in complete if item.maximum_adverse_excursion is not None]) if complete else None,
            "tp1_rate": sum(item.tp1_result == "hit" for item in complete) / len(complete) if complete else None,
            "stop_rate": sum(item.stop_loss_result == "hit" for item in complete) / len(complete) if complete else None,
            "performance_after_spread_and_slippage": sum(returns) if returns else None,
        }


class ProbabilityCalibration:
    def calculate(
        self,
        observations: tuple[dict[str, Any], ...],
        *,
        generated_at: datetime | None = None,
    ) -> ProbabilityCalibrationReport:
        now = generated_at or datetime.now(UTC)
        if not observations:
            return ProbabilityCalibrationReport(
                report_id=uuid5(NAMESPACE_URL, "ten:ai-calibration:empty"),
                status="uncalibrated",
                sample_count=0,
                brier_score=None,
                log_loss=None,
                expected_calibration_error=None,
                reliability_buckets=(),
                dimensions={},
                generated_at=now,
            )
        labels = ("BUY", "SELL", "NEUTRAL")
        brier_values: list[float] = []
        log_losses: list[float] = []
        bucket_values: list[dict[str, Any]] = []
        for item in observations:
            actual = item["actual"]
            probabilities = item["probabilities"]
            brier_values.append(sum((float(probabilities[label]) - float(actual == label)) ** 2 for label in labels) / 3)
            log_losses.append(-log(max(1e-12, float(probabilities[actual]))))
        for label in labels:
            for low in (0.0, 0.2, 0.4, 0.6, 0.8):
                high = low + 0.2
                members = [item for item in observations if low <= float(item["probabilities"][label]) <= high]
                bucket_values.append(
                    {
                        "label": label,
                        "probability_low": low,
                        "probability_high": high,
                        "count": len(members),
                        "mean_probability": mean([float(item["probabilities"][label]) for item in members]) if members else None,
                        "observed_frequency": mean([float(item["actual"] == label) for item in members]) if members else None,
                    }
                )
        nonempty = [item for item in bucket_values if item["count"]]
        ece = sum(
            item["count"] / len(observations) * abs(float(item["mean_probability"]) - float(item["observed_frequency"]))
            for item in nonempty
        ) / 3
        dimensions: dict[str, dict[str, float | int | None]] = {}
        for dimension in ("horizon", "setup_family", "regime", "session", "confidence_band"):
            values = {str(item.get(dimension, "unknown")) for item in observations}
            dimensions[dimension] = {value: sum(str(item.get(dimension, "unknown")) == value for item in observations) for value in values}
        return ProbabilityCalibrationReport(
            report_id=uuid5(NAMESPACE_URL, f"ten:ai-calibration:{len(observations)}:{now.date().isoformat()}"),
            status="measured_uncalibrated" if len(observations) < 100 else "measured",
            sample_count=len(observations),
            brier_score=mean(brier_values),
            log_loss=mean(log_losses),
            expected_calibration_error=ece,
            reliability_buckets=tuple(bucket_values),
            dimensions=dimensions,
            generated_at=now,
        )


class ProductionReadinessEvaluator:
    def __init__(self, config: GuardrailPolicyConfig) -> None:
        self.config = config

    def evaluate(self, measurements: dict[str, Any], *, generated_at: datetime | None = None) -> ProductionReadinessReport:
        now = generated_at or datetime.now(UTC)
        sample_count = int(measurements.get("sample_count", 0))
        checks = {
            "data_integrity": self._check(bool(measurements.get("data_integrity")), True),
            "future_data_safety": self._check(bool(measurements.get("future_data_safety")), True),
            "model_availability": self._check(bool(measurements.get("model_availability")), True),
            "structured_output_reliability": self._check(float(measurements.get("llm_failure_rate", 1.0)) <= self.config.maximum_llm_failure_rate, self.config.maximum_llm_failure_rate),
            "proposal_validation": self._check(bool(measurements.get("proposal_validation")), True),
            "guardrail_correctness": self._check(bool(measurements.get("guardrail_correctness")), True),
            "lifecycle_consistency": self._check(bool(measurements.get("lifecycle_consistency")), True),
            "duplicate_prevention": self._check(bool(measurements.get("duplicate_prevention")), True),
            "calibration_quality": self._check(measurements.get("calibration_quality") is not None, "measured value required"),
            "expected_value": self._check(measurements.get("expected_value") is not None, "measured value required"),
            "drawdown": self._check(measurements.get("maximum_drawdown") is not None, "measured value required"),
            "operational_latency": self._check(measurements.get("latency_ms") is not None, "measured value required"),
            "publication_reliability": self._check(float(measurements.get("publication_failure_rate", 1.0)) <= self.config.maximum_publication_failure_rate, self.config.maximum_publication_failure_rate),
            "usage_accounting": self._check(bool(measurements.get("usage_accounting")), True),
            "sample_size": self._check(sample_count >= self.config.minimum_readiness_sample_size, self.config.minimum_readiness_sample_size),
        }
        blockers = tuple(name for name, value in checks.items() if not value["passed"])
        status = "ready_for_analytical_live" if not blockers else "not_ready"
        return ProductionReadinessReport(
            report_id=uuid5(NAMESPACE_URL, f"ten:readiness:{now.isoformat()}"),
            status=status,
            measured_checks=checks,
            sample_count=sample_count,
            blockers=blockers,
            warnings=("broker_execution_not_available", "profitability_not_guaranteed"),
            generated_at=now,
        )

    @staticmethod
    def _check(passed: bool, threshold: Any) -> dict[str, Any]:
        return {"passed": passed, "threshold": threshold}
