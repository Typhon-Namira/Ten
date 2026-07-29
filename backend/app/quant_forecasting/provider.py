"""Provider port and deterministic infrastructure baseline."""

from __future__ import annotations

from datetime import UTC, datetime
from math import exp, sqrt
from collections.abc import Callable
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from .config import QuantForecastingConfig
from .models import (
    CalibrationStatus,
    ForecastStatus,
    HorizonPrediction,
    ModelHealth,
    ModelMetadata,
    QuantFeatureVector,
    QuantForecastRequest,
    QuantForecastResult,
    UncertaintyInterval,
)


class QuantModelProvider(Protocol):
    async def predict(self, request: QuantForecastRequest, features: QuantFeatureVector) -> QuantForecastResult: ...
    def health(self) -> ModelHealth: ...
    def metadata(self) -> ModelMetadata: ...


def _softmax(values: tuple[float, float, float]) -> tuple[float, float, float]:
    ceiling = max(values)
    terms = tuple(exp(value - ceiling) for value in values)
    total = sum(terms)
    return tuple(value / total for value in terms)  # type: ignore[return-value]


class DeterministicBaselineProvider:
    """Non-trained, uncalibrated baseline used only to validate infrastructure."""

    def __init__(self, config: QuantForecastingConfig, *, clock: Callable[[], datetime] | None = None) -> None:
        self.config = config
        self.clock = clock or (lambda: datetime.now(UTC))

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            model_kind="deterministic_baseline",
            training_dataset_version="none_infrastructure_baseline",
            feature_schema_version=self.config.feature_schema_version,
            calibration_version="unavailable",
            calibration_status=CalibrationStatus.UNCALIBRATED,
            shadow_only=True,
            approved_for_publication=False,
            deterministic=True,
        )

    def health(self) -> ModelHealth:
        return ModelHealth(
            status="degraded",
            ready=True,
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            calibration_status=CalibrationStatus.UNCALIBRATED,
            detail="deterministic infrastructure baseline; uncalibrated and not approved for publication",
        )

    async def predict(self, request: QuantForecastRequest, features: QuantFeatureVector) -> QuantForecastResult:
        result_id = uuid5(NAMESPACE_URL, f"ten:quant-result:{request.request_id}:{self.config.model_version}")
        metadata = self.metadata()
        if features.schema_version != request.feature_schema_version:
            return self._empty(result_id, request, ForecastStatus.INCOMPATIBLE_FEATURES, ("feature_schema_mismatch",))
        values = {
            item.name: float(item.value)
            for item in features.features
            if item.availability.value == "available" and isinstance(item.value, (int, float)) and not isinstance(item.value, bool)
        }
        if len(values) < self.config.minimum_numeric_features:
            return self._empty(result_id, request, ForecastStatus.INSUFFICIENT_HISTORY, ("insufficient_point_in_time_features",))
        reference_price = values.get("m5.close")
        if reference_price is None or reference_price <= 0:
            return self._empty(result_id, request, ForecastStatus.UNAVAILABLE, ("m5_reference_price_unavailable",))

        momentum = (
            values.get("m5.body_return", 0.0) * 0.60
            + values.get("m15.body_return", 0.0) * 0.40
        )
        observed_range = max(
            values.get("m5.range_return", 0.0) / sqrt(5),
            values.get("m15.range_return", 0.0) / sqrt(15),
            self.config.neutral_band,
        )
        predictions: list[HorizonPrediction] = []
        for horizon in request.requested_horizons:
            scale = sqrt(horizon.duration_seconds / 60)
            directional_signal = momentum * scale
            buy, sell, neutral = _softmax(
                (
                    directional_signal / self.config.neutral_band,
                    -directional_signal / self.config.neutral_band,
                    -abs(directional_signal) / self.config.neutral_band + 0.5,
                )
            )
            base = observed_range * scale
            expected_return = (buy - sell) * base
            predictions.append(
                HorizonPrediction(
                    horizon=horizon,
                    reference_price=reference_price,
                    buy_probability=buy,
                    sell_probability=sell,
                    neutral_probability=neutral,
                    expected_return=expected_return,
                    expected_base_movement=base,
                    expected_minimum_movement=base * 0.5,
                    expected_maximum_movement=base * 1.5,
                    expected_volatility=base,
                    expected_mfe=base,
                    expected_mae=base * 0.75,
                    tp1_probability=max(buy, sell) * 0.8,
                    tp2_probability=max(buy, sell) * 0.55,
                    stop_loss_probability=min(1.0, min(buy, sell) + neutral * 0.5),
                    sl_before_tp_probability=min(1.0, min(buy, sell) + neutral * 0.5),
                    uncertainty_interval=UncertaintyInterval(
                        low=expected_return - base,
                        high=expected_return + base,
                        confidence_level=0.80,
                    ),
                    transition_probabilities={"trend_continuation": max(buy, sell), "range": neutral, "reversal": min(buy, sell)},
                )
            )
        return QuantForecastResult(
            result_id=result_id,
            request_id=request.request_id,
            market_state_id=request.market_state_id,
            cycle_id=request.cycle_id,
            instrument=request.instrument,
            point_in_time=request.point_in_time,
            status=ForecastStatus.AVAILABLE,
            mode=request.mode,
            model_name=metadata.model_name,
            model_version=metadata.model_version,
            training_dataset_version=metadata.training_dataset_version,
            feature_schema_version=metadata.feature_schema_version,
            calibration_version=metadata.calibration_version,
            model_kind=metadata.model_kind,
            calibration_status=metadata.calibration_status,
            predictions=tuple(predictions),
            metadata={
                "label": "baseline",
                "calibration": "uncalibrated",
                "use": "shadow_only",
                "publication": "not_approved_for_publication",
                "feature_vector_id": str(features.vector_id),
            },
            generated_at=max(self.clock(), request.point_in_time),
        )

    def _empty(
        self,
        result_id: UUID,
        request: QuantForecastRequest,
        status: ForecastStatus,
        reasons: tuple[str, ...],
    ) -> QuantForecastResult:
        metadata = self.metadata()
        return QuantForecastResult(
            result_id=result_id,
            request_id=request.request_id,
            market_state_id=request.market_state_id,
            cycle_id=request.cycle_id,
            instrument=request.instrument,
            point_in_time=request.point_in_time,
            status=status,
            mode=request.mode,
            model_name=metadata.model_name,
            model_version=metadata.model_version,
            training_dataset_version=metadata.training_dataset_version,
            feature_schema_version=metadata.feature_schema_version,
            calibration_version=metadata.calibration_version,
            model_kind=metadata.model_kind,
            calibration_status=metadata.calibration_status,
            reason_codes=reasons,
            metadata={"label": "baseline", "use": "shadow_only", "publication": "not_approved_for_publication"},
            generated_at=max(self.clock(), request.point_in_time),
        )
