"""Failure-isolated orchestration for Phase 2 shadow forecasts."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from backend.app.market_state import UnifiedMarketState

from .config import QuantForecastingConfig
from .features import PointInTimeFeatureExtractor
from .models import (
    CalibrationStatus,
    ForecastMode,
    ForecastStatus,
    QuantForecastRequest,
    QuantForecastResult,
)
from .provider import QuantModelProvider
from .repository import QuantForecastRepository


class QuantForecastService:
    def __init__(
        self,
        repository: QuantForecastRepository,
        provider: QuantModelProvider,
        extractor: PointInTimeFeatureExtractor,
        config: QuantForecastingConfig,
        *,
        enabled: bool,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.provider = provider
        self.extractor = extractor
        self.config = config
        self.enabled = enabled
        self.clock = clock or (lambda: datetime.now(UTC))

    async def forecast(self, state: UnifiedMarketState) -> QuantForecastResult | None:
        if not self.enabled:
            return None
        request_id = uuid5(
            NAMESPACE_URL,
            f"ten:quant-request:{state.state_id}:{self.config.model_name}:{self.config.model_version}",
        )
        request = QuantForecastRequest(
            request_id=request_id,
            market_state_id=state.state_id,
            cycle_id=state.cycle_id,
            instrument=state.instrument,
            point_in_time=state.market_data_boundary,
            requested_horizons=self.config.horizons,
            feature_schema_version=self.config.feature_schema_version,
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            mode=ForecastMode.REPLAY if state.mode == "replay" else ForecastMode.SHADOW,
            data_quality_status=state.status.value,
            created_at=max(self.clock(), state.market_data_boundary),
        )
        await self.repository.save_model_metadata(self.provider.metadata())
        await self.repository.save_request(request)
        try:
            features = self.extractor.extract(state)
            await self.repository.save_features(features)
            result = await self.provider.predict(request, features)
        except Exception as exc:
            result = QuantForecastResult(
                result_id=uuid5(NAMESPACE_URL, f"ten:quant-result:{request_id}:failed"),
                request_id=request.request_id,
                market_state_id=request.market_state_id,
                cycle_id=request.cycle_id,
                instrument=request.instrument,
                point_in_time=request.point_in_time,
                status=ForecastStatus.FAILED,
                mode=request.mode,
                model_name=request.model_name,
                model_version=request.model_version,
                training_dataset_version="unavailable_provider_failed",
                feature_schema_version=request.feature_schema_version,
                calibration_version="unavailable",
                model_kind="provider_failure",
                calibration_status=CalibrationStatus.UNAVAILABLE,
                reason_codes=("shadow_provider_failed", type(exc).__name__),
                metadata={"use": "shadow_only", "publication": "not_approved_for_publication"},
                generated_at=max(self.clock(), request.point_in_time),
            )
        await self.repository.save_result(result)
        return result

    def health(self) -> dict[str, object]:
        provider = self.provider.health()
        return {
            "enabled": self.enabled,
            "status": "disabled" if not self.enabled else provider.status,
            "ready": self.enabled and provider.ready,
            "model": provider.model_name,
            "version": provider.model_version,
            "calibration_status": provider.calibration_status.value,
            "shadow_only": True,
            "approved_for_publication": False,
            "detail": provider.detail,
        }
