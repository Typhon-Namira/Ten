"""Transparent OHLCV-based flow estimation; never represented as CME order flow."""

from abc import ABC

from backend.app.engines.common import AnalysisEngine
from backend.app.engines.market_data_engine import Candle

from .config import InstitutionalFlowConfig
from .models import FlowBias, FlowScore


class InstitutionalFlowEngine(AnalysisEngine[list[Candle], FlowScore], ABC):
    """Contract that can later accept a licensed exchange-data implementation."""


class BaselineInstitutionalFlowEngine(InstitutionalFlowEngine):
    name = "institutional_flow_estimation"
    version = "1.0.0"

    def __init__(self, config: InstitutionalFlowConfig | None = None) -> None:
        self.config = config or InstitutionalFlowConfig()

    def analyze(self, data: list[Candle]) -> FlowScore:
        if len(data) < 2:
            return FlowScore(score=0, bias=FlowBias.BALANCED, volume_pressure=0, price_acceleration=0, delta_estimate=0, absorption_probability=0, observations=["Insufficient OHLCV history."])
        candles = sorted(data, key=lambda item: item.timestamp)
        recent = candles[-20:]
        signed_volume = [item.volume * _signed_close_location(item) for item in recent]
        gross_volume = sum(item.volume for item in recent) or 1.0
        volume_pressure = _clamp(sum(signed_volume) / gross_volume)
        returns = [(right.close - left.close) / left.close for left, right in zip(recent, recent[1:], strict=False) if left.close]
        acceleration = _clamp((returns[-1] - (sum(returns[:-1]) / max(len(returns) - 1, 1))) * 1000) if returns else 0.0
        location = _signed_close_location(recent[-1])
        score = _clamp(self.config.volume_weight * volume_pressure + self.config.acceleration_weight * acceleration + self.config.close_location_weight * location)
        bias = FlowBias.BUYING if score > 0.1 else FlowBias.SELLING if score < -0.1 else FlowBias.BALANCED
        effort = recent[-1].volume / (sum(item.volume for item in recent) / len(recent) or 1.0)
        result = abs(recent[-1].close - recent[-1].open) / max(recent[-1].high - recent[-1].low, 1e-9)
        absorption = _clamp01((effort - result) / 2)
        return FlowScore(score=score, bias=bias, volume_pressure=volume_pressure, price_acceleration=acceleration, delta_estimate=volume_pressure, absorption_probability=absorption)


def _signed_close_location(candle: Candle) -> float:
    span = candle.high - candle.low
    return 0.0 if span == 0 else _clamp(((candle.close - candle.low) / span) * 2 - 1)


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))

