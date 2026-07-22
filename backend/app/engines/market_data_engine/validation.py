"""Deterministic validation, anomaly classification, and explicit quality scoring."""

from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from statistics import median

from pydantic import BaseModel, Field

from .config import ValidationConfig
from .exceptions import MarketDataValidationError
from .models import Candle, DataQualityLevel, Timeframe


class AnomalyType(StrEnum):
    FUTURE_TIMESTAMP = "future_timestamp"
    DUPLICATE = "duplicate"
    NON_MONOTONIC = "non_monotonic"
    OUT_OF_ORDER = "out_of_order"
    MISSING_CANDLE = "missing_candle"
    MARKET_GAP = "market_gap"
    WEEKEND_GAP = "weekend_gap"
    HOLIDAY_GAP = "holiday_gap"
    CLOCK_DRIFT = "clock_drift"
    VOLATILITY_SPIKE = "volatility_spike"
    PROVIDER_INCONSISTENCY = "provider_inconsistency"


class DataAnomaly(BaseModel):
    type: AnomalyType
    timestamp: datetime
    severity: int = Field(ge=1, le=3)
    detail: str
    missing_count: int = Field(default=0, ge=0)


class ValidationReport(BaseModel):
    candles: list[Candle]
    anomalies: list[DataAnomaly]
    valid: bool


class MarketDataValidator:
    def __init__(self, config: ValidationConfig | None = None, holidays: set[date] | None = None) -> None:
        self.config = config or ValidationConfig()
        self.holidays = holidays or set()

    def validate(self, candles: list[Candle], *, now: datetime | None = None) -> ValidationReport:
        if not candles:
            return ValidationReport(candles=[], anomalies=[], valid=True)
        current = (now or datetime.now(UTC)).astimezone(UTC)
        anomalies: list[DataAnomaly] = []
        seen: set[tuple[str, Timeframe, datetime]] = set()
        previous: Candle | None = None
        ranges = [item.high - item.low for item in candles]
        typical_range = median(ranges) if ranges else 0.0
        for candle in candles:
            key = (candle.symbol, candle.timeframe, candle.timestamp)
            if key in seen:
                raise MarketDataValidationError(f"duplicate candle at {candle.timestamp.isoformat()}")
            seen.add(key)
            if candle.timestamp > current + timedelta(seconds=self.config.future_tolerance_seconds):
                raise MarketDataValidationError(f"future candle at {candle.timestamp.isoformat()}")
            candle_age = (current - candle.timestamp).total_seconds()
            if candle_age <= self.config.clock_drift_recency_window_seconds:
                drift = abs((candle.ingestion_timestamp - candle.timestamp).total_seconds())
                if drift > self.config.clock_drift_tolerance_seconds:
                    anomalies.append(DataAnomaly(type=AnomalyType.CLOCK_DRIFT, timestamp=candle.timestamp, severity=1, detail=f"ingestion clock drift {drift:.3f}s"))
            if previous is not None:
                if candle.timestamp <= previous.timestamp:
                    raise MarketDataValidationError(f"non-monotonic candle at {candle.timestamp.isoformat()}")
                expected = previous.timestamp + candle.timeframe.duration
                if candle.timestamp > expected:
                    missing = max(1, int((candle.timestamp - previous.timestamp) / candle.timeframe.duration) - 1)
                    anomaly_type = self._gap_type(previous.timestamp, candle.timestamp)
                    anomalies.append(
                        DataAnomaly(
                            type=anomaly_type,
                            timestamp=expected,
                            severity=1 if anomaly_type in {AnomalyType.WEEKEND_GAP, AnomalyType.HOLIDAY_GAP} else 2,
                            detail=f"{missing} expected candle(s) absent",
                            missing_count=missing,
                        )
                    )
            if typical_range > 0 and candle.high - candle.low > typical_range * self.config.volatility_spike_multiplier:
                anomalies.append(DataAnomaly(type=AnomalyType.VOLATILITY_SPIKE, timestamp=candle.timestamp, severity=2, detail="range exceeded deterministic spike threshold"))
            previous = candle
        scored = [self.score(candle, [item for item in anomalies if item.timestamp == candle.timestamp]) for candle in candles]
        return ValidationReport(candles=scored, anomalies=anomalies, valid=not any(item.severity == 3 for item in anomalies))

    def compare(self, primary: list[Candle], secondary: list[Candle], tolerance: float | None = None, *, quarantine_tolerance: float | None = None) -> tuple[list[DataAnomaly], set[datetime]]:
        """Cross-source validation. A close-price deviation beyond `tolerance` is flagged as a
        `PROVIDER_INCONSISTENCY` anomaly but the candle is still served; a deviation beyond the
        much larger `quarantine_tolerance` is an implausible outlier — its timestamp is returned in
        the second element so the caller can drop that candle entirely rather than silently
        shipping bad data downstream (never fabricated back in; the caller marks it a gap)."""
        resolved_tolerance = tolerance if tolerance is not None else self.config.cross_source_tolerance
        resolved_quarantine_tolerance = quarantine_tolerance if quarantine_tolerance is not None else self.config.cross_source_quarantine_tolerance
        other = {item.timestamp: item for item in secondary}
        anomalies: list[DataAnomaly] = []
        quarantined: set[datetime] = set()
        for candle in primary:
            peer = other.get(candle.timestamp)
            if peer is None:
                continue
            deviation = abs(candle.close - peer.close) / max(candle.close, peer.close)
            if deviation > resolved_quarantine_tolerance:
                quarantined.add(candle.timestamp)
                anomalies.append(
                    DataAnomaly(
                        type=AnomalyType.PROVIDER_INCONSISTENCY,
                        timestamp=candle.timestamp,
                        severity=3,
                        detail=f"quarantined: {candle.provider} close deviates {deviation:.4%} from {peer.provider} (exceeds {resolved_quarantine_tolerance:.4%})",
                        missing_count=1,
                    )
                )
            elif deviation > resolved_tolerance:
                anomalies.append(DataAnomaly(type=AnomalyType.PROVIDER_INCONSISTENCY, timestamp=candle.timestamp, severity=2, detail=f"{candle.provider} close differs from {peer.provider} by {deviation:.4%}"))
        return anomalies, quarantined

    @staticmethod
    def score(candle: Candle, anomalies: list[DataAnomaly], *, recovered: bool = False, interpolated: bool = False, verified: bool = False) -> Candle:
        if interpolated:
            score, level = 90.0, DataQualityLevel.INTERPOLATED
        elif recovered:
            score, level = 95.0, DataQualityLevel.RECOVERED
        elif verified:
            score, level = 98.0, DataQualityLevel.VERIFIED
        elif not anomalies and candle.quality_level != DataQualityLevel.NATIVE:
            return candle
        elif not anomalies:
            score, level = 100.0, DataQualityLevel.NATIVE
        else:
            severity = max(item.severity for item in anomalies)
            score, level = {
                1: (80.0, DataQualityLevel.MINOR_ANOMALY),
                2: (60.0, DataQualityLevel.MAJOR_ANOMALY),
                3: (40.0, DataQualityLevel.CORRUPTED),
            }[severity]
        return candle.model_copy(update={"quality_score": score, "quality_level": level})

    def _gap_type(self, start: datetime, end: datetime) -> AnomalyType:
        cursor = start + timedelta(days=1)
        while cursor.date() <= end.date():
            if cursor.date() in self.holidays:
                return AnomalyType.HOLIDAY_GAP
            cursor += timedelta(days=1)
        if start.weekday() >= 4 and end.weekday() <= 1:
            return AnomalyType.WEEKEND_GAP
        return AnomalyType.MARKET_GAP
