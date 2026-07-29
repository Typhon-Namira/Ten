"""Auditable point-in-time feature extraction from UnifiedMarketState."""

from __future__ import annotations

from datetime import UTC, datetime
from collections.abc import Callable
from typing import Any, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.app.market_state import EvidenceAvailability, UnifiedMarketState

from .models import FeatureAvailability, QuantFeatureValue, QuantFeatureVector


EXPECTED_FIELDS = ("open", "high", "low", "close", "volume", "spread")


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="python")
        return dumped if isinstance(dumped, dict) else {}
    return {}


class PointInTimeFeatureExtractor:
    """Builds a fixed-schema vector without imputing unavailable evidence."""

    def __init__(self, schema_version: str, *, clock: Callable[[], datetime] | None = None) -> None:
        self.schema_version = schema_version
        self.clock = clock or (lambda: datetime.now(UTC))

    def extract(self, state: UnifiedMarketState) -> QuantFeatureVector:
        # Revalidation makes the anti-future-leakage contract explicit at this boundary.
        state = UnifiedMarketState.model_validate(state.model_dump(mode="python"))
        features: list[QuantFeatureValue] = []
        for timeframe in ("M5", "M15"):
            item = next(
                (
                    evidence
                    for evidence in state.evidence
                    if evidence.source_timeframe == timeframe and evidence.source_engine == "market_data"
                ),
                None,
            )
            raw = _mapping(item.raw_value) if item is not None else {}
            source_ids: tuple[UUID, ...] = (item.evidence_id,) if item is not None else ()
            availability = (
                FeatureAvailability(item.availability.value)
                if item is not None
                else FeatureAvailability.UNAVAILABLE
            )
            for field in EXPECTED_FIELDS:
                value = raw.get(field)
                usable = isinstance(value, (int, float)) and not isinstance(value, bool)
                numeric_value = float(cast(int | float, value)) if usable else None
                features.append(
                    QuantFeatureValue(
                        name=f"{timeframe.lower()}.{field}",
                        value=numeric_value,
                        availability=availability if usable else FeatureAvailability.UNAVAILABLE,
                        source_evidence_ids=source_ids,
                        source_paths=(f"evidence[{item.evidence_id}].raw_value.{field}",) if item else (),
                        reason_codes=() if usable else ("source_field_unavailable",),
                    )
                )
            open_value, high, low, close = (raw.get(name) for name in ("open", "high", "low", "close"))
            ohlc_are_numeric = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (open_value, high, low, close))
            if ohlc_are_numeric:
                numeric_open = float(cast(int | float, open_value))
                numeric_high = float(cast(int | float, high))
                numeric_low = float(cast(int | float, low))
                numeric_close = float(cast(int | float, close))
            else:
                numeric_open = numeric_high = numeric_low = numeric_close = 0.0
            derived = {
                "body_return": (numeric_close - numeric_open) / numeric_open if ohlc_are_numeric and numeric_open else None,
                "range_return": (numeric_high - numeric_low) / numeric_open if ohlc_are_numeric and numeric_open else None,
            }
            for name, value in derived.items():
                features.append(
                    QuantFeatureValue(
                        name=f"{timeframe.lower()}.{name}",
                        value=float(value) if value is not None else None,
                        availability=availability if value is not None else FeatureAvailability.UNAVAILABLE,
                        source_evidence_ids=source_ids,
                        source_paths=tuple(f"evidence[{item.evidence_id}].raw_value.{field}" for field in ("open", "high", "low", "close")) if item else (),
                        reason_codes=() if value is not None else ("derived_inputs_unavailable",),
                    )
                )

        for timeframe in ("M5", "M15"):
            scoped = [item for item in state.evidence if item.source_timeframe == timeframe]
            for item in scoped:
                feature_availability = FeatureAvailability(item.availability.value)
                features.append(
                    QuantFeatureValue(
                        name=f"{timeframe.lower()}.{item.source_engine}.availability",
                        value=item.availability.value if feature_availability != FeatureAvailability.UNAVAILABLE else None,
                        availability=feature_availability,
                        source_evidence_ids=(item.evidence_id,),
                        source_paths=(f"evidence[{item.evidence_id}].availability",),
                        reason_codes=item.reason_codes,
                    )
                )
            confidences = [item.confidence for item in scoped if item.confidence is not None]
            qualities = [item.quality for item in scoped if item.quality is not None]
            for name, value in (
                ("evidence_completeness", sum(item.availability == EvidenceAvailability.AVAILABLE for item in scoped) / len(scoped) if scoped else None),
                ("mean_confidence", sum(confidences) / len(confidences) if confidences else None),
                ("mean_quality", sum(qualities) / len(qualities) if qualities else None),
            ):
                features.append(
                    QuantFeatureValue(
                        name=f"{timeframe.lower()}.{name}",
                        value=float(value) if value is not None else None,
                        availability=FeatureAvailability.AVAILABLE if value is not None else FeatureAvailability.UNAVAILABLE,
                        source_evidence_ids=tuple(item.evidence_id for item in scoped),
                        source_paths=(f"state.evidence[{timeframe}]",),
                        reason_codes=() if value is not None else ("timeframe_evidence_unavailable",),
                    )
                )
        vector_id = uuid5(NAMESPACE_URL, f"ten:{self.schema_version}:{state.state_id}")
        return QuantFeatureVector(
            schema_version=self.schema_version,
            vector_id=vector_id,
            market_state_id=state.state_id,
            instrument=state.instrument,
            point_in_time=state.market_data_boundary,
            features=tuple(features),
            created_at=max(self.clock(), state.market_data_boundary),
        )
