"""Deterministic SMC Production 1.0 analysis orchestration."""

from abc import ABC
from datetime import UTC, datetime

from backend.app.engines.common import AnalysisEngine
from backend.app.engines.market_data_engine import Candle, Timeframe

from .advanced import AdvancedSMCAnalyzer
from .config import SMCConfig
from .context import CandleContext
from .models import (
    AnalysisStatus,
    ConfirmationMethod,
    Evidence,
    MarketStructureState,
    ProcessingMode,
    SMCAnalysisSnapshot,
    SMCResult,
    StructureDirection,
    StructureEvent,
    StructureEventType,
    StructureScope,
    stable_id,
)
from .structure import StructureAnalyzer
from .swing import SwingDetector


class SMCAnalyzer(AnalysisEngine[list[Candle], SMCResult], ABC):
    """Contract for provider-neutral SMC analyzers."""


class BaselineSMCAnalyzer(SMCAnalyzer):
    name = "smc"
    version = "3.0.0"

    def __init__(self, config: SMCConfig | None = None) -> None:
        self.config = config or SMCConfig()
        self.swings = SwingDetector(self.config)
        self.structure = StructureAnalyzer(self.config)
        self.advanced = AdvancedSMCAnalyzer(self.config)

    def analyze(self, data: list[Candle]) -> SMCResult:
        snapshot = self.analyze_snapshot(data, ProcessingMode.HISTORICAL)
        observations = ["Insufficient confirmed history for structural analysis."] if snapshot.status == AnalysisStatus.INSUFFICIENT_HISTORY else []
        gaps = [item for item in snapshot.zones if "fvg" in item.zone_type.value]
        position = "equilibrium"
        if snapshot.dealing_ranges and data:
            active_range = snapshot.dealing_ranges[-1]
            position = "premium" if data[-1].close > active_range.equilibrium else "discount" if data[-1].close < active_range.equilibrium else "equilibrium"
        return SMCResult(bias=snapshot.structure_state.current_direction, structure_events=list(snapshot.structure_events), fair_value_gaps=list(gaps), premium_discount_position=position, snapshot=snapshot, observations=observations)

    def analyze_snapshot(self, data: list[Candle], mode: ProcessingMode = ProcessingMode.HISTORICAL) -> SMCAnalysisSnapshot:
        if not data:
            timestamp = datetime(1970, 1, 1, tzinfo=UTC)
            return self._empty("XAUUSD", Timeframe.M1, timestamp, mode, AnalysisStatus.INSUFFICIENT_HISTORY, "empty")
        if len(data) < self.config.processing.minimum_history:
            latest = data[-1]
            return self._empty(latest.symbol, latest.timeframe, latest.timestamp, mode, AnalysisStatus.INSUFFICIENT_HISTORY, f"insufficient:{len(data)}")
        context = CandleContext.build(data, self.config)
        swings = self.swings.detect(context)
        state, legs, events = self.structure.analyze(context, swings)
        if not events:
            bootstrap = self._bootstrap_break(context)
            if bootstrap is not None:
                events = (bootstrap,)
                state = state.model_copy(update={"current_direction": bootstrap.direction, "external_direction": bootstrap.direction, "last_bos_id": bootstrap.id, "state_version": 1})
        displacements, zones, liquidity_references, dealing_ranges = self.advanced.analyze(context, swings, events)
        if dealing_ranges:
            state = state.model_copy(update={"active_dealing_range_id": dealing_ranges[-1].id})
        status = AnalysisStatus.DEGRADED_INPUT if context.degraded else AnalysisStatus.COMPLETE
        timestamp = context.candles[-1].timestamp
        scored = [item.confidence_score for item in events] + [item.confidence_score for item in displacements] + [item.confidence_score for item in zones] + [item.confidence_score for item in dealing_ranges]
        confidence = sum(scored) / len(scored) if scored else context.average_quality * 0.5
        return SMCAnalysisSnapshot(
            id=stable_id("snapshot", context.symbol, context.timeframe, timestamp.isoformat(), context.boundary, self.config.version, mode.value),
            symbol=context.symbol,
            timeframe=context.timeframe,
            analysis_timestamp=timestamp,
            market_data_boundary=context.boundary,
            status=status,
            processing_mode=mode,
            structure_state=state,
            swings=swings,
            structure_legs=legs,
            structure_events=events,
            displacements=displacements,
            zones=zones,
            liquidity_references=liquidity_references,
            dealing_ranges=dealing_ranges,
            confidence_summary={"overall": max(0.0, min(100.0, confidence)), "structure": self._average(events), "displacement": self._average(displacements), "zones": self._average(zones), "ranges": self._average(dealing_ranges)},
            quality_summary={"minimum": context.minimum_quality, "average": context.average_quality},
            reasoning_metadata={"methodology": "confirmed volatility-aware pivots and explicit structural-level state transitions", "no_lookahead": True},
            configuration_version=self.config.version,
            created_at=timestamp,
        )

    @staticmethod
    def _average(items: tuple[object, ...]) -> float:
        scores = [float(getattr(item, "confidence_score", 0.0)) for item in items]
        return sum(scores) / len(scores) if scores else 0.0

    def _bootstrap_break(self, context: CandleContext) -> StructureEvent | None:
        """Seed structure from a confirmed break of the pre-analysis range."""
        last = context.candles[-1]
        prior = context.candles[:-1]
        prior_high = max(item.high for item in prior)
        prior_low = min(item.low for item in prior)
        if last.close > prior_high:
            direction, level = StructureDirection.BULLISH, prior_high
        elif last.close < prior_low:
            direction, level = StructureDirection.BEARISH, prior_low
        else:
            return None
        distance = abs(last.close - level)
        broken_id = stable_id("bootstrap-level", context.symbol, context.timeframe, level)
        quality = min(item.quality_score for item in context.candles)
        return StructureEvent(
            id=stable_id("structure-event", context.symbol, context.timeframe, "bootstrap", last.timestamp.isoformat(), self.config.version),
            event_type=StructureEventType.BOS,
            symbol=context.symbol,
            timeframe=context.timeframe,
            scope=StructureScope.EXTERNAL,
            direction=direction,
            timestamp=last.timestamp,
            broken_level=level,
            broken_swing_id=broken_id,
            confirmation_candle_id=context.candle_id(len(context.candles) - 1),
            confirmation_method=ConfirmationMethod.CLOSE,
            close_confirmed=True,
            wick_confirmed=True,
            previous_direction=StructureDirection.NEUTRAL,
            resulting_direction=direction,
            break_distance=distance,
            displacement_score=self.structure.displacement(last, context.atr_at(len(context.candles) - 1)),
            confidence_score=min(100.0, 70.0 * quality / 100.0),
            quality_score=quality,
            evidence=(Evidence(code="initial_range_break", description="close broke the range established before the analysis boundary", value=distance, threshold=0.0),),
            invalidation_metadata={"rule": "opposing confirmed break of the initial range"},
            algorithm_version=self.config.algorithm_version,
            created_at=last.timestamp,
        )

    def _empty(self, symbol: str, timeframe: Timeframe, timestamp: datetime, mode: ProcessingMode, status: AnalysisStatus, boundary: str) -> SMCAnalysisSnapshot:
        state = MarketStructureState(symbol=symbol, timeframe=timeframe, current_direction=StructureDirection.NEUTRAL, updated_at=timestamp, last_processed_candle=timestamp)
        return SMCAnalysisSnapshot(
            id=stable_id("snapshot", symbol, timeframe, timestamp.isoformat(), boundary, self.config.version, mode.value),
            symbol=symbol,
            timeframe=timeframe,
            analysis_timestamp=timestamp,
            market_data_boundary=boundary,
            status=status,
            processing_mode=mode,
            structure_state=state,
            confidence_summary={"overall": 0.0, "structure": 0.0},
            quality_summary={"minimum": 0.0, "average": 0.0},
            reasoning_metadata={"minimum_history": self.config.processing.minimum_history, "received_history": 0},
            configuration_version=self.config.version,
            created_at=timestamp,
        )
