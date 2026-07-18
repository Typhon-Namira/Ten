"""Deterministic, no-lookahead regime synthesis over public upstream observations."""

from collections import defaultdict
from datetime import datetime
from math import exp, log
from statistics import fmean

from backend.app.engines.market_data_engine import Candle

from .config import MarketRegimeConfig
from .contracts import MarketRegimeContext
from .models import (
    AuctionRegime,
    CrossSessionRegimeState,
    DegradationState,
    DominantRegime,
    EvidenceDirection,
    EvidenceFamily,
    ExpansionRegime,
    InventoryRegime,
    MarketRegimeEvidence,
    MarketRegimeSnapshot,
    MultiTimeframeRegimeState,
    ParticipationRegime,
    ProcessingMode,
    RegimeLifecycle,
    RegimePersistence,
    StructuralRegime,
    TransitionState,
    TrendMaturity,
    TrendRegime,
    VolatilityRegime,
    stable_id,
)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class BaselineMarketRegimeAnalyzer:
    name = "market_regime"
    version = "1.0.0"

    def __init__(self, config: MarketRegimeConfig | None = None) -> None:
        self.config = config or MarketRegimeConfig()

    def analyze_snapshot(
        self,
        context: MarketRegimeContext,
        mode: ProcessingMode = ProcessingMode.SNAPSHOT,
        previous: MarketRegimeSnapshot | None = None,
        repository_mode: str = "memory",
        recovery_state: str = "clean_start",
    ) -> MarketRegimeSnapshot:
        boundary = context.analysis_boundary or (context.candles[-1].timestamp if context.candles else None)
        if boundary is None:
            raise ValueError("analysis boundary is required")
        candles = tuple(sorted((item for item in context.candles if item.timestamp <= boundary), key=lambda item: item.timestamp))
        generated = self._market_evidence(candles, boundary)
        evidence = self._align_and_normalize((*context.evidence, *generated), boundary)
        accepted = tuple(item for item in evidence if item.accepted)
        enough = len(candles) >= self.config.thresholds.minimum_candles
        quality = clamp(fmean([item.source_quality for item in accepted]) if accepted else 0)
        sources = {item.source_engine for item in accepted}
        families = {item.family for item in accepted}
        source_diversity = clamp(len(sources) / 5)
        family_diversity = clamp(len(families) / 6)

        bullish = sum(item.effective_weight for item in accepted if item.direction == EvidenceDirection.BULLISH)
        bearish = sum(item.effective_weight for item in accepted if item.direction == EvidenceDirection.BEARISH)
        neutral = sum(item.effective_weight for item in accepted if item.direction in {EvidenceDirection.NEUTRAL, EvidenceDirection.UNKNOWN})
        total = bullish + bearish + neutral
        bullish_score = clamp(bullish / total if total else 0)
        bearish_score = clamp(bearish / total if total else 0)
        neutral_score = clamp(neutral / total if total else 1)
        net = clamp((bullish - bearish) / max(bullish + bearish, 1e-9), -1, 1) if bullish + bearish else 0.0
        conflict = clamp(2 * min(bullish, bearish) / max(bullish + bearish, 1e-9)) if bullish + bearish else 0.0

        market = self._market_metrics(candles)
        volatility_score = market["volatility_score"]
        percentile = market["volatility_percentile"]
        expansion_score = market["expansion_score"]
        compression_score = market["compression_score"]
        balance_score = clamp((market["overlap"] + neutral_score + (1 - abs(net))) / 3)
        profile_direction = self._family_direction(accepted, EvidenceFamily.VOLUME_PROFILE)
        imbalance_score = clamp((abs(net) + abs(profile_direction) + expansion_score) / 3)
        trend_strength = clamp((abs(net) + market["directional_efficiency"] + max(0.0, 1 - market["overlap"])) / 3)

        trend = self._trend(enough, net, trend_strength, balance_score, families)
        volatility = self._volatility(enough, percentile, market["volatility_change"])
        auction = self._auction(enough, balance_score, imbalance_score, net, EvidenceFamily.VOLUME_PROFILE in families)
        expansion = self._expansion(enough, compression_score, expansion_score, previous)
        structure = self._structure(enough, trend, conflict)
        participation = self._participation(enough, net, accepted)
        inventory = self._inventory(enough, accepted)
        persistence = self._persistence(enough, previous, net, conflict)
        maturity = self._maturity(enough, trend, persistence, previous, accepted)
        dominant = self._dominant(enough, trend, volatility, auction, expansion, inventory, net)

        transition_score = clamp(
            conflict * 0.4 + (1 - (previous.confidence if previous else 0)) * 0.2 + (1.0 if previous and previous.dominant_regime != dominant else 0.0) * 0.4
        )
        transition_state = self._transition_state(previous, dominant, transition_score)
        lifecycle = self._lifecycle(enough, previous, persistence, transition_state)
        transition_started = (
            boundary
            if transition_state in {TransitionState.WATCH, TransitionState.DEVELOPING, TransitionState.CONFIRMED}
            and (not previous or previous.transition_state == TransitionState.NONE)
            else previous.transition_started_at
            if previous
            else None
        )
        transition_confirmed = boundary if transition_state == TransitionState.CONFIRMED else None

        mtf = context.multi_timeframe or MultiTimeframeRegimeState(
            requested_timeframe=candles[-1].timeframe.value if candles else "unknown",
            unavailable_timeframes=self.config.multi_timeframe.hierarchy,
            alignment_score=0,
            conflict_score=0,
            directional_alignment=0,
            volatility_alignment=0,
            auction_alignment=0,
            confidence=0,
            ambiguity=1,
            explanation="No time-valid multi-timeframe context was supplied.",
        )
        session = context.cross_session or CrossSessionRegimeState(
            current_session="unknown",
            continuation_score=0,
            handoff_score=0,
            reversal_score=0,
            session_alignment="no_prior_session_context",
            confidence=0,
            ambiguity=1,
            explanation="No completed prior-session context was supplied.",
        )
        missing = tuple(sorted(set(context.missing_dependencies)))
        failed = tuple(sorted(set(context.failed_dependencies)))
        degradation_penalty = clamp(0.15 * len(missing) + 0.2 * len(failed))
        correlation_penalty = clamp(sum(1 - item.correlation_discount for item in accepted) / max(len(accepted), 1))
        contradiction_penalty = conflict * 0.35
        missing_penalty = degradation_penalty
        confidence_components = {
            "evidence_strength_component": clamp(fmean([item.effective_weight for item in accepted]) if accepted else 0),
            "evidence_quality_component": quality,
            "source_diversity_component": source_diversity,
            "family_diversity_component": family_diversity,
            "temporal_alignment_component": clamp(sum(item.available_at <= boundary for item in evidence) / max(len(evidence), 1)),
            "persistence_component": 0
            if persistence == RegimePersistence.INSUFFICIENT_DATA
            else 0.8
            if persistence in {RegimePersistence.PERSISTENT, RegimePersistence.STABLE, RegimePersistence.STRENGTHENING}
            else 0.4,
            "multi_timeframe_alignment_component": mtf.alignment_score,
            "session_alignment_component": session.confidence,
            "contradiction_penalty": contradiction_penalty,
            "correlation_penalty": correlation_penalty,
            "missing_data_penalty": missing_penalty,
            "instability_penalty": clamp(transition_score * 0.25),
        }
        positives = [confidence_components[key] for key in confidence_components if key.endswith("component")]
        penalties = sum(value for key, value in confidence_components.items() if key.endswith("penalty"))
        confidence = clamp((fmean(positives) if positives else 0) - penalties)
        if not enough:
            confidence = 0.0
        ambiguity_components = {
            "directional_conflict": conflict,
            "timeframe_conflict": mtf.conflict_score,
            "session_conflict": session.reversal_score,
            "missing_sources": clamp((len(missing) + len(failed)) / 5),
            "low_diversity": 1 - family_diversity,
            "transition_instability": transition_score,
        }
        ambiguity = clamp(fmean(ambiguity_components.values()))
        direction = (
            EvidenceDirection.BULLISH
            if net > self.config.thresholds.trend
            else EvidenceDirection.BEARISH
            if net < -self.config.thresholds.trend
            else EvidenceDirection.NEUTRAL
        )
        primary = self._interpretation(dominant, direction, volatility, auction)
        alternative = self._alternative(dominant, direction, balance_score, conflict)
        degradation = DegradationState(
            is_degraded=bool(missing or failed or not enough),
            missing_dependencies=missing,
            failed_dependencies=failed,
            degradation_reasons=tuple(
                (["insufficient market-data history"] if not enough else [])
                + [f"missing optional dependency: {item}" for item in missing]
                + [f"failed optional dependency: {item}" for item in failed]
            ),
            confidence_penalty=degradation_penalty,
        )
        symbol = candles[-1].symbol.replace("/", "") if candles else "UNKNOWN"
        timeframe = candles[-1].timeframe if candles else context.evidence[0].timeframe
        identifier = stable_id(
            "snapshot",
            symbol,
            timeframe,
            boundary,
            self.config.version,
            self.config.versions.algorithm_version,
            dominant,
            *(item.evidence_id for item in evidence),
        )
        return MarketRegimeSnapshot(
            snapshot_id=identifier,
            engine_version=self.version,
            schema_version=self.config.versions.schema_version,
            configuration_version=self.config.version,
            algorithm_version=self.config.versions.algorithm_version,
            symbol=symbol,
            timeframe=timeframe,
            analysis_timestamp=boundary,
            historical_boundary=boundary,
            created_at=boundary,
            dominant_regime=dominant,
            trend_regime=trend,
            volatility_regime=volatility,
            auction_regime=auction,
            expansion_regime=expansion,
            structural_regime=structure,
            participation_regime=participation,
            inventory_regime=inventory,
            lifecycle=lifecycle,
            persistence=persistence,
            trend_maturity=maturity,
            directional_bias=direction,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            neutral_score=neutral_score,
            net_directional_score=net,
            balance_score=balance_score,
            imbalance_score=imbalance_score,
            compression_score=compression_score,
            expansion_score=expansion_score,
            trend_strength=trend_strength,
            volatility_score=volatility_score,
            volatility_percentile=percentile,
            confidence=confidence,
            quality=quality,
            ambiguity=ambiguity,
            conflict_score=conflict,
            evidence_diversity=family_diversity,
            source_diversity=source_diversity,
            primary_interpretation=primary,
            alternative_interpretation=alternative,
            reasoning_summary=f"The available evidence is consistent with {dominant.value}; {len(accepted)} of {len(evidence)} observations were time-valid.",
            transition_state=transition_state,
            previous_dominant_regime=previous.dominant_regime if previous else None,
            transition_score=transition_score,
            transition_started_at=transition_started,
            transition_confirmed_at=transition_confirmed,
            multi_timeframe=mtf,
            cross_session=session,
            evidence=evidence,
            confidence_components=confidence_components,
            ambiguity_components=ambiguity_components,
            metrics={"candles": len(candles), "accepted_evidence": len(accepted), "correlation_groups": len({item.correlation_group for item in accepted})},
            degradation=degradation,
            repository_mode=repository_mode,
            recovery_state=recovery_state,
            processing_mode=mode,
        )

    def _market_evidence(self, candles: tuple[Candle, ...], boundary: datetime) -> tuple[MarketRegimeEvidence, ...]:
        if len(candles) < 2:
            return ()
        metrics = self._market_metrics(candles)
        last = candles[-1]
        direction = (
            EvidenceDirection.BULLISH
            if last.close > candles[0].close
            else EvidenceDirection.BEARISH
            if last.close < candles[0].close
            else EvidenceDirection.NEUTRAL
        )
        values = (
            (EvidenceFamily.MARKET_DATA, "directional_efficiency", metrics["directional_efficiency"], direction, "price-path efficiency"),
            (EvidenceFamily.VOLATILITY, "normalized_range", metrics["volatility_score"], EvidenceDirection.NEUTRAL, "historically normalized realized range"),
            (EvidenceFamily.MARKET_DATA, "candle_overlap", metrics["overlap"], EvidenceDirection.NEUTRAL, "candle overlap and balance"),
            (EvidenceFamily.VOLATILITY, "range_expansion", metrics["expansion_score"], direction, "range expansion rate"),
            (EvidenceFamily.VOLATILITY, "range_contraction", metrics["compression_score"], EvidenceDirection.NEUTRAL, "range contraction rate"),
        )
        return tuple(
            MarketRegimeEvidence(
                evidence_id=stable_id("market", last.symbol, last.timeframe, boundary, subfamily),
                source_engine="market_data",
                source_engine_version="1.0.0",
                source_object_type="candle_window",
                source_object_id=f"{candles[0].timestamp.isoformat()}..{boundary.isoformat()}:{len(candles)}",
                symbol=last.symbol.replace("/", ""),
                timeframe=last.timeframe,
                event_timestamp=last.timestamp,
                available_at=max(last.timestamp, last.ingestion_timestamp if last.ingestion_timestamp <= boundary else last.timestamp),
                analysis_boundary=boundary,
                direction=item_direction,
                family=family,
                subfamily=subfamily,
                raw_strength=strength,
                normalized_strength=clamp(strength),
                source_confidence=clamp(len(candles) / self.config.thresholds.minimum_candles),
                source_quality=clamp(fmean(item.quality_score for item in candles) / 100),
                effective_weight=0,
                correlation_group=f"market-window:{last.timestamp.isoformat()}",
                correlation_discount=1,
                decay_factor=1,
                payload_summary=summary,
            )
            for family, subfamily, strength, item_direction, summary in values
        )

    def _align_and_normalize(self, items: tuple[MarketRegimeEvidence, ...], boundary: datetime) -> tuple[MarketRegimeEvidence, ...]:
        ordered = sorted(items, key=lambda item: (item.available_at, item.source_engine, item.family, item.subfamily, str(item.evidence_id)))[
            : self.config.evidence.maximum_items
        ]
        totals: dict[str, float] = defaultdict(float)
        result = []
        for item in ordered:
            future = item.available_at > boundary
            age_seconds = max(0.0, (boundary - item.available_at).total_seconds())
            candle_seconds = max(60.0, item.timeframe.duration.total_seconds())
            decay = exp(-log(2) * (age_seconds / candle_seconds) / self.config.evidence.decay_half_life_candles)
            family_weight = getattr(self.config.weights, item.family.value)
            original = clamp(item.normalized_strength * item.source_confidence * item.source_quality * family_weight * decay)
            remaining = max(0.0, self.config.evidence.correlation_group_cap - totals[item.correlation_group])
            effective = min(original, remaining)
            discount = effective / original if original else 1.0
            if not future:
                totals[item.correlation_group] += effective
            result.append(
                item.model_copy(
                    update={
                        "analysis_boundary": boundary,
                        "effective_weight": 0.0 if future else effective,
                        "correlation_discount": discount,
                        "decay_factor": clamp(decay),
                        "accepted": not future and item.source_quality >= self.config.evidence.minimum_quality,
                        "rejected": future or item.source_quality < self.config.evidence.minimum_quality,
                        "discounted": discount < 0.999,
                        "unavailable": future,
                        "rejection_reason": "available after analysis boundary"
                        if future
                        else "source quality below minimum"
                        if item.source_quality < self.config.evidence.minimum_quality
                        else None,
                    }
                )
            )
        return tuple(result)

    @staticmethod
    def _market_metrics(candles: tuple[Candle, ...]) -> dict[str, float]:
        if len(candles) < 2:
            return {
                key: 0.0
                for key in (
                    "volatility_score",
                    "volatility_percentile",
                    "volatility_change",
                    "expansion_score",
                    "compression_score",
                    "overlap",
                    "directional_efficiency",
                )
            }
        ranges = [(item.high - item.low) / item.close for item in candles]
        current = fmean(ranges[-min(5, len(ranges)) :])
        history = sorted(ranges[:-1]) or [current]
        percentile = sum(value <= current for value in history) / len(history)
        baseline = fmean(ranges[: max(1, len(ranges) // 2)])
        change = (current - baseline) / max(baseline, 1e-9)
        path = sum(abs(right.close - left.close) for left, right in zip(candles, candles[1:], strict=False))
        efficiency = abs(candles[-1].close - candles[0].close) / max(path, 1e-9)
        overlaps = []
        for left, right in zip(candles, candles[1:], strict=False):
            intersection = max(0.0, min(left.high, right.high) - max(left.low, right.low))
            union = max(left.high, right.high) - min(left.low, right.low)
            overlaps.append(intersection / union if union else 1.0)
        volatility_score = clamp(current / max(max(history), 1e-9))
        return {
            "volatility_score": volatility_score,
            "volatility_percentile": clamp(percentile),
            "volatility_change": clamp(abs(change)),
            "expansion_score": clamp(max(0.0, change)),
            "compression_score": clamp(max(0.0, -change)),
            "overlap": clamp(fmean(overlaps)),
            "directional_efficiency": clamp(efficiency),
        }

    @staticmethod
    def _family_direction(evidence: tuple[MarketRegimeEvidence, ...], family: EvidenceFamily) -> float:
        values = [
            item.effective_weight
            if item.direction == EvidenceDirection.BULLISH
            else -item.effective_weight
            if item.direction == EvidenceDirection.BEARISH
            else 0
            for item in evidence
            if item.family == family
        ]
        return clamp(sum(values) / max(sum(abs(value) for value in values), 1e-9), -1, 1) if values else 0.0

    def _trend(self, enough: bool, net: float, strength: float, balance: float, families: set[EvidenceFamily]) -> TrendRegime:
        if not enough:
            return TrendRegime.INSUFFICIENT_DATA
        if balance >= self.config.thresholds.balance:
            return TrendRegime.RANGE
        if len(families) < 2:
            return TrendRegime.UNCERTAIN
        if net >= self.config.thresholds.trend and strength >= self.config.thresholds.trend:
            return TrendRegime.BULL_TREND
        if net <= -self.config.thresholds.trend and strength >= self.config.thresholds.trend:
            return TrendRegime.BEAR_TREND
        return TrendRegime.NEUTRAL

    def _volatility(self, enough: bool, percentile: float, change: float) -> VolatilityRegime:
        if not enough:
            return VolatilityRegime.INSUFFICIENT_DATA
        if change > 0.7:
            return VolatilityRegime.UNSTABLE
        if percentile >= 0.95:
            return VolatilityRegime.VERY_HIGH
        if percentile >= self.config.thresholds.high_volatility_percentile:
            return VolatilityRegime.HIGH
        if percentile <= 0.05:
            return VolatilityRegime.VERY_LOW
        if percentile <= self.config.thresholds.low_volatility_percentile:
            return VolatilityRegime.LOW
        return VolatilityRegime.NORMAL

    def _auction(self, enough: bool, balance: float, imbalance: float, net: float, profile_available: bool) -> AuctionRegime:
        if not enough:
            return AuctionRegime.INSUFFICIENT_DATA
        if not profile_available:
            return AuctionRegime.UNCERTAIN
        if balance >= self.config.thresholds.balance:
            return AuctionRegime.BALANCED_AUCTION
        if imbalance >= 0.5 and net > 0:
            return AuctionRegime.BULLISH_IMBALANCE
        if imbalance >= 0.5 and net < 0:
            return AuctionRegime.BEARISH_IMBALANCE
        return AuctionRegime.MIXED_AUCTION

    def _expansion(self, enough: bool, compression: float, expansion: float, previous: MarketRegimeSnapshot | None) -> ExpansionRegime:
        if not enough:
            return ExpansionRegime.INSUFFICIENT_DATA
        if compression >= self.config.thresholds.compression:
            return ExpansionRegime.COMPRESSION
        if expansion >= self.config.thresholds.expansion:
            if previous and previous.expansion_regime in {ExpansionRegime.EXPANSION, ExpansionRegime.LATE_EXPANSION}:
                return ExpansionRegime.LATE_EXPANSION
            return ExpansionRegime.EXPANSION if previous and previous.expansion_regime == ExpansionRegime.EARLY_EXPANSION else ExpansionRegime.EARLY_EXPANSION
        if previous and previous.expansion_score > expansion + 0.2:
            return ExpansionRegime.DECELERATION
        return ExpansionRegime.NEUTRAL

    @staticmethod
    def _structure(enough: bool, trend: TrendRegime, conflict: float) -> StructuralRegime:
        if not enough:
            return StructuralRegime.INSUFFICIENT_DATA
        if conflict > 0.65:
            return StructuralRegime.MIXED_STRUCTURE
        if trend == TrendRegime.BULL_TREND:
            return StructuralRegime.BULLISH_CONTINUATION
        if trend == TrendRegime.BEAR_TREND:
            return StructuralRegime.BEARISH_CONTINUATION
        if trend == TrendRegime.RANGE:
            return StructuralRegime.RANGE_STRUCTURE
        return StructuralRegime.STRUCTURAL_TRANSITION

    @staticmethod
    def _participation(enough: bool, net: float, evidence: tuple[MarketRegimeEvidence, ...]) -> ParticipationRegime:
        if not enough:
            return ParticipationRegime.INSUFFICIENT_DATA
        flow = [item for item in evidence if item.family == EvidenceFamily.INSTITUTIONAL_FLOW]
        if not flow:
            return ParticipationRegime.UNCERTAIN
        if any(item.contradicting for item in flow):
            return ParticipationRegime.CONFLICTED_PARTICIPATION
        intensity = fmean(item.effective_weight for item in flow)
        if net >= 0.45:
            return ParticipationRegime.STRONG_BULLISH_PARTICIPATION if intensity >= 0.55 else ParticipationRegime.MODERATE_BULLISH_PARTICIPATION
        if net <= -0.45:
            return ParticipationRegime.STRONG_BEARISH_PARTICIPATION if intensity >= 0.55 else ParticipationRegime.MODERATE_BEARISH_PARTICIPATION
        return ParticipationRegime.NEUTRAL_PARTICIPATION

    @staticmethod
    def _inventory(enough: bool, evidence: tuple[MarketRegimeEvidence, ...]) -> InventoryRegime:
        if not enough:
            return InventoryRegime.INSUFFICIENT_DATA
        mapping = {item.value: item for item in InventoryRegime}
        for item in evidence:
            candidate = str(item.metadata.get("campaign", ""))
            if candidate in mapping:
                return mapping[candidate]
        return InventoryRegime.AMBIGUOUS

    @staticmethod
    def _persistence(enough: bool, previous: MarketRegimeSnapshot | None, net: float, conflict: float) -> RegimePersistence:
        if not enough:
            return RegimePersistence.INSUFFICIENT_DATA
        if not previous:
            return RegimePersistence.TRANSIENT
        if previous.net_directional_score * net < 0:
            return RegimePersistence.REVERSING
        if conflict > 0.7:
            return RegimePersistence.UNSTABLE
        if abs(net) > abs(previous.net_directional_score) + 0.1:
            return RegimePersistence.STRENGTHENING
        if abs(net) + 0.1 < abs(previous.net_directional_score):
            return RegimePersistence.WEAKENING
        return RegimePersistence.STABLE if previous.dominant_regime else RegimePersistence.DEVELOPING

    @staticmethod
    def _maturity(
        enough: bool, trend: TrendRegime, persistence: RegimePersistence, previous: MarketRegimeSnapshot | None, evidence: tuple[MarketRegimeEvidence, ...]
    ) -> TrendMaturity:
        if not enough:
            return TrendMaturity.INSUFFICIENT_DATA
        if trend not in {TrendRegime.BULL_TREND, TrendRegime.BEAR_TREND}:
            return TrendMaturity.NOT_APPLICABLE
        if any(item.subfamily == "exhaustion" for item in evidence):
            return TrendMaturity.EXHAUSTION_RISK
        if persistence == RegimePersistence.WEAKENING:
            return TrendMaturity.WEAKENING
        if not previous:
            return TrendMaturity.EARLY
        if previous.trend_maturity in {TrendMaturity.ESTABLISHED, TrendMaturity.MATURE}:
            return TrendMaturity.MATURE
        return TrendMaturity.ESTABLISHED if persistence in {RegimePersistence.STABLE, RegimePersistence.STRENGTHENING} else TrendMaturity.DEVELOPING

    @staticmethod
    def _dominant(
        enough: bool,
        trend: TrendRegime,
        volatility: VolatilityRegime,
        auction: AuctionRegime,
        expansion: ExpansionRegime,
        inventory: InventoryRegime,
        net: float,
    ) -> DominantRegime:
        if not enough:
            return DominantRegime.INSUFFICIENT_DATA
        if expansion == ExpansionRegime.COMPRESSION:
            return DominantRegime.COMPRESSION
        if expansion in {ExpansionRegime.EARLY_EXPANSION, ExpansionRegime.EXPANSION, ExpansionRegime.LATE_EXPANSION}:
            return DominantRegime.EXPANSION_BULL if net > 0 else DominantRegime.EXPANSION_BEAR if net < 0 else DominantRegime.HIGH_VOLATILITY
        if trend == TrendRegime.BULL_TREND:
            return DominantRegime.TRENDING_BULL
        if trend == TrendRegime.BEAR_TREND:
            return DominantRegime.TRENDING_BEAR
        if auction == AuctionRegime.BALANCED_AUCTION:
            return DominantRegime.BALANCED
        if auction == AuctionRegime.BULLISH_IMBALANCE:
            return DominantRegime.IMBALANCED_BULL
        if auction == AuctionRegime.BEARISH_IMBALANCE:
            return DominantRegime.IMBALANCED_BEAR
        campaign_map = {
            InventoryRegime.ACCUMULATION_LIKE: DominantRegime.ACCUMULATION_LIKE,
            InventoryRegime.DISTRIBUTION_LIKE: DominantRegime.DISTRIBUTION_LIKE,
            InventoryRegime.REACCUMULATION_LIKE: DominantRegime.REACCUMULATION_LIKE,
            InventoryRegime.REDISTRIBUTION_LIKE: DominantRegime.REDISTRIBUTION_LIKE,
        }
        if inventory in campaign_map:
            return campaign_map[inventory]
        if volatility in {VolatilityRegime.HIGH, VolatilityRegime.VERY_HIGH, VolatilityRegime.UNSTABLE}:
            return DominantRegime.HIGH_VOLATILITY
        if volatility in {VolatilityRegime.LOW, VolatilityRegime.VERY_LOW}:
            return DominantRegime.LOW_VOLATILITY
        return DominantRegime.RANGING if trend == TrendRegime.RANGE else DominantRegime.UNCERTAIN

    def _transition_state(self, previous: MarketRegimeSnapshot | None, dominant: DominantRegime, score: float) -> TransitionState:
        if not previous or previous.dominant_regime == dominant:
            return TransitionState.NONE
        if score >= self.config.thresholds.transition_confirm and previous.transition_state in {TransitionState.DEVELOPING, TransitionState.WATCH}:
            return TransitionState.CONFIRMED
        if score >= self.config.thresholds.transition_confirm:
            return TransitionState.DEVELOPING
        if score >= self.config.thresholds.transition_watch:
            return TransitionState.WATCH
        return TransitionState.FAILED

    @staticmethod
    def _lifecycle(enough: bool, previous: MarketRegimeSnapshot | None, persistence: RegimePersistence, transition: TransitionState) -> RegimeLifecycle:
        if not enough:
            return RegimeLifecycle.INSUFFICIENT_DATA
        if transition in {TransitionState.WATCH, TransitionState.DEVELOPING, TransitionState.CONFIRMED}:
            return RegimeLifecycle.TRANSITIONING
        if not previous:
            return RegimeLifecycle.INITIAL
        if persistence == RegimePersistence.WEAKENING:
            return RegimeLifecycle.WEAKENING
        if persistence in {RegimePersistence.PERSISTENT, RegimePersistence.STABLE}:
            return RegimeLifecycle.MATURE
        return RegimeLifecycle.DEVELOPING

    @staticmethod
    def _interpretation(dominant: DominantRegime, direction: EvidenceDirection, volatility: VolatilityRegime, auction: AuctionRegime) -> str:
        return f"The dominant interpretation is probabilistic {dominant.value}, with {direction.value} directional context, {volatility.value} volatility, and {auction.value}."

    @staticmethod
    def _alternative(dominant: DominantRegime, direction: EvidenceDirection, balance: float, conflict: float) -> str:
        if dominant == DominantRegime.INSUFFICIENT_DATA:
            return "An alternative regime cannot be ranked until additional time-valid evidence becomes available."
        if balance >= 0.5:
            return f"An alternative interpretation remains ordinary balance with temporary {direction.value} pressure."
        if conflict >= 0.4:
            return "An alternative interpretation remains a structural transition because contradictory evidence is material."
        return "An alternative interpretation remains a transient move inside a broader balanced auction."
