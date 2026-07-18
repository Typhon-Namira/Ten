from collections import defaultdict
from datetime import UTC, datetime
from statistics import fmean

from backend.app.engines.market_data_engine import Candle

from .config import InstitutionalFlowConfig
from .contracts import InstitutionalFlowContext
from .models import (
    AbsorptionInference,
    AbsorptionType,
    AccumulationDistributionInference,
    ActivityType,
    AnalysisStatus,
    CampaignPhase,
    CampaignPhaseInference,
    CorrelationGroup,
    CrossSessionFlow,
    DataQualityLevel,
    DirectionalFlowPressure,
    EvidenceRole,
    EvidenceSourceEngine,
    EvidenceType,
    ExhaustionInference,
    ExhaustionType,
    FlowDirection,
    FlowPersistence,
    FlowPersistenceState,
    FlowState,
    InitiativeActivity,
    InstitutionalFlowAnalysisSnapshot,
    InstitutionalFlowConfluence,
    InstitutionalFlowEvidence,
    InstitutionalFlowEvidenceBundle,
    InstitutionalFlowExplanation,
    InstitutionalFlowQuality,
    InstitutionalFlowState,
    InstitutionalFlowTransition,
    InventoryBehaviorType,
    LifecycleState,
    ParticipationIntensity,
    ParticipationLevel,
    ProcessingMode,
    ResponsiveActivity,
    SessionType,
    stable_id,
)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


class BaselineInstitutionalFlowAnalyzer:
    """Explainable cross-engine inference; it never asserts participant identity."""

    version = "1.0.0"

    def __init__(self, config: InstitutionalFlowConfig | None = None) -> None:
        self.config = config or InstitutionalFlowConfig()

    def market_evidence(self, candles: tuple[Candle, ...], session: SessionType = SessionType.UNKNOWN) -> tuple[InstitutionalFlowEvidence, ...]:
        if len(candles) < 2:
            return ()
        ordered = tuple(sorted(candles, key=lambda item: item.timestamp))
        recent = ordered[-20:]
        last = recent[-1]
        prior = recent[:-1]
        mean_range = fmean(max(item.high - item.low, 0.0) for item in prior) or 1e-12
        mean_volume = fmean(max(item.volume, 0.0) for item in prior) or 1e-12
        span = max(last.high - last.low, 1e-12)
        body = abs(last.close - last.open)
        direction = FlowDirection.BULLISH if last.close > last.open else FlowDirection.BEARISH if last.close < last.open else FlowDirection.NEUTRAL
        range_ratio = clamp((last.high - last.low) / mean_range / 2)
        volume_ratio = clamp(max(last.volume, 0.0) / mean_volume / 2)
        efficiency = clamp(body / span)
        closes = [item.close - item.open for item in recent]
        aligned = sum(1 for change in closes if (change > 0) == (direction == FlowDirection.BULLISH) and change != 0) / len(closes)

        def item(kind: EvidenceType, strength: float, group: CorrelationGroup, explanation: str, item_direction: FlowDirection = direction) -> InstitutionalFlowEvidence:
            return InstitutionalFlowEvidence(
                id=stable_id("market", kind, last.symbol, last.timeframe, last.timestamp),
                source_engine=EvidenceSourceEngine.MARKET_DATA,
                evidence_type=kind,
                source_object_id=f"candle:{last.symbol}:{last.timeframe.value}:{last.timestamp.isoformat()}",
                source_timestamp=last.timestamp,
                availability_timestamp=last.timestamp,
                timeframe=last.timeframe,
                session=session,
                direction=item_direction,
                strength=strength,
                confidence=0.7,
                quality=0.8 if last.volume > 0 else 0.3,
                correlation_group=group,
                explanation=explanation,
                configuration_version=self.config.version,
                engine_version=self.version,
            )

        result = [
            item(EvidenceType.RANGE_EXPANSION, range_ratio, CorrelationGroup.PRICE_ACTION, "Range expansion normalized to the recent window."),
            item(EvidenceType.VOLUME_EXPANSION, volume_ratio, CorrelationGroup.VOLUME, "Reported volume expanded relative to its recent baseline."),
            item(EvidenceType.DIRECTIONAL_PERSISTENCE, aligned, CorrelationGroup.PRICE_ACTION, "Directional candle persistence in the bounded recent window."),
        ]
        if volume_ratio >= 0.5 and efficiency <= 0.35:
            result.append(item(EvidenceType.LIMITED_PROGRESS, clamp(volume_ratio - efficiency + 0.25), CorrelationGroup.VOLUME, "Elevated reported volume produced limited candle-body progress."))
        if len(recent) >= 4:
            previous_efficiency = fmean(abs(x.close - x.open) / max(x.high - x.low, 1e-12) for x in recent[-4:-1])
            if efficiency < previous_efficiency:
                result.append(item(EvidenceType.EFFICIENCY_DECLINE, clamp(previous_efficiency - efficiency + 0.35), CorrelationGroup.PRICE_ACTION, "Directional efficiency declined versus the preceding candles."))
        return tuple(result)

    def normalize(self, evidence: tuple[InstitutionalFlowEvidence, ...], boundary: datetime) -> InstitutionalFlowEvidenceBundle:
        accepted: list[InstitutionalFlowEvidence] = []
        future: list[object] = []
        invalid: list[object] = []
        duplicate: list[object] = []
        discounted: list[object] = []
        seen: set[object] = set()
        group_totals: dict[tuple[CorrelationGroup, FlowDirection], float] = defaultdict(float)
        ordered = sorted(evidence, key=lambda item: (item.availability_timestamp, str(item.id)))
        for item in ordered[: self.config.evidence.maximum_items]:
            if item.availability_timestamp > boundary:
                future.append(item.id)
                continue
            if item.invalidated or item.quality < self.config.evidence.minimum_quality:
                invalid.append(item.id)
                continue
            key = (item.source_engine, item.source_object_id, item.evidence_type)
            if item.id in seen or key in seen:
                duplicate.append(item.id)
                continue
            seen.update((item.id, key))
            group_key = (item.correlation_group, item.direction)
            adjusted = item
            if group_totals[group_key] + item.strength > self.config.correlation.maximum_group_contribution:
                adjusted = item.model_copy(update={"strength": item.strength * self.config.correlation.correlated_discount})
                discounted.append(item.id)
            group_totals[group_key] += adjusted.strength
            accepted.append(adjusted)
        return InstitutionalFlowEvidenceBundle(
            accepted=tuple(accepted),
            rejected_future_ids=tuple(future),
            rejected_invalid_ids=tuple(invalid),
            deduplicated_ids=tuple(duplicate),
            discounted_ids=tuple(discounted),
        )

    def analyze_snapshot(
        self,
        context: InstitutionalFlowContext,
        mode: ProcessingMode = ProcessingMode.HISTORICAL,
        previous: InstitutionalFlowAnalysisSnapshot | None = None,
    ) -> InstitutionalFlowAnalysisSnapshot:
        if not context.candles:
            raise ValueError("Institutional Flow requires at least one candle")
        candles = tuple(sorted(context.candles, key=lambda item: item.timestamp))
        boundary = context.analysis_boundary or candles[-1].timestamp
        visible = tuple(item for item in candles if item.timestamp <= boundary)
        if not visible:
            raise ValueError("analysis boundary precedes all candles")
        generated = self.market_evidence(visible, context.session)
        bundle = self.normalize((*generated, *context.evidence), boundary)
        evidence = bundle.accepted
        quality = self._quality(evidence)
        pressure = self._pressure(evidence, quality.score)
        participation = self._participation(evidence, pressure, quality.score)
        initiative, responsive, activity = self._activities(evidence, pressure, boundary)
        absorption = self._absorption(evidence, pressure)
        exhaustion = self._exhaustion(evidence, pressure, boundary)
        inventory = self._inventory(evidence, pressure, absorption)
        campaign = self._campaign(inventory, initiative, exhaustion, pressure, previous)
        persistence = self._persistence(evidence, pressure, previous)
        cross_session = self._cross_session(evidence)
        confluences = self._confluences(evidence)
        conflict_ids = tuple(item.id for item in evidence if item.role == EvidenceRole.CONTRADICTING)
        alternative = inventory.alternative_interpretation or ("responsive activity" if activity == ActivityType.INITIATIVE else "initiative activity")
        explanation = InstitutionalFlowExplanation(
            summary=f"Probabilistic {campaign.phase.value} inference from {len(evidence)} time-valid evidence items; no participant identity is observed.",
            supporting_evidence_ids=tuple(item.id for item in evidence if item.role == EvidenceRole.SUPPORTING),
            contradicting_evidence_ids=conflict_ids,
            alternative_interpretation=alternative if pressure.conflict >= self.config.thresholds.conflict else None,
        )
        lifecycle = LifecycleState.DEGRADED if quality.level in {DataQualityLevel.LOW, DataQualityLevel.UNUSABLE} else LifecycleState.PERSISTENT if persistence.state == FlowPersistenceState.PERSISTENT else LifecycleState.ACTIVE
        version = (previous.state.version + 1) if previous else 1
        state_id = stable_id("state", visible[-1].symbol, visible[-1].timeframe, boundary, version, pressure.state)
        state = InstitutionalFlowState(
            id=state_id,
            lifecycle_state=lifecycle,
            participation=participation,
            initiative=initiative,
            responsive=responsive,
            absorption=absorption,
            exhaustion=exhaustion,
            inventory=inventory,
            campaign=campaign,
            pressure=pressure,
            persistence=persistence,
            cross_session=cross_session,
            confluences=confluences,
            explanation=explanation,
            version=version,
        )
        transitions: tuple[InstitutionalFlowTransition, ...] = ()
        if previous and previous.state.pressure.state != pressure.state:
            transitions = (
                InstitutionalFlowTransition(
                    id=stable_id("transition", previous.state.id, state.id),
                    previous_state=previous.state.pressure.state,
                    current_state=pressure.state,
                    available_at=boundary,
                    reason="time-valid directional evidence changed the inferred pressure state",
                ),
            )
        status = AnalysisStatus.INSUFFICIENT_EVIDENCE if len(evidence) < 3 else AnalysisStatus.DEGRADED if quality.level in {DataQualityLevel.LOW, DataQualityLevel.UNUSABLE} else AnalysisStatus.COMPLETE
        identifier = stable_id("snapshot", visible[-1].symbol, visible[-1].timeframe, boundary, mode, self.config.version, *(item.id for item in evidence))
        return InstitutionalFlowAnalysisSnapshot(
            id=identifier,
            symbol=visible[-1].symbol.replace("/", ""),
            timeframe=visible[-1].timeframe,
            session=context.session,
            analysis_timestamp=boundary,
            availability_timestamp=boundary,
            processing_mode=mode,
            status=status,
            state=state,
            evidence=bundle,
            quality=quality,
            transitions=transitions,
            configuration_version=self.config.version,
            engine_version=self.version,
            market_data_boundary=f"{visible[0].timestamp.isoformat()}..{visible[-1].timestamp.isoformat()}:{len(visible)}",
            upstream_versions=dict(context.upstream_versions),
            created_at=datetime.now(UTC),
        )

    def _weighted(self, item: InstitutionalFlowEvidence) -> float:
        source_weight = {
            EvidenceSourceEngine.MARKET_DATA: self.config.weights.market_data,
            EvidenceSourceEngine.SMC: self.config.weights.smc,
            EvidenceSourceEngine.LIQUIDITY: self.config.weights.liquidity,
            EvidenceSourceEngine.VOLUME_PROFILE: self.config.weights.volume_profile,
        }[item.source_engine]
        return item.strength * item.confidence * item.quality * source_weight

    def _quality(self, evidence: tuple[InstitutionalFlowEvidence, ...]) -> InstitutionalFlowQuality:
        if not evidence:
            return InstitutionalFlowQuality(level=DataQualityLevel.UNUSABLE, score=0, source_diversity=0, limitations=("no time-valid evidence",))
        score = fmean(item.quality for item in evidence)
        diversity = len({item.source_engine for item in evidence})
        score = clamp(score * (0.75 + 0.25 * min(diversity, 4) / 4))
        level = DataQualityLevel.HIGH if score >= 0.8 else DataQualityLevel.MEDIUM if score >= 0.55 else DataQualityLevel.LOW if score >= 0.2 else DataQualityLevel.UNUSABLE
        limits = () if diversity >= 3 else ("limited upstream source diversity",)
        return InstitutionalFlowQuality(level=level, score=score, source_diversity=diversity, limitations=limits)

    def _pressure(self, evidence: tuple[InstitutionalFlowEvidence, ...], quality: float) -> DirectionalFlowPressure:
        bullish = sum(self._weighted(item) for item in evidence if item.direction == FlowDirection.BULLISH)
        bearish = sum(self._weighted(item) for item in evidence if item.direction == FlowDirection.BEARISH)
        neutral = sum(self._weighted(item) for item in evidence if item.direction in {FlowDirection.NEUTRAL, FlowDirection.INDETERMINATE})
        total = bullish + bearish + neutral
        net = (bullish - bearish) / total if total else 0.0
        conflict = (2 * min(bullish, bearish) / (bullish + bearish)) if bullish + bearish else 0.0
        threshold = self.config.thresholds
        state = FlowState.STRONG_BULLISH if net >= threshold.strong_pressure else FlowState.MODERATE_BULLISH if net >= threshold.moderate_pressure else FlowState.STRONG_BEARISH if net <= -threshold.strong_pressure else FlowState.MODERATE_BEARISH if net <= -threshold.moderate_pressure else FlowState.INDETERMINATE if total == 0 else FlowState.BALANCED
        confidence = clamp(abs(net) * (1 - conflict) * quality + 0.15 * min(len({x.source_engine for x in evidence}), 3))
        return DirectionalFlowPressure(
            bullish_weight=bullish,
            bearish_weight=bearish,
            neutral_weight=neutral,
            net_pressure=net,
            state=state,
            evidence_diversity=len({item.source_engine for item in evidence}),
            conflict=conflict,
            confidence=confidence,
            quality=quality,
            persistence=clamp(len(evidence) / 12),
        )

    def _participation(self, evidence: tuple[InstitutionalFlowEvidence, ...], pressure: DirectionalFlowPressure, quality: float) -> ParticipationIntensity:
        relevant = tuple(item for item in evidence if item.evidence_type in {EvidenceType.VOLUME_EXPANSION, EvidenceType.RANGE_EXPANSION, EvidenceType.DISPLACEMENT, EvidenceType.STRUCTURAL_BREAK, EvidenceType.LIQUIDITY_EVENT, EvidenceType.PROFILE_MIGRATION})
        score = clamp(fmean([self._weighted(item) for item in relevant]) if relevant else 0)
        level = ParticipationLevel.HIGH if score >= self.config.thresholds.high_participation else ParticipationLevel.MODERATE if score >= self.config.thresholds.moderate_participation else ParticipationLevel.LOW if relevant else ParticipationLevel.INDETERMINATE
        direction = FlowDirection.BULLISH if pressure.net_pressure > 0.1 else FlowDirection.BEARISH if pressure.net_pressure < -0.1 else FlowDirection.INDETERMINATE
        return ParticipationIntensity(score=score, level=level, direction=direction, persistence=clamp(len(relevant) / 6), confidence=clamp(score * quality * (1 - pressure.conflict)), quality=quality, evidence_ids=tuple(item.id for item in relevant), ambiguity=pressure.conflict)

    def _activities(self, evidence: tuple[InstitutionalFlowEvidence, ...], pressure: DirectionalFlowPressure, boundary: datetime) -> tuple[InitiativeActivity | None, ResponsiveActivity | None, ActivityType]:
        initiative_items = tuple(item for item in evidence if item.evidence_type in {EvidenceType.DISPLACEMENT, EvidenceType.STRUCTURAL_BREAK, EvidenceType.RANGE_EXPANSION, EvidenceType.PROFILE_MIGRATION})
        responsive_items = tuple(item for item in evidence if item.evidence_type in {EvidenceType.VALUE_REJECTION, EvidenceType.NODE_INTERACTION, EvidenceType.LIQUIDITY_EVENT, EvidenceType.REPEATED_TEST})
        initiative_score = clamp(fmean([self._weighted(item) for item in initiative_items]) if initiative_items else 0)
        responsive_score = clamp(fmean([self._weighted(item) for item in responsive_items]) if responsive_items else 0)
        direction = FlowDirection.BULLISH if pressure.net_pressure > 0 else FlowDirection.BEARISH if pressure.net_pressure < 0 else FlowDirection.INDETERMINATE
        initiative = InitiativeActivity(direction=direction, initiation_timestamp=min((item.availability_timestamp for item in initiative_items), default=boundary), strength=initiative_score, continuation=clamp(len(initiative_items) / 4), structural_consequence=clamp(sum(item.evidence_type == EvidenceType.STRUCTURAL_BREAK for item in initiative_items) / 2), confidence=initiative_score * (1 - pressure.conflict), evidence_ids=tuple(item.id for item in initiative_items)) if initiative_score >= self.config.thresholds.initiative else None
        responsive = ResponsiveActivity(defended_reference=responsive_items[0].source_object_id if responsive_items else None, direction=direction, strength=responsive_score, persistence=clamp(len(responsive_items) / 4), confidence=responsive_score * (1 - pressure.conflict), evidence_ids=tuple(item.id for item in responsive_items)) if responsive_score >= self.config.thresholds.responsive else None
        activity = ActivityType.MIXED if initiative and responsive else ActivityType.INITIATIVE if initiative else ActivityType.RESPONSIVE if responsive else ActivityType.NONE
        return initiative, responsive, activity

    def _absorption(self, evidence: tuple[InstitutionalFlowEvidence, ...], pressure: DirectionalFlowPressure) -> AbsorptionInference | None:
        items = tuple(item for item in evidence if item.evidence_type in {EvidenceType.LIMITED_PROGRESS, EvidenceType.REPEATED_TEST, EvidenceType.VALUE_REJECTION})
        score = clamp(fmean([self._weighted(item) for item in items]) if items else 0)
        if score < self.config.thresholds.absorption:
            return None
        absorbed = FlowDirection.BEARISH if pressure.net_pressure >= 0 else FlowDirection.BULLISH
        defending = FlowDirection.BULLISH if absorbed == FlowDirection.BEARISH else FlowDirection.BEARISH
        kind = AbsorptionType.BULLISH if defending == FlowDirection.BULLISH else AbsorptionType.BEARISH
        return AbsorptionInference(absorption_type=kind, absorbed_pressure=absorbed, defending_side=defending, test_count=sum(item.evidence_type == EvidenceType.REPEATED_TEST for item in items), estimated_intensity=score, efficiency_change=-score, confidence=score * (1 - pressure.conflict), ambiguity=pressure.conflict, evidence_ids=tuple(item.id for item in items))

    def _exhaustion(self, evidence: tuple[InstitutionalFlowEvidence, ...], pressure: DirectionalFlowPressure, boundary: datetime) -> ExhaustionInference | None:
        items = tuple(item for item in evidence if item.evidence_type in {EvidenceType.EFFICIENCY_DECLINE, EvidenceType.STRUCTURAL_FAILURE, EvidenceType.LIMITED_PROGRESS})
        score = clamp(fmean([self._weighted(item) for item in items]) if items else 0)
        if score < self.config.thresholds.exhaustion:
            return None
        direction = FlowDirection.BULLISH if pressure.net_pressure >= 0 else FlowDirection.BEARISH
        kind = ExhaustionType.BULLISH if direction == FlowDirection.BULLISH else ExhaustionType.BEARISH
        return ExhaustionInference(exhaustion_type=kind, exhausted_direction=direction, onset=min((item.availability_timestamp for item in items), default=boundary), strength=score, persistence=clamp(len(items) / 4), reversal_evidence=clamp(sum(item.role == EvidenceRole.CONTRADICTING for item in items) / 3), ambiguity=pressure.conflict, confidence=score * (1 - pressure.conflict), evidence_ids=tuple(item.id for item in items))

    def _inventory(self, evidence: tuple[InstitutionalFlowEvidence, ...], pressure: DirectionalFlowPressure, absorption: AbsorptionInference | None) -> AccumulationDistributionInference:
        families = {item.source_engine for item in evidence}
        acceptance = sum(self._weighted(item) for item in evidence if item.evidence_type in {EvidenceType.VALUE_ACCEPTANCE, EvidenceType.REPEATED_TEST, EvidenceType.LIQUIDITY_EVENT})
        score = clamp((acceptance + (absorption.estimated_intensity if absorption else 0)) / 3)
        if len(families) < 2 or score < self.config.thresholds.inventory:
            behavior = InventoryBehaviorType.INSUFFICIENT if len(evidence) < 4 else InventoryBehaviorType.BALANCE
        elif pressure.conflict >= self.config.thresholds.conflict:
            behavior = InventoryBehaviorType.AMBIGUOUS
        elif pressure.net_pressure > 0:
            behavior = InventoryBehaviorType.ACCUMULATION
        else:
            behavior = InventoryBehaviorType.DISTRIBUTION
        direction = FlowDirection.BULLISH if behavior in {InventoryBehaviorType.ACCUMULATION, InventoryBehaviorType.REACCUMULATION} else FlowDirection.BEARISH if behavior in {InventoryBehaviorType.DISTRIBUTION, InventoryBehaviorType.REDISTRIBUTION} else FlowDirection.INDETERMINATE
        return AccumulationDistributionInference(behavior=behavior, direction=direction, strength=score, confidence=score * (1 - pressure.conflict), ambiguity=pressure.conflict, evidence_family_count=len(families), alternative_interpretation="ordinary balance" if behavior not in {InventoryBehaviorType.BALANCE, InventoryBehaviorType.INSUFFICIENT} else None, evidence_ids=tuple(item.id for item in evidence if item.evidence_type in {EvidenceType.VALUE_ACCEPTANCE, EvidenceType.REPEATED_TEST, EvidenceType.LIQUIDITY_EVENT, EvidenceType.LIMITED_PROGRESS}))

    def _campaign(self, inventory: AccumulationDistributionInference, initiative: InitiativeActivity | None, exhaustion: ExhaustionInference | None, pressure: DirectionalFlowPressure, previous: InstitutionalFlowAnalysisSnapshot | None) -> CampaignPhaseInference:
        behavior = inventory.behavior
        if behavior == InventoryBehaviorType.ACCUMULATION:
            phase = CampaignPhase.REACCUMULATION if previous and previous.state.campaign.phase == CampaignPhase.MARKUP else CampaignPhase.ACCUMULATION
        elif behavior == InventoryBehaviorType.DISTRIBUTION:
            phase = CampaignPhase.REDISTRIBUTION if previous and previous.state.campaign.phase == CampaignPhase.MARKDOWN else CampaignPhase.DISTRIBUTION
        elif initiative and pressure.net_pressure > 0.2:
            phase = CampaignPhase.MARKUP
        elif initiative and pressure.net_pressure < -0.2:
            phase = CampaignPhase.MARKDOWN
        elif exhaustion:
            phase = CampaignPhase.TRANSITION
        elif behavior == InventoryBehaviorType.AMBIGUOUS:
            phase = CampaignPhase.AMBIGUOUS
        elif behavior == InventoryBehaviorType.INSUFFICIENT:
            phase = CampaignPhase.INSUFFICIENT
        else:
            phase = CampaignPhase.PREPARATION
        ids = inventory.evidence_ids + (initiative.evidence_ids if initiative else ()) + (exhaustion.evidence_ids if exhaustion else ())
        return CampaignPhaseInference(phase=phase, previous_phase=previous.state.campaign.phase if previous else None, confidence=clamp(max(inventory.confidence, initiative.confidence if initiative else 0) * (1 - pressure.conflict)), ambiguity=pressure.conflict, explanation=f"Approximate evidence-based {phase.value}; not verified institutional activity.", evidence_ids=tuple(dict.fromkeys(ids)))

    def _persistence(self, evidence: tuple[InstitutionalFlowEvidence, ...], pressure: DirectionalFlowPressure, previous: InstitutionalFlowAnalysisSnapshot | None) -> FlowPersistence:
        score = clamp(len(evidence) / 12 * (1 - pressure.conflict))
        direction_changes = 0
        if previous and previous.state.pressure.net_pressure * pressure.net_pressure < 0:
            direction_changes = previous.state.persistence.direction_changes + 1
            state = FlowPersistenceState.REVERSING
        elif score >= 0.7:
            state = FlowPersistenceState.PERSISTENT
        elif previous and score > previous.state.persistence.score:
            state = FlowPersistenceState.STRENGTHENING
        elif previous and score < previous.state.persistence.score:
            state = FlowPersistenceState.WEAKENING
        elif score >= 0.35:
            state = FlowPersistenceState.DEVELOPING
        else:
            state = FlowPersistenceState.TRANSIENT
        return FlowPersistence(state=state, score=score, window_observations=len(evidence), decay_factor=self.config.evidence.decay_per_candle, direction_changes=direction_changes)

    @staticmethod
    def _cross_session(evidence: tuple[InstitutionalFlowEvidence, ...]) -> tuple[CrossSessionFlow, ...]:
        sessions: dict[SessionType, list[InstitutionalFlowEvidence]] = defaultdict(list)
        for item in evidence:
            if item.session != SessionType.UNKNOWN:
                sessions[item.session].append(item)
        ordered = sorted(sessions.items(), key=lambda pair: min(item.availability_timestamp for item in pair[1]))
        result = []
        for (previous_name, previous), (current_name, current) in zip(ordered, ordered[1:], strict=False):
            previous_net = sum(item.strength if item.direction == FlowDirection.BULLISH else -item.strength if item.direction == FlowDirection.BEARISH else 0 for item in previous)
            current_net = sum(item.strength if item.direction == FlowDirection.BULLISH else -item.strength if item.direction == FlowDirection.BEARISH else 0 for item in current)
            relationship = "continuation" if previous_net * current_net > 0 else "reversal" if previous_net * current_net < 0 else "handoff"
            direction = FlowDirection.BULLISH if current_net > 0 else FlowDirection.BEARISH if current_net < 0 else FlowDirection.INDETERMINATE
            result.append(CrossSessionFlow(previous_session=previous_name, current_session=current_name, relationship=relationship, direction=direction, strength=clamp(abs(current_net) / max(len(current), 1)), confidence=clamp((abs(previous_net) + abs(current_net)) / max(len(previous) + len(current), 1)), completed=True, evidence_ids=tuple(item.id for item in (*previous, *current))))
        return tuple(result)

    def _confluences(self, evidence: tuple[InstitutionalFlowEvidence, ...]) -> tuple[InstitutionalFlowConfluence, ...]:
        result = []
        for direction in (FlowDirection.BULLISH, FlowDirection.BEARISH):
            values = tuple(item for item in evidence if item.direction == direction)
            sources = tuple(dict.fromkeys(item.source_engine for item in values))
            if len(sources) < 2:
                continue
            raw = sum(self._weighted(item) for item in values)
            groups = len({item.correlation_group for item in values})
            discount = clamp(groups / len(values))
            result.append(InstitutionalFlowConfluence(id=stable_id("confluence", direction, *(item.id for item in values)), source_evidence_ids=tuple(item.id for item in values), source_engines=sources, direction=direction, raw_score=raw, correlation_discount=discount, adjusted_score=clamp(raw * discount / max(len(sources), 1)), confidence=clamp((len(sources) / 4) * discount)))
        return tuple(result)
