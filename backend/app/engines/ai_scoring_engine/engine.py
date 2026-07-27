from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import datetime, timedelta
from hashlib import sha256
import itertools
import logging
import math

from .clock import Clock, SystemClock
from backend.app.ai.provider_client import AIProviderClient
from backend.app.ai.prompts.loader import PromptLoader
from .config import AIScoringConfig
from .models import (
    AIScoreSnapshot,
    DirectionalLabel,
    EvidenceConflict,
    ExplanationContributor,
    FreshnessState,
    ScoreComponent,
    ScoreExplanation,
    ScoreMetadata,
    ScoreMode,
    ScoreStatus,
    ScoringInput,
    ScoringContext,
    SignalScore,
    SourceEvidence,
    stable_id,
)

logger = logging.getLogger(__name__)


class DeterministicAIScoringEngine:
    """Pure, versioned weighted-intelligence policy with no network or clock side effects."""

    name = "ai_scoring"

    def __init__(self, config: AIScoringConfig | None = None, clock: Clock | None = None) -> None:
        self.config = config or AIScoringConfig()
        self.clock = clock or SystemClock()
        self.configuration_hash = sha256(self.config.model_dump_json().encode()).hexdigest()

    def score(self, scoring_input: ScoringInput) -> AIScoreSnapshot:
        ordered = tuple(sorted(scoring_input.sources(), key=lambda item: item.source))
        by_source = {item.source: item for item in ordered}
        input_fingerprint = scoring_input.fingerprint(self.config.policy_name, self.config.policy_version, self.configuration_hash)
        raw: list[tuple[SourceEvidence, float, FreshnessState]] = []
        for evidence in ordered:
            component_config = self.config.components[evidence.source]
            factor, state = self._freshness(evidence, scoring_input.as_of, component_config.fresh_seconds, component_config.stale_seconds)
            raw.append((evidence, factor, state))

        directional_denominator = math.fsum(
            self.config.components[item.source].directional_weight * item.quality * freshness
            for item, freshness, _ in raw
            if self.config.components[item.source].directional_weight > 0
        )
        risk_denominator = math.fsum(self.config.components[item.source].risk_weight for item, _, _ in raw)
        components = tuple(
            self._component(item, freshness, state, directional_denominator, risk_denominator)
            for item, freshness, state in raw
        )
        directional = self._bounded(math.fsum(item.directional_contribution for item in components), -100, 100)
        conflicts = self._conflicts(ordered)
        alignment = self._alignment(ordered)
        missing = tuple(name for name in sorted(self.config.components) if name not in by_source)
        degraded = tuple(sorted(item.source for item in ordered if item.degraded))
        freshness_average = self._average(item.freshness_factor for item in components)
        quality_average = math.fsum(item.quality_factor for item in components) / len(self.config.components)
        availability = len(components) / len(self.config.components)
        groups = {item.source_group for item in ordered if abs(item.direction) >= 0.05 and self.config.components[item.source].directional_weight > 0}
        independence = min(1.0, len(groups) / self.config.evidence.minimum_independent_groups)
        conflict_penalty = min(self.config.conflicts.maximum_total_penalty, math.fsum(item.confidence_penalty for item in conflicts))
        confidence = 100 * availability * freshness_average * quality_average * (0.5 + alignment / 200) * independence
        confidence -= conflict_penalty
        if len(ordered) <= 1:
            confidence = min(confidence, self.config.evidence.single_source_confidence_ceiling)
        if degraded:
            confidence = min(confidence, self.config.evidence.degraded_confidence_ceiling)
        if any(item.freshness_state in {FreshnessState.STALE, FreshnessState.EXPIRED} for item in components):
            confidence = min(confidence, self.config.evidence.stale_confidence_ceiling)
        confidence = self._bounded(confidence, 0, 100)
        quality = self._bounded(100 * quality_average * freshness_average, 0, 100)
        base_risk = math.fsum(item.risk_contribution for item in components)
        stale_risk = 10 * sum(item.freshness_state in {FreshnessState.STALE, FreshnessState.EXPIRED} for item in components)
        risk = self._bounded(base_risk + stale_risk + conflict_penalty * 0.5, 0, 100)
        composite = self._bounded(directional * (confidence / 100) * (1 - 0.25 * risk / 100), -100, 100)
        status = self._status(scoring_input.mode, components, missing, degraded, groups)
        explanation = self._explanation(directional, confidence, risk, components, conflicts, missing, degraded)
        precision = self.config.output_precision
        snapshot_id = stable_id("ai-score", input_fingerprint, scoring_input.mode)
        return AIScoreSnapshot(
            snapshot_id=snapshot_id,
            instrument=scoring_input.instrument,
            timeframe=scoring_input.timeframe,
            as_of=scoring_input.as_of,
            calculated_at=self.clock.now(),
            mode=scoring_input.mode,
            policy_name=self.config.policy_name,
            policy_version=self.config.policy_version,
            directional_score=round(directional, precision),
            directional_label=self.label(directional),
            confidence_score=round(confidence, precision),
            market_risk_score=round(risk, precision),
            evidence_alignment_score=round(alignment, precision),
            data_quality_score=round(quality, precision),
            composite_score=round(composite, precision),
            components=components,
            conflicts=conflicts,
            missing_sources=missing,
            degraded_sources=degraded,
            explanation=explanation,
            status=status,
            metadata=ScoreMetadata(
                engine_version=self.config.engine_version,
                schema_version=self.config.schema_version,
                configuration_version=self.config.configuration_version,
                configuration_hash=self.configuration_hash,
                input_fingerprint=input_fingerprint,
                independent_group_count=len(groups),
                available_source_count=len(components),
            ),
        )

    def label(self, score: float) -> DirectionalLabel:
        value = self.config.labels
        if score < value.strong_bearish:
            return DirectionalLabel.STRONG_BEARISH
        if score < value.bearish:
            return DirectionalLabel.BEARISH
        if score < value.slightly_bearish:
            return DirectionalLabel.SLIGHTLY_BEARISH
        if score <= value.slightly_bullish:
            return DirectionalLabel.NEUTRAL
        if score <= value.bullish:
            return DirectionalLabel.SLIGHTLY_BULLISH
        if score <= value.strong_bullish:
            return DirectionalLabel.BULLISH
        return DirectionalLabel.STRONG_BULLISH

    def _freshness(self, evidence: SourceEvidence, as_of: datetime, fresh_seconds: int, stale_seconds: int) -> tuple[float, FreshnessState]:
        age = scoring_age = as_of - evidence.publication_timestamp
        if scoring_age < -timedelta(seconds=self.config.future_clock_skew_seconds):
            raise ValueError(f"future source timestamp: {evidence.source}")
        seconds = max(0.0, age.total_seconds())
        if seconds <= fresh_seconds:
            return 1.0, FreshnessState.FRESH
        if seconds <= stale_seconds:
            span = stale_seconds - fresh_seconds
            return 1 - 0.5 * ((seconds - fresh_seconds) / span), FreshnessState.AGING
        if seconds <= stale_seconds * 2:
            return 0.25, FreshnessState.STALE
        return 0.0, FreshnessState.EXPIRED

    def _component(self, evidence: SourceEvidence, freshness: float, state: FreshnessState, directional_denominator: float, risk_denominator: float) -> ScoreComponent:
        cfg = self.config.components[evidence.source]
        numerator = cfg.directional_weight * evidence.quality * freshness
        effective_weight = numerator / directional_denominator if directional_denominator else 0.0
        risk_weight = cfg.risk_weight / risk_denominator if risk_denominator else 0.0
        precision = self.config.output_precision
        return ScoreComponent(
            component_name=evidence.source,
            source_engine=evidence.source,
            source_group=cfg.source_group,
            evidence_id=evidence.evidence_id,
            normalized_direction=evidence.direction,
            directional_contribution=round(100 * evidence.direction * effective_weight, precision),
            confidence_contribution=round(100 * cfg.confidence_weight * evidence.confidence * evidence.quality * freshness, precision),
            risk_contribution=round(100 * evidence.risk * risk_weight, precision),
            quality_contribution=round(100 * evidence.quality * freshness / len(self.config.components), precision),
            configured_weight=cfg.directional_weight,
            effective_weight=round(effective_weight, precision),
            freshness_factor=round(freshness, precision),
            availability_factor=1,
            quality_factor=evidence.quality,
            freshness_state=state,
            reason_codes=tuple(sorted(set(evidence.reason_codes))),
            source_timestamp=evidence.source_timestamp,
            observation_timestamp=evidence.observation_timestamp,
        )

    def _conflicts(self, evidence: tuple[SourceEvidence, ...]) -> tuple[EvidenceConflict, ...]:
        result = []
        for left, right in itertools.combinations(evidence, 2):
            gap = abs(left.direction - right.direction)
            if gap < self.config.conflicts.directional_gap_threshold:
                continue
            severe = gap >= self.config.conflicts.severe_gap_threshold
            severity = "severe" if severe else "moderate"
            penalty = self.config.conflicts.severe_penalty if severe else self.config.conflicts.moderate_penalty
            conflict_type = "cross_group_directional" if left.source_group != right.source_group else "same_group_directional"
            sources = tuple(sorted((left.source, right.source)))
            result.append(
                EvidenceConflict(
                    conflict_id=stable_id("conflict", *sources, round(gap, 6)),
                    conflict_type=conflict_type,
                    severity=severity,
                    sources=sources,
                    directional_gap=round(gap, self.config.output_precision),
                    description_code=f"{conflict_type}_{severity}",
                    confidence_penalty=penalty,
                )
            )
        return tuple(sorted(result, key=lambda item: (-item.confidence_penalty, item.sources)))

    def _alignment(self, evidence: tuple[SourceEvidence, ...]) -> float:
        directional = tuple(item for item in evidence if self.config.components[item.source].directional_weight > 0 and abs(item.direction) >= 0.05)
        if len(directional) < 2:
            return 50.0 if directional else 0.0
        total = math.fsum(self.config.components[item.source].directional_weight * abs(item.direction) for item in directional)
        signed = math.fsum(self.config.components[item.source].directional_weight * item.direction for item in directional)
        return self._bounded(100 * abs(signed) / total if total else 0, 0, 100)

    def _status(self, mode: ScoreMode, components: tuple[ScoreComponent, ...], missing: tuple[str, ...], degraded: tuple[str, ...], groups: set[str]) -> ScoreStatus:
        directional_count = sum(item.configured_weight > 0 and item.freshness_state != FreshnessState.EXPIRED for item in components)
        if directional_count < self.config.evidence.minimum_components or len(groups) < self.config.evidence.minimum_independent_groups:
            return ScoreStatus.INSUFFICIENT_EVIDENCE
        if any(item.freshness_state in {FreshnessState.STALE, FreshnessState.EXPIRED} for item in components):
            return ScoreStatus.STALE
        if mode == ScoreMode.REPLAY:
            return ScoreStatus.REPLAY
        if degraded or missing:
            return ScoreStatus.DEGRADED
        return ScoreStatus.READY

    def _explanation(self, directional: float, confidence: float, risk: float, components: tuple[ScoreComponent, ...], conflicts: tuple[EvidenceConflict, ...], missing: tuple[str, ...], degraded: tuple[str, ...]) -> ScoreExplanation:
        contributors = [
            ExplanationContributor(source=item.source_engine, reason_code=(item.reason_codes[0] if item.reason_codes else "normalized_evidence"), contribution=item.directional_contribution)
            for item in components
            if item.directional_contribution
        ]
        positive = tuple(sorted((item for item in contributors if item.contribution > 0), key=lambda item: (-item.contribution, item.source)))
        negative = tuple(sorted((item for item in contributors if item.contribution < 0), key=lambda item: (item.contribution, item.source)))
        risk_items = tuple(
            sorted(
                (ExplanationContributor(source=item.source_engine, reason_code="contextual_risk", contribution=item.risk_contribution) for item in components if item.risk_contribution > 0),
                key=lambda item: (-item.contribution, item.source),
            )
        )
        limitations = tuple(sorted((*[f"{name}_missing" for name in missing], *[f"{name}_degraded" for name in degraded], *[item.description_code for item in conflicts])))
        return ScoreExplanation(
            summary_code=f"{self.label(directional).value}_{'low' if confidence < 40 else 'moderate' if confidence < 70 else 'high'}_confidence_{'high' if risk >= 70 else 'normal'}_risk",
            positive_contributors=positive,
            negative_contributors=negative,
            risk_contributors=risk_items,
            limitations=limitations,
        )

    @staticmethod
    def _bounded(value: float, lower: float, upper: float) -> float:
        if not math.isfinite(value):
            raise ValueError("non-finite score")
        value = min(upper, max(lower, value))
        return 0.0 if value == 0 else value

    @staticmethod
    def _average(values: Iterable[float]) -> float:
        items: tuple[float, ...] = tuple(values)
        return math.fsum(items) / len(items) if items else 0.0


class AIScoringEngine(ABC):
    """Deprecated async pipeline port retained for legacy Signal Engine compatibility."""

    @abstractmethod
    async def score(self, context: ScoringContext) -> SignalScore: ...


class ProviderScoringEngine(AIScoringEngine):
    """Legacy opt-in adapter; Production 1.0 registration does not construct it."""

    def __init__(self, client: AIProviderClient, prompts: PromptLoader) -> None:
        self.client = client
        self.prompts = prompts

    async def score(self, context: ScoringContext) -> SignalScore:
        model = "gpt-oss-120b"
        prompt_version = "signal_analysis_v1"
        now = datetime.now().astimezone()
        bucket = now.replace(minute=(now.minute // 10) * 10, second=0, microsecond=0)
        idempotency_key = sha256(context.model_dump_json().encode()).hexdigest()
        call_context = {
            "trigger": "legacy_provider_scoring_adapter",
            "instrument": "unknown",
            "idempotency_key": idempotency_key,
            "request_boundary": bucket.isoformat(),
        }
        logger.info("legacy_ai_scoring.provider_call.started", extra=call_context)
        try:
            completion = await self.client.complete_json(
                system_prompt=self.prompts.load(prompt_version),
                payload=context.model_dump(mode="json"),
                model=model,
                temperature=0.1,
                max_tokens=900,
                request_id=idempotency_key,
            )
        except Exception as exc:
            logger.info(
                "legacy_ai_scoring.provider_call.completed",
                extra={
                    **call_context,
                    "result": "failed",
                    "exception_class": type(exc).__name__,
                },
            )
            raise
        logger.info(
            "legacy_ai_scoring.provider_call.completed",
            extra={**call_context, "result": "success"},
        )
        response = completion.content
        response.setdefault("model", model)
        response.setdefault("prompt_version", prompt_version)
        return SignalScore.model_validate(response)
