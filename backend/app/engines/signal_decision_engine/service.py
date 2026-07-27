from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import logging
from time import perf_counter
from typing import Any
from uuid import UUID

from backend.app.engines.ai_scoring_engine import AIScoringService
from backend.app.engines.market_data_engine import Timeframe
from backend.app.events import Event, EventBus
from backend.app.features import FeatureRecord, FeatureStore

from .clock import Clock, SystemClock
from .config import SignalDecisionConfig
from .engine import ConservativeSignalDecisionPolicy
from .events import SignalDecisionBlocked, SignalDecisionEligible, SignalDecisionInsufficientEvidence, SignalDecisionInvalid, SignalDecisionObserveOnly
from .exceptions import SignalDecisionInputError, SignalDecisionSnapshotNotFound
from .models import (
    DecisionDirection,
    DecisionHistory,
    DecisionLifecycleStatus,
    DecisionMode,
    DecisionRequest,
    DecisionState,
    DependencyCriticality,
    DependencyHealth,
    DependencyState,
    EconomicRiskReference,
    MarketRegimeReference,
    SignalDecision,
    SignalDecisionInput,
    stable_id,
)
from .policies import DecisionPolicyRegistry
from .repository import SignalDecisionRepository

logger = logging.getLogger(__name__)


@dataclass
class SignalDecisionMetrics:
    requests_total: int = 0
    completed_total: int = 0
    failed_total: int = 0
    state_totals: dict[str, int] = field(default_factory=dict)
    duplicates_total: int = 0
    persistence_failures_total: int = 0
    feature_store_failures_total: int = 0
    event_publish_failures_total: int = 0
    expiration_runs_total: int = 0
    durations_ms: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    latest_decision_id: str | None = None

    def snapshot(self) -> dict[str, object]:
        return {
            "requests_total": self.requests_total,
            "completed_total": self.completed_total,
            "failed_total": self.failed_total,
            "state_totals": dict(sorted(self.state_totals.items())),
            "duplicates_total": self.duplicates_total,
            "persistence_failures_total": self.persistence_failures_total,
            "feature_store_failures_total": self.feature_store_failures_total,
            "event_publish_failures_total": self.event_publish_failures_total,
            "expiration_runs_total": self.expiration_runs_total,
            "average_duration_ms": sum(self.durations_ms) / len(self.durations_ms) if self.durations_ms else 0,
            "maximum_duration_ms": max(self.durations_ms, default=0),
            "latest_decision_id": self.latest_decision_id,
        }


class SignalDecisionService:
    def __init__(
        self,
        repository: SignalDecisionRepository,
        ai_scoring: AIScoringService,
        event_bus: EventBus,
        feature_store: FeatureStore,
        config: SignalDecisionConfig,
        *,
        economic_calendar: Any = None,
        market_regime: Any = None,
        repository_mode: str = "memory",
        clock: Clock | None = None,
    ) -> None:
        self.repository = repository
        self.ai_scoring = ai_scoring
        self.event_bus = event_bus
        self.feature_store = feature_store
        self.economic_calendar = economic_calendar
        self.market_regime = market_regime
        self.config = config
        self.repository_mode = repository_mode
        self.clock = clock or SystemClock()
        self.policy = ConservativeSignalDecisionPolicy(config)
        self.policies = DecisionPolicyRegistry()
        self.policies.register(self.policy)
        self.metrics = SignalDecisionMetrics()
        self.initialized = False
        self.closed = False

    async def start(self) -> None:
        self.policies.get(self.config.policy_name, self.config.policy_version)
        self.initialized = True
        self.closed = False
        logger.info("Signal Decision Engine initialized", extra={"engine": "signal_decision", "policy_version": self.config.policy_version})

    async def stop(self) -> None:
        self.closed = True
        logger.info("Signal Decision Engine stopped", extra={"engine": "signal_decision"})

    async def evaluate(self, request: DecisionRequest) -> SignalDecision:
        started = perf_counter()
        self.metrics.requests_total += 1
        try:
            decision_input = await self.collect_input(request)
            fingerprint = decision_input.fingerprint(self.policy.configuration_hash)
            existing = await self.repository.find_by_fingerprint(fingerprint, request.mode)
            if existing:
                self.metrics.duplicates_total += 1
                return existing
            candidate = self.policies.get(decision_input.policy_name, decision_input.policy_version).evaluate(decision_input)
            try:
                decision = await self.repository.save_decision(candidate) if request.persist else candidate
            except Exception:
                self.metrics.persistence_failures_total += 1
                raise
            if request.persist:
                await self._publish(decision, request.publish_events)
            self.metrics.completed_total += 1
            self.metrics.state_totals[decision.state.value] = self.metrics.state_totals.get(decision.state.value, 0) + 1
            self.metrics.latest_decision_id = str(decision.decision_id)
            logger.info(
                "Signal decision completed",
                extra={
                    "engine": "signal_decision",
                    "instrument": decision.instrument,
                    "timeframe": decision.timeframe,
                    "direction": decision.direction.value,
                    "state": decision.state.value,
                    "policy_version": decision.decision_policy_version,
                    "rule_count": len(decision.rules),
                    "blocker_count": len(decision.blockers),
                    "warning_count": len(decision.warnings),
                    "mode": decision.mode.value,
                },
            )
            return decision
        except Exception:
            self.metrics.failed_total += 1
            logger.exception("Signal decision failed", extra={"engine": "signal_decision", "instrument": request.instrument, "timeframe": request.timeframe})
            raise
        finally:
            self.metrics.durations_ms.append((perf_counter() - started) * 1000)

    async def collect_input(self, request: DecisionRequest) -> SignalDecisionInput:
        now = self.clock.now()
        boundary = request.as_of or now
        if boundary.tzinfo is None:
            raise SignalDecisionInputError("as_of must be timezone-aware")
        boundary = boundary.astimezone(UTC)
        if request.mode == DecisionMode.LIVE and boundary > now + timedelta(seconds=self.config.future_clock_skew_seconds):
            raise SignalDecisionInputError("future decision boundary is prohibited")
        policy_name = request.decision_policy_name or self.config.policy_name
        policy_version = request.decision_policy_version or self.config.policy_version
        self.policies.get(policy_name, policy_version)
        score = await self.ai_scoring.repository.get_snapshot(request.ai_score_snapshot_id)
        if score is None:
            raise SignalDecisionSnapshotNotFound("AI Scoring snapshot not found")
        recent_all = await self.repository.find_recent_decisions(request.instrument, request.timeframe, boundary, 50)
        recent = tuple(item for item in recent_all if item.mode.value == request.mode.value)
        direction = DecisionDirection(self.config.direction_mapping[score.directional_label.value])
        latest = recent[0] if recent else None
        active = next((item for item in recent if item.valid_from <= boundary < item.valid_until), None)
        same = next((item for item in recent if item.direction == direction), None)
        opposite = next((item for item in recent if item.state == DecisionState.ELIGIBLE and item.direction not in {direction, DecisionDirection.NEUTRAL} and item.valid_until > boundary), None)
        economic = await self._economic_context(request.instrument, boundary)
        regime = await self._regime_context(request.instrument, request.timeframe, boundary)
        dependencies = self._dependency_health(boundary)
        return SignalDecisionInput(
            instrument=request.instrument,
            timeframe=request.timeframe,
            as_of=boundary,
            requested_at=now,
            ai_score=score,
            economic_risk=economic,
            market_regime=regime,
            dependency_health=dependencies,
            history=DecisionHistory(
                latest=latest.history_reference() if latest else None,
                active=active.history_reference() if active else None,
                recent_same_direction=same.history_reference() if same else None,
                recent_opposite_eligible=opposite.history_reference() if opposite else None,
            ),
            current_ai_analysis=getattr(request, "current_ai_analysis", None),
            temporal_context=getattr(request, "temporal_context", None),
            temporal_metrics=getattr(request, "temporal_metrics", None),
            market_snapshot_id=getattr(request, "market_snapshot_id", None),
            quantitative_forecast_id=getattr(request, "quantitative_forecast_id", None),
            current_price=getattr(request, "current_price", None),
            expected_move=getattr(request, "expected_move", None),
            mode=request.mode,
            policy_name=policy_name,
            policy_version=policy_version,
        )

    async def _economic_context(self, instrument: str, boundary: datetime) -> EconomicRiskReference | None:
        if self.economic_calendar is None:
            return None
        try:
            context = await self.economic_calendar.context(instrument, as_of=boundary, publish=False)
            event_ids = tuple(item.event_id for item in context.active_relevant_events)
            if context.next_relevant_event:
                event_ids += (context.next_relevant_event.event_id,)
            return EconomicRiskReference(
                context_id=context.context_id,
                phase=context.risk_window_phase.value,
                risk_score=context.risk_score * 100,
                as_of=context.analysis_timestamp,
                degraded=bool(context.unavailable_context),
                context_state=context.context_state.value,
                event_ids=event_ids,
            )
        except Exception:
            return None

    async def _regime_context(self, instrument: str, timeframe: str, boundary: datetime) -> MarketRegimeReference | None:
        if self.market_regime is None:
            return None
        try:
            snapshot = await self.market_regime.state(instrument, Timeframe(timeframe), at=boundary)
            if snapshot is None:
                return None
            return MarketRegimeReference(
                snapshot_id=snapshot.snapshot_id,
                regime=snapshot.dominant_regime.value,
                as_of=snapshot.analysis_timestamp,
                degraded=snapshot.degradation.is_degraded,
            )
        except Exception:
            return None

    def _dependency_health(self, at: datetime) -> tuple[DependencyHealth, ...]:
        persistence = DependencyState.AVAILABLE if self.repository_mode != "memory" else DependencyState.UNAVAILABLE
        ai_status = self.ai_scoring.health()["status"]
        ai_state = DependencyState.AVAILABLE if ai_status == "healthy" else DependencyState.DEGRADED if ai_status == "degraded" else DependencyState.UNAVAILABLE
        return (
            DependencyHealth(name="ai_scoring", state=ai_state, criticality=DependencyCriticality.CRITICAL, checked_at=at),
            DependencyHealth(name="persistence", state=persistence, criticality=DependencyCriticality.CRITICAL if self.config.persistence_required_in_production else DependencyCriticality.OPTIONAL, checked_at=at),
            DependencyHealth(name="economic_calendar", state=DependencyState.AVAILABLE if self.economic_calendar else DependencyState.UNAVAILABLE, criticality=DependencyCriticality.REQUIRED_FOR_ELIGIBILITY, checked_at=at),
            DependencyHealth(name="market_regime", state=DependencyState.AVAILABLE if self.market_regime else DependencyState.UNAVAILABLE, criticality=DependencyCriticality.OPTIONAL, checked_at=at),
        )

    async def _publish(self, decision: SignalDecision, publish_events: bool) -> None:
        correlation_id = stable_id("decision-correlation", decision.input_fingerprint)
        if decision.mode == DecisionMode.LIVE or self.config.publish_replay_features:
            try:
                await self.feature_store.write(
                    FeatureRecord(
                        feature_id=stable_id("decision-feature", decision.decision_id),
                        correlation_id=correlation_id,
                        namespace="signal_decision",
                        engine_name="signal_decision",
                        engine_version=self.config.engine_version,
                        compatibility_version=self.config.schema_version,
                        values={
                            "state": decision.state.value,
                            "direction": decision.direction.value,
                            "eligibility_score": decision.eligibility_score,
                            "valid_until": decision.valid_until.isoformat(),
                            "decision_id": str(decision.decision_id),
                            "policy_version": decision.decision_policy_version,
                            "ai_score_snapshot_id": str(decision.ai_score_snapshot_id),
                            "blocker_count": len(decision.blockers),
                            "warning_count": len(decision.warnings),
                            "is_eligible": decision.state == DecisionState.ELIGIBLE,
                            "analytical_decision": True,
                            "trade_execution": False,
                        },
                        created_at=decision.decided_at,
                    )
                )
            except Exception:
                self.metrics.feature_store_failures_total += 1
        if not publish_events or decision.mode == DecisionMode.REPLAY and not self.config.publish_replay_events:
            return
        event_types: dict[DecisionState, type[Event]] = {
            DecisionState.ELIGIBLE: SignalDecisionEligible,
            DecisionState.OBSERVE_ONLY: SignalDecisionObserveOnly,
            DecisionState.BLOCKED: SignalDecisionBlocked,
            DecisionState.INSUFFICIENT_EVIDENCE: SignalDecisionInsufficientEvidence,
            DecisionState.INVALID: SignalDecisionInvalid,
        }
        event_type = event_types.get(decision.state)
        if event_type is None:
            return
        payload: dict[str, object] = {
            "schema_version": self.config.schema_version,
            "decision_id": str(decision.decision_id),
            "decision_key": decision.decision_key,
            "instrument": decision.instrument,
            "timeframe": decision.timeframe,
            "direction": decision.direction.value,
            "state": decision.state.value,
            "as_of": decision.as_of.isoformat(),
            "valid_until": decision.valid_until.isoformat(),
            "ai_score_snapshot_id": str(decision.ai_score_snapshot_id),
            "policy_version": decision.decision_policy_version,
            "eligibility_score": decision.eligibility_score,
            "blocker_count": len(decision.blockers),
            "warning_count": len(decision.warnings),
            "input_fingerprint": decision.input_fingerprint,
            "mode": decision.mode.value,
            "trade_execution": False,
        }
        try:
            await self.event_bus.publish(event_type(event_id=stable_id("decision-event", event_type.__name__, decision.decision_id), correlation_id=correlation_id, occurred_at=decision.decided_at, source="signal_decision", payload=payload))
        except Exception:
            self.metrics.event_publish_failures_total += 1

    async def cleanup(self) -> int:
        self.metrics.expiration_runs_total += 1
        now = self.clock.now()
        live = await self.repository.prune(now - timedelta(days=self.config.retention.live_days), DecisionMode.LIVE, self.config.retention.cleanup_batch_size)
        replay = await self.repository.prune(now - timedelta(days=self.config.retention.replay_days), DecisionMode.REPLAY, self.config.retention.cleanup_batch_size)
        return live + replay

    async def get_decision(self, decision_id: UUID) -> SignalDecision | None:
        decision = await self.repository.get_decision(decision_id)
        if decision is None or self.clock.now() < decision.valid_until:
            return decision
        return decision.model_copy(
            update={
                "state": DecisionState.EXPIRED,
                "status": DecisionLifecycleStatus.EXPIRED,
                "explanation": decision.explanation.model_copy(
                    update={"summary_code": f"{decision.direction.value}_expired", "decision_state": DecisionState.EXPIRED}
                ),
            }
        )

    def health(self) -> dict[str, object]:
        running = self.initialized and not self.closed
        persistence = "healthy" if self.repository_mode != "memory" else "unavailable" if self.config.persistence_required_in_production else "degraded"
        ai_status = self.ai_scoring.health()["status"]
        if not running or persistence == "unavailable" or ai_status == "unavailable":
            status = "unavailable"
        elif persistence != "healthy" or ai_status != "healthy" or self.metrics.feature_store_failures_total or self.metrics.event_publish_failures_total:
            status = "degraded"
        else:
            status = "healthy"
        return {
            "status": status,
            "engine": "signal_decision",
            "version": self.config.engine_version,
            "policy": {"name": self.config.policy_name, "version": self.config.policy_version},
            "persistence": {"status": persistence, "mode": self.repository_mode},
            "ai_scoring": {"status": ai_status},
            "economic_calendar": {"status": "healthy" if self.economic_calendar else "degraded"},
            "market_regime": {"status": "healthy" if self.market_regime else "degraded"},
            "feature_store": {"status": "healthy" if not self.metrics.feature_store_failures_total else "degraded"},
            "event_bus": {"status": "healthy" if not self.metrics.event_publish_failures_total else "degraded"},
            "analytical_decision": True,
            "trade_execution": False,
            "timestamp": self.clock.now(),
        }
