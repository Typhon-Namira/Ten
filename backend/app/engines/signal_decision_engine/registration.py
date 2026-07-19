from collections.abc import Mapping
from datetime import date
from typing import Any

from backend.app.engines.ai_scoring_engine import AIScoreSnapshot
from backend.app.engines.common import EngineMetadata
from backend.app.services.engine_factory import EngineBuildContext, EngineFactory
from backend.app.services.pipeline_contracts import EngineExecutionResult, PipelineExecutionContext

from .config import SignalDecisionConfig
from .engine import ConservativeSignalDecisionPolicy
from .events import SignalDecisionBlocked
from .models import DecisionMode, SignalDecisionInput


def _build(_: EngineBuildContext, config: Mapping[str, Any]) -> ConservativeSignalDecisionPolicy:
    return ConservativeSignalDecisionPolicy(SignalDecisionConfig.model_validate(config))


async def _execute(policy: Any, context: PipelineExecutionContext) -> EngineExecutionResult:
    if not isinstance(policy, ConservativeSignalDecisionPolicy):
        raise TypeError("signal_decision requires an approved deterministic policy")
    score = context.results.get("ai_scoring")
    if not isinstance(score, AIScoreSnapshot):
        raise ValueError("persisted AI Scoring output is required")
    decision = policy.evaluate(
        SignalDecisionInput(
            instrument=score.instrument,
            timeframe=score.timeframe,
            as_of=score.as_of,
            requested_at=context.now,
            ai_score=score,
            mode=DecisionMode.REPLAY if score.mode.value == "replay" else DecisionMode.LIVE,
            policy_name=policy.name,
            policy_version=policy.version,
        )
    )
    return EngineExecutionResult(
        output=decision,
        features={
            "state": decision.state.value,
            "direction": decision.direction.value,
            "eligibility_score": decision.eligibility_score,
            "is_eligible": False,
            "analytical_decision": True,
            "trade_execution": False,
        },
        namespace="signal_decision",
        event_type=SignalDecisionBlocked,
        confidence_factor=decision.confidence_score / 100,
    )


def register(factory: EngineFactory) -> None:
    factory.register(
        EngineMetadata(
            name="signal_decision",
            version="1.0.0",
            compatibility_version="1.0",
            created_date=date(2026, 7, 19),
            dependencies=("ai_scoring", "economic_calendar", "market_regime"),
            description="Deterministic, fail-closed analytical decision policy; no order generation or execution.",
            config_key="signal_decision",
        ),
        _build,
        _execute,
    )
