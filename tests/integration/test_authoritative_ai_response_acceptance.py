from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from backend.app.ai_reasoning.config import AIReasoningConfig
from backend.app.ai_reasoning.llm_context import build_llm_analysis_context
from backend.app.ai_reasoning.models import MarketMemorySummary
from backend.app.ai_reasoning.request_builder import AIReasoningRequestBuilder
from backend.app.core.config import YamlConfigRepository
from backend.app.engines.market_data_engine import Timeframe
from backend.app.engines.signal_decision_engine import ConservativeSignalDecisionPolicy
from backend.app.scenario_forecasting import InMemoryScenarioForecastRepository
from backend.app.scenario_forecasting.simulation_engine import (
    MarketSimulationConfig,
    MarketSimulationEngine,
)
from backend.app.scenario_forecasting.simulation_repository import (
    InMemoryMarketSimulationRepository,
)
from backend.app.scenario_forecasting.simulation_service import MarketSimulationService
from backend.app.signal_notifications.service import (
    primary_scenario_email_outbox_values,
)
from backend.app.signal_synthesis import MultiTimeframeSignalSynthesizer
from tests.ai_reasoning.test_ai_reasoning_lifecycle import (
    InMemoryAIReasoningRepository,
    build_service,
    state_and_quant,
)
from tests.ai_reasoning.test_token_budget import CompactClient, compact_output, provider
from tests.engines.signal_decision_engine.test_signal_decision_engine import (
    ai_score,
    decision_input,
)


@pytest.mark.asyncio
async def test_three_m15_cutoffs_normalize_commit_select_and_enqueue_once() -> None:
    """Exercise the authoritative path across three consecutive M15 cutoffs."""

    first_cutoff = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)
    ai_repository = InMemoryAIReasoningRepository()
    simulation_repository = InMemoryMarketSimulationRepository()
    simulation_service = MarketSimulationService(
        simulation_repository,
        InMemoryScenarioForecastRepository(),
        MarketSimulationEngine(
            MarketSimulationConfig(
                primary_scenario_threshold=0,
                email_scenario_threshold=0,
            )
        ),
        ai_analysis_repository=ai_repository,
        ai_lookup_retry_delays_seconds=(),
    )
    queued_email_ids: set[object] = set()
    analysis_ids: set[object] = set()
    primary_scenario_ids: set[object] = set()

    for index in range(3):
        cutoff = first_cutoff + timedelta(minutes=15 * index)
        state, quant = await state_and_quant(
            cutoff,
            trigger=Timeframe.M15,
        )
        config = YamlConfigRepository().load_model(
            "ai_reasoning",
            AIReasoningConfig,
        )
        request = AIReasoningRequestBuilder(
            config,
            model_identifier="openai/gpt-oss-120b",
            clock=lambda cutoff=cutoff: cutoff + timedelta(seconds=5),
        ).build(
            state,
            quant,
            MarketMemorySummary(entry_count=0),
            existing_signal=None,
            previous_forecast=None,
            previous_proposal=None,
        )
        raw = compact_output(request)
        raw["market_regime"]["classification"] = " Bullish-Trend "
        raw["higher_timeframe_context"]["summary"] = [
            "M5 structure is constructive.",
            "M15 context remains aligned.",
        ]
        evidence_refs = [
            item.evidence_id
            for item in build_llm_analysis_context(request).evidence_catalog[:3]
        ]
        assert len(evidence_refs) == 3
        raw["higher_timeframe_context"]["evidence_refs"] = evidence_refs
        client = CompactClient(raw)
        selected_provider = provider(client, config)

        reasoning = await build_service(
            ai_repository,
            selected_provider,
            now=cutoff + timedelta(seconds=5),
        ).process(state, quant)

        assert reasoning is not None
        assert reasoning.analysis.validation_passed
        assert client.calls and len(client.calls) == 1
        assert selected_provider.correction_attempts == 0
        assert selected_provider.request_attempts[0]["local_shape_normalizations"]
        assert "market_regime.classification" in selected_provider.request_attempts[0][
            "local_shape_normalizations"
        ]
        analysis_ids.add(reasoning.analysis.analysis_id)
        synthesis = MultiTimeframeSignalSynthesizer().synthesize(
            state,
            quant,
            reasoning.analysis,
        )
        selection = await simulation_service.process(
            state,
            quant,
            synthesis,
            trigger_timeframe="M15",
            evaluated_at=cutoff,
        )
        assert selection is not None
        assert selection.primary is not None
        primary_scenario_ids.add(selection.primary.candidate_id)

        score = ai_score(as_of=cutoff, calculated_at=cutoff)
        decision = ConservativeSignalDecisionPolicy().evaluate(
            decision_input(score=score, as_of=cutoff).model_copy(
                update={
                    "current_ai_analysis": reasoning.analysis,
                    "market_snapshot_id": state.state_id,
                    "quantitative_forecast_id": quant.result_id,
                    "current_price": selection.primary.reference_price,
                    "expected_move": selection.primary.expected_move,
                    "current_primary_scenario": selection,
                }
            )
        )
        assert decision.publication_eligible
        first_email = primary_scenario_email_outbox_values(
            selection,
            decision,
            "operator@example.com",
            decision.decided_at,
        )
        repeated_email = primary_scenario_email_outbox_values(
            selection,
            decision,
            "operator@example.com",
            decision.decided_at,
        )
        assert first_email is not None
        assert repeated_email is not None
        assert first_email["id"] == repeated_email["id"]
        queued_email_ids.add(first_email["id"])

    assert len(analysis_ids) == 3
    assert len(primary_scenario_ids) == 3
    assert len(queued_email_ids) == 3
    assert len(ai_repository.response_artifacts) == 3
    assert all(
        artifact.status.value == "COMMITTED"
        for artifact in ai_repository.response_artifacts.values()
    )
    assert len(simulation_repository.simulations) == 3
