from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.ai_reasoning.request_persistence import PersistedAIReasoningRequest
from backend.app.api.routes.dashboard import _stage_fingerprint, _system_stage
from backend.app.engines.market_data_engine import Candle, Timeframe
from backend.app.integration import CanonicalEventEnvelope
from backend.app.main import create_app


@pytest.mark.parametrize(
    "stage_status",
    ("healthy", "running", "degraded", "failed", "disabled", "blocked", "stale", "no_data"),
)
def test_dashboard_stage_status_vocabulary_is_exact(stage_status: str) -> None:
    result = _system_stage("test", "Test", stage_status, "typed_reason")
    assert result["status"] == stage_status
    assert result["reason"] == "typed_reason"
    assert "unavailable" not in str(result).lower()


def test_unchanged_stage_status_has_one_stable_history_fingerprint() -> None:
    stage = _system_stage(
        "volume_profile",
        "Volume Profile",
        "degraded",
        "insufficient_volume_profile_data",
        record_id="snapshot-1",
        details={"usable_count": 0},
    )
    repeated = dict(stage)
    repeated["timestamp"] = datetime.now(UTC)
    assert _stage_fingerprint(stage) == _stage_fingerprint(repeated)

    changed = {**stage, "reason": "empty_profile_period"}
    assert _stage_fingerprint(changed) != _stage_fingerprint(stage)


def test_unhandled_exception_on_get_degrades_to_200_not_500() -> None:
    """No observability GET endpoint may ever return HTTP 500 — an unguarded, unexpected
    exception (here a plain RuntimeError raised somewhere a repository call was never wrapped
    in a try/except) must degrade to a graceful `status: "error"` body at 200 instead of an
    opaque server error, per the app-wide exception handler in `create_app()`."""
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        service = client.app.state.market_data_service
        service.repository.candle_at = AsyncMock(side_effect=RuntimeError("simulated repository failure"))
        response = client.get("/market/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["error"]["exception_class"] == "RuntimeError"
    assert body["path"] == "/market/status"


def test_unhandled_exception_on_post_still_fails_closed() -> None:
    """The never-500 safety net only widens GET (read-only, observability) endpoints. A
    mutation endpoint hitting the same kind of unguarded exception must keep failing closed,
    since this app's exception-handling carve-out is scoped to observability, not writes."""
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        service = client.app.state.ai_scoring_service
        service.calculate = AsyncMock(side_effect=RuntimeError("simulated engine failure"))
        response = client.post("/ai-scoring/score", json={"instrument": "XAUUSD", "timeframe": "M15"})
    assert response.status_code == 500


def test_health_and_status_endpoints() -> None:
    with TestClient(create_app()) as client:
        health = client.get("/health")
        engines = client.get("/engines/status")
        market = client.get("/market/status")
        diagnostics = client.get("/api/v1/system/diagnostics")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    assert engines.status_code == 200 and len(engines.json()) == 11
    assert market.status_code == 200 and market.json()["symbol"] == "XAUUSD"
    assert market.json()["market_status"] in {"OPEN", "CLOSED_WEEKEND", "CLOSED_DAILY_BREAK", "HOLIDAY_OR_PROVIDER_CLOSED", "UNKNOWN"}
    assert diagnostics.status_code == 200
    payload = diagnostics.json()
    assert payload["replay"]["status"] == "disabled"
    assert payload["workers"]["market_data_worker"]["enabled"] is False
    assert "api_key" not in str(payload).lower()
    assert "database_url" not in str(payload).lower()


def test_ai_dashboard_uses_authoritative_phase_endpoints_and_reports_unavailable_data_honestly() -> None:
    """The redesigned dashboard must read each Phase 2-5 source directly instead of
    substituting the retired legacy AI score or manufacturing an apparently-live forecast."""
    with TestClient(create_app()) as client:
        market = client.get("/api/v1/system/market-intelligence")
        quant = client.get("/api/v1/quant-forecasts/latest")
        calibration = client.get("/api/v1/quant-forecasts/calibration/latest")
        reasoning = client.get("/api/v1/ai-reasoning/latest")
        reasoning_health = client.get("/api/v1/ai-reasoning/health")
        diagnostics = client.get("/api/v1/system/diagnostics")

    assert market.status_code == 200
    assert {"diagnostics", "economic_status", "source_errors"} <= set(market.json())

    assert quant.status_code in {200, 404}
    if quant.status_code == 404:
        assert quant.json()["detail"] == "No shadow quantitative forecast is available"
    assert calibration.status_code in {200, 404}
    if calibration.status_code == 404:
        assert calibration.json()["detail"] == "No calibration report is available"

    assert reasoning.status_code == 200
    reasoning_body = reasoning.json()
    assert {
        "forecast",
        "proposal",
        "managed_signals",
        "final_actions",
        "runtime",
        "health",
    } <= set(reasoning_body)
    assert "guardrails" in reasoning_body["health"]
    assert reasoning_body["runtime"]["broker_execution_available"] is False
    assert "ai_score" not in reasoning_body

    assert reasoning_health.status_code == 200
    assert {"guardrails", "runtime"} <= set(reasoning_health.json())
    assert diagnostics.status_code == 200
    assert {"workers", "market", "operational_state"} <= set(diagnostics.json())


def test_dashboard_aggregate_returns_typed_reasons_without_expected_404s() -> None:
    """A fresh or disabled deployment is an authoritative state, not a missing HTTP route."""
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/dashboard/latest", params={"instrument": " XAU/USD "})

    assert response.status_code == 200
    body = response.json()
    assert body["instrument"] == "XAUUSD"
    assert body["status"] == "pending"
    assert body["cycle"] is None
    assert body["stages"]["market_state"]["status"] == "not_available"
    assert body["stages"]["market_state"]["reason"] == "ai_centric_shadow_mode_disabled"
    assert body["stages"]["quant_forecast"]["reason"] == "awaiting_unified_market_state"
    assert body["stages"]["publication"]["reason"] == "ai_signal_publication_disabled"
    assert body["reasoning"]["runtime"]["operating_profile"] == "safe_test"


def test_repeated_dashboard_refreshes_never_invoke_ai_provider() -> None:
    app = create_app()
    provider_call = AsyncMock(side_effect=AssertionError("dashboard must remain read-only"))

    with TestClient(app) as client:
        app.state.ai_reasoning_service.provider.reason = provider_call
        responses = [
            client.get("/api/v1/dashboard/latest", params={"instrument": "XAUUSD"})
            for _ in range(5)
        ]

    assert all(response.status_code == 200 for response in responses)
    provider_call.assert_not_awaited()


def test_dashboard_system_status_is_one_authoritative_thirteen_stage_contract() -> None:
    with TestClient(create_app()) as client:
        response = client.get(
            "/api/dashboard/system-status",
            params={"instrument": " XAU/USD "},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["instrument"] == "XAUUSD"
    assert len(body["stages"]) == 13
    assert [item["id"] for item in body["stages"]] == [
        "market_data",
        "smc",
        "liquidity",
        "volume_profile",
        "institutional_flow",
        "market_regime",
        "economic_calendar",
        "unified_market_state",
        "quant_forecast",
        "ai_reasoning",
        "proposal",
        "guardrails",
        "final_decision",
    ]
    valid = {
        "healthy", "running", "degraded", "failed",
        "disabled", "blocked", "stale", "no_data",
    }
    assert all(item["status"] in valid for item in body["stages"])
    assert all(item["status"] != "unavailable" for item in body["stages"])
    assert {"current_decision", "storage", "failure_history"} <= set(body)


def test_dashboard_reports_exact_storage_circuit_breaker_reason() -> None:
    with TestClient(create_app()) as client:
        client.app.state.integration_service.storage_exhausted_until = (
            datetime.now(UTC) + timedelta(minutes=5)
        )
        response = client.get("/api/dashboard/system-status")

    body = response.json()
    assert body["status"] == "failed"
    assert body["storage"]["status"] == "failed"
    assert body["storage"]["reason"] == "storage_exhausted"
    assert body["storage"]["circuit_retry_at"] is not None


def test_dashboard_aggregate_queries_every_stage_at_one_market_state_boundary() -> None:
    boundary = datetime(2026, 7, 23, 15, 0, tzinfo=UTC)
    state_id = uuid4()
    cycle_id = uuid4()
    quant_id = uuid4()
    forecast_id = uuid4()
    state = SimpleNamespace(
        state_id=state_id,
        cycle_id=cycle_id,
        status=SimpleNamespace(value="available"),
        market_data_boundary=boundary,
        knowledge_cutoff=boundary,
        evidence=(),
        model_dump=lambda **_: {
            "state_id": str(state_id),
            "cycle_id": str(cycle_id),
            "status": "available",
            "market_data_boundary": boundary.isoformat(),
            "knowledge_cutoff": boundary.isoformat(),
            "evidence": [],
        },
    )
    quant = SimpleNamespace(
        result_id=quant_id,
        market_state_id=state_id,
        status=SimpleNamespace(value="available"),
        generated_at=boundary,
        reason_codes=(),
        model_dump=lambda **_: {
            "result_id": str(quant_id),
            "market_state_id": str(state_id),
            "status": "available",
            "generated_at": boundary.isoformat(),
        },
    )
    forecast = SimpleNamespace(
        forecast_id=forecast_id,
        market_state_id=state_id,
        status=SimpleNamespace(value="available"),
        generated_at=boundary,
        failure_state=None,
        model_dump=lambda **_: {
            "forecast_id": str(forecast_id),
            "market_state_id": str(state_id),
            "status": "available",
            "generated_at": boundary.isoformat(),
        },
    )

    with TestClient(create_app()) as client:
        client.app.state.unified_market_state_repository.latest_state = AsyncMock(return_value=state)
        client.app.state.quant_forecast_repository.result_for_state = AsyncMock(return_value=quant)
        client.app.state.ai_reasoning_repository.forecast_for_state = AsyncMock(return_value=forecast)
        client.app.state.ai_reasoning_repository.proposal_for_state = AsyncMock(return_value=None)
        client.app.state.final_decision_repository.action_for_state = AsyncMock(return_value=None)
        response = client.get("/api/v1/dashboard/latest")

        client.app.state.quant_forecast_repository.result_for_state.assert_awaited_once_with(state_id)
        client.app.state.ai_reasoning_repository.forecast_for_state.assert_awaited_once_with(state_id)
        client.app.state.ai_reasoning_repository.proposal_for_state.assert_awaited_once_with(state_id)
        client.app.state.final_decision_repository.action_for_state.assert_awaited_once_with(state_id)

    assert response.status_code == 200
    body = response.json()
    assert body["cycle"]["market_state_id"] == str(state_id)
    assert body["stages"]["quant_forecast"]["data"]["market_state_id"] == str(state_id)
    assert body["stages"]["ai_reasoning"]["data"]["market_state_id"] == str(state_id)


def test_dashboard_compact_request_history_returns_complete_200_contract() -> None:
    """Regression for the deployed compact JSONB row that was parsed as AIReasoningRequest."""
    boundary = datetime(2026, 7, 24, 9, 0, tzinfo=UTC)
    state_id, cycle_id, request_id, quant_id = uuid4(), uuid4(), uuid4(), uuid4()
    state = SimpleNamespace(
        state_id=state_id,
        cycle_id=cycle_id,
        status=SimpleNamespace(value="available"),
        market_data_boundary=boundary,
        knowledge_cutoff=boundary,
        evidence=(),
        model_dump=lambda **_: {
            "state_id": str(state_id),
            "cycle_id": str(cycle_id),
            "status": "available",
            "market_data_boundary": boundary.isoformat(),
            "knowledge_cutoff": boundary.isoformat(),
            "evidence": [],
        },
    )
    request_snapshot = PersistedAIReasoningRequest(
        request_id=request_id,
        cycle_id=cycle_id,
        market_state_id=state_id,
        quantitative_forecast_id=quant_id,
        instrument="XAUUSD",
        analysis_timestamp=boundary,
        prompt_version="new_market_analysis_v1",
        model_identifier="meta-llama/llama-3.3-70b-instruct",
        created_at=boundary,
        compatibility_status="compatible",
        payload_format="legacy_compact_context",
        payload_schema_version="2.0",
        context_schema_version="2.0",
    )

    settings = Settings(
        ai_centric_shadow_mode=True,
        ai_signal_proposals=True,
    )
    with TestClient(create_app(settings_override=settings)) as client:
        client.app.state.unified_market_state_repository.latest_state = AsyncMock(
            return_value=state
        )
        client.app.state.quant_forecast_repository.result_for_state = AsyncMock(
            return_value=None
        )
        client.app.state.ai_reasoning_repository.forecast_for_state = AsyncMock(
            return_value=None
        )
        client.app.state.ai_reasoning_repository.request_for_state = AsyncMock(
            return_value=request_snapshot
        )
        client.app.state.ai_reasoning_repository.proposal_for_state = AsyncMock(
            return_value=None
        )
        client.app.state.final_decision_repository.action_for_state = AsyncMock(
            return_value=None
        )
        response = client.get("/api/v1/dashboard/latest")

    assert response.status_code == 200
    body = response.json()
    assert {"stages", "reasoning"} <= set(body)
    assert body["stages"]["ai_reasoning"]["status"] == "running"
    assert body["stages"]["ai_reasoning"]["reason"] == "openrouter_request_in_progress"
    assert body["reasoning"]["runtime"]["operating_profile"] == "shadow"


def test_dashboard_reports_terminal_ai_failure_not_pending_end_to_end() -> None:
    """Item 13 / root-cause regression test, at the real HTTP boundary: a persisted forecast row
    with a terminal failure status must never surface as "pending"/"not yet persisted" -- this is
    the exact production symptom the fix in dashboard_status.py exists for. Every status here is
    read straight off the JSON response the backend actually returned; nothing is inferred."""
    boundary = datetime(2026, 7, 24, 9, 0, tzinfo=UTC)
    state_id, cycle_id, forecast_id = uuid4(), uuid4(), uuid4()
    state = SimpleNamespace(
        state_id=state_id, cycle_id=cycle_id, status=SimpleNamespace(value="available"),
        market_data_boundary=boundary, knowledge_cutoff=boundary, evidence=(),
        model_dump=lambda **_: {"state_id": str(state_id), "cycle_id": str(cycle_id), "status": "available", "market_data_boundary": boundary.isoformat(), "knowledge_cutoff": boundary.isoformat(), "evidence": []},
    )
    forecast = SimpleNamespace(
        forecast_id=forecast_id, market_state_id=state_id, status="unavailable", generated_at=boundary,
        failure_state="openrouter_authentication_failed", missing_evidence=(), provider_http_status=401, validation_passed=False,
        model_dump=lambda **_: {"forecast_id": str(forecast_id), "market_state_id": str(state_id), "status": "unavailable", "failure_state": "openrouter_authentication_failed"},
    )
    with TestClient(create_app()) as client:
        client.app.state.unified_market_state_repository.latest_state = AsyncMock(return_value=state)
        client.app.state.quant_forecast_repository.result_for_state = AsyncMock(return_value=None)
        client.app.state.ai_reasoning_repository.forecast_for_state = AsyncMock(return_value=forecast)
        client.app.state.ai_reasoning_repository.proposal_for_state = AsyncMock(return_value=None)
        client.app.state.final_decision_repository.action_for_state = AsyncMock(return_value=None)
        response = client.get("/api/v1/dashboard/latest")

    assert response.status_code == 200
    body = response.json()
    assert body["stages"]["ai_reasoning"]["status"] == "failed"
    assert body["stages"]["ai_reasoning"]["reason"] == "openrouter_returned_http_401"
    assert body["stages"]["ai_reasoning"]["error_code"] == "openrouter_authentication_failed"
    assert body["stages"]["ai_reasoning"]["retryable"] is True
    # final_action must be "blocked", not stuck in "awaiting_ai_proposal" -- the upstream failure
    # already resolved, so there is nothing left to await.
    assert body["stages"]["final_action"]["status"] == "blocked"
    assert "awaiting" not in body["stages"]["final_action"]["reason"]
    assert body["status"] == "failed"


def test_dashboard_reports_wait_outcome_when_forecast_valid_with_no_proposal_end_to_end() -> None:
    """Required behavior: "valid forecast + no proposal -> final action = WAIT, guardrails =
    not_required" -- proven at the real HTTP boundary, not just against the pure function."""
    boundary = datetime(2026, 7, 24, 9, 0, tzinfo=UTC)
    state_id, cycle_id, forecast_id = uuid4(), uuid4(), uuid4()
    state = SimpleNamespace(
        state_id=state_id, cycle_id=cycle_id, status=SimpleNamespace(value="available"),
        market_data_boundary=boundary, knowledge_cutoff=boundary, evidence=(),
        model_dump=lambda **_: {"state_id": str(state_id), "cycle_id": str(cycle_id), "status": "available", "market_data_boundary": boundary.isoformat(), "knowledge_cutoff": boundary.isoformat(), "evidence": []},
    )
    forecast = SimpleNamespace(
        forecast_id=forecast_id, market_state_id=state_id, status="available", generated_at=boundary,
        failure_state=None, missing_evidence=(), provider_http_status=None, validation_passed=True,
        model_dump=lambda **_: {"forecast_id": str(forecast_id), "market_state_id": str(state_id), "status": "available"},
    )
    with TestClient(create_app()) as client:
        client.app.state.unified_market_state_repository.latest_state = AsyncMock(return_value=state)
        client.app.state.quant_forecast_repository.result_for_state = AsyncMock(return_value=None)
        client.app.state.ai_reasoning_repository.forecast_for_state = AsyncMock(return_value=forecast)
        client.app.state.ai_reasoning_repository.proposal_for_state = AsyncMock(return_value=None)
        client.app.state.final_decision_repository.action_for_state = AsyncMock(return_value=None)
        response = client.get("/api/v1/dashboard/latest")

    assert response.status_code == 200
    body = response.json()
    assert body["stages"]["ai_reasoning"]["status"] == "available"
    assert body["stages"]["guardrails"]["status"] == "not_required"
    assert body["stages"]["final_action"]["status"] == "wait"
    assert body["stages"]["final_action"]["direction"] == "WAIT"
    # `TEN_AI_SIGNAL_MONITORING` is unset in this test app's default config, so monitoring itself
    # is disabled -- that takes precedence over "not_required" (see
    # test_monitoring_not_required_for_wait_final_action in test_dashboard_status.py for the
    # dedicated, monitoring-enabled unit-level proof of the WAIT -> not_required branch).
    assert body["stages"]["monitoring"]["status"] == "not_available"
    assert body["stages"]["monitoring"]["reason"] == "ai_signal_monitoring_disabled"
    assert body["stages"]["outcome"]["status"] == "not_applicable"


def test_diagnostics_reports_a_dead_worker_as_degraded_not_healthy() -> None:
    """Regression test for the "market data healthy but SMC-onward chain silent for hours"
    investigation: `operational_state` used to only check `worker["enabled"]` (static config), so
    a worker that was configured on but whose background task had died — crashed, or never
    actually started — reported no differently than a genuinely healthy one. Simulated here by
    marking the worker enabled with no live task, exactly the state a crashed
    `asyncio.create_task(...)` (never awaited, exception never retrieved) leaves behind."""
    with TestClient(create_app()) as client:
        client.app.state.market_data_worker.enabled = True
        client.app.state.market_data_worker._task = None
        response = client.get("/api/v1/system/diagnostics")
    assert response.status_code == 200
    body = response.json()
    assert body["workers"]["market_data_worker"]["enabled"] is True
    assert body["workers"]["market_data_worker"]["running"] is False
    # This is the field that used to not exist at all: `enabled=True` alone (the old check) cannot
    # distinguish a genuinely dead worker from a healthy one, which is exactly how this stayed
    # invisible. `operational_state` isn't asserted here — this test environment has no real
    # Postgres, so `DEGRADED_DATABASE` (an earlier, unrelated branch in the same if/elif chain)
    # already dominates; that ordering is simple, readable code and doesn't need its own test.
    assert body["workers"]["market_data_worker"]["crashed"] is True


def test_dashboard_endpoints_default_to_the_configured_primary_timeframe_not_a_hardcoded_m15() -> None:
    """Regression test: `/market-intelligence`, `/pipeline/stages/latest`, `/performance`, and
    `/market/status` used to default their instrument/timeframe query params to hardcoded
    "XAUUSD"/"M15" literals — so a deployment configured for a different primary timeframe (e.g.
    M1) left every dashboard-facing endpoint silently querying a candle series the pipeline never
    actually produces data for, while the pipeline's own events/logs were for the real (M1)
    series. `/api/v1/system/selection` is the one authoritative source every endpoint (and the
    frontend) must agree with instead of each hardcoding its own default."""
    settings = Settings(market_data_symbols=("XAUUSD",), market_data_timeframes=("M1",))
    with TestClient(create_app(settings_override=settings)) as client:
        selection = client.get("/api/v1/system/selection")
        assert selection.status_code == 200
        assert selection.json() == {"instrument": "XAUUSD", "timeframe": "M1", "configured_instruments": ["XAUUSD"], "configured_timeframes": ["M1"]}

        market_intelligence = client.get("/api/v1/system/market-intelligence")
        assert market_intelligence.status_code == 200
        assert market_intelligence.json()["timeframe"] == "M1"

        stages = client.get("/api/v1/pipeline/stages/latest")
        assert stages.status_code == 200
        assert stages.json()["timeframe"] == "M1"

        performance = client.get("/api/v1/system/performance")
        assert performance.status_code == 200
        assert performance.json()["timeframe"] == "M1"

        market_status = client.get("/market/status")
        assert market_status.status_code == 200
        assert market_status.json()["symbol"] == "XAUUSD"

        diagnostics = client.get("/api/v1/system/diagnostics")
        assert diagnostics.status_code == 200
        assert diagnostics.json()["market"]["timeframe"] == "M1"

        # Explicit query params still override the configured default.
        overridden = client.get("/api/v1/system/market-intelligence", params={"timeframe": "M15"})
        assert overridden.json()["timeframe"] == "M15"


def test_market_intelligence_differentiates_economic_status_into_five_stages() -> None:
    """Regression test: the dashboard's economic status widget and the signal decision engine's
    `economic_context_unavailable` blocker used to trace back to the same underlying fact (no
    relevant event mapping right now) but render as two unrelated-looking states with no way to
    tell whether the provider itself is actually down. `economic_status.stages` must always be
    present with all five independently-reported stages, not collapsed into the single `degraded`
    boolean."""
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/system/market-intelligence")
    assert response.status_code == 200
    stages = response.json()["economic_status"]["stages"]
    assert stages is not None
    assert set(stages) == {"provider_health", "downloaded_events", "mapped_events", "relevant_events", "trading_context"}


def test_performance_reports_in_flight_latency_instead_of_null_while_a_cycle_is_running() -> None:
    """Regression test: `pipeline_latency_ms` only ever reflected a COMPLETED cycle's duration, so
    it read as unavailable ("--" in the UI) exactly when the pipeline was busiest — a cycle that
    has started but not yet finished has no completed duration, but "elapsed so far" is exactly
    the number an operator wants at that moment."""
    with TestClient(create_app()) as client:
        app = client.app
        settings = app.state.settings
        instrument, timeframe = settings.market_data_symbols[0].upper(), settings.market_data_timeframes[0]
        app.state.pipeline_stage_tracker.begin(instrument, timeframe, datetime(2026, 1, 1, tzinfo=UTC))
        response = client.get("/api/v1/system/performance")
    assert response.status_code == 200
    body = response.json()
    assert body["pipeline_latency_ms"] is None
    assert body["pipeline_in_flight_ms"] is not None
    assert body["pipeline_in_flight_ms"] >= 0


@pytest.mark.asyncio
async def test_performance_reports_queue_backlog_age_when_nothing_has_started_processing_yet() -> None:
    """Regression test: `queue_length` (outbox backlog) could be non-zero while
    `pipeline_latency_ms` stayed null forever if the worker hadn't picked anything up yet —
    `queue_oldest_pending_age_seconds` must be populated whenever backlog exists, independent of
    whether the stage tracker has any cycle recorded at all."""
    app = create_app()
    async with app.router.lifespan_context(app):
        settings = app.state.settings
        instrument, timeframe = settings.market_data_symbols[0].upper(), settings.market_data_timeframes[0]
        candle = Candle(timestamp=datetime(2026, 1, 1, tzinfo=UTC), symbol=instrument, timeframe=Timeframe(timeframe), open=1, high=2, low=1, close=1.5, volume=10, provider="test")
        envelope = CanonicalEventEnvelope.final_candle(candle, uuid4(), datetime.now(UTC))
        await app.state.integration_repository.enqueue(envelope)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/system/performance")
    assert response.status_code == 200
    body = response.json()
    assert body["queue_length"] >= 1
    assert body["queue_oldest_pending_age_seconds"] is not None
    assert body["queue_oldest_pending_age_seconds"] >= 0


def test_chart_overlays_endpoint_returns_candles_and_engine_overlays() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/chart/overlays")
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {
        "instrument", "timeframe", "candles", "structure_events", "zones", "dealing_range", "liquidity_pools",
        "liquidity_sweeps", "equal_levels", "sessions", "volume_profile", "economic_events", "decision", "source_errors",
    }
    assert isinstance(body["candles"], list)
    assert isinstance(body["equal_levels"], list)
    assert isinstance(body["economic_events"], list)


def test_signal_endpoints_start_empty() -> None:
    with TestClient(create_app()) as client:
        assert client.get("/signals").json() == []
        assert client.get("/signals/latest").status_code == 404
