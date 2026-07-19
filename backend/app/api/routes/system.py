from datetime import UTC, datetime

from fastapi import APIRouter, Request

from backend.app.engines.market_data_engine import Timeframe
from backend.app.engines.market_data_engine.repository import SqlAlchemyMarketDataRepository
from backend.app.engines.market_data_engine.symbols import provider_symbol

router = APIRouter(prefix="/api/v1/system", tags=["system-diagnostics"])


@router.get("/diagnostics")
async def diagnostics(request: Request) -> dict[str, object]:
    app = request.app
    settings = app.state.settings
    market = app.state.market_data_service
    integration = app.state.integration_service
    symbol = settings.market_data_symbols[0]
    timeframe = Timeframe(settings.market_data_timeframes[0])
    now = datetime.now(UTC)
    schedule = market.sessions.status_at(now)
    candle = await market.repository.candle_at(symbol, timeframe, now)
    candle_count = await market.repository.count(symbol, timeframe)
    snapshot = await integration.repository.latest_snapshot(symbol.upper().replace("/", ""), timeframe.value)
    decisions = await app.state.signal_decision_service.repository.find_recent_decisions(symbol.upper().replace("/", ""), timeframe.value, now, 1)
    decision = decisions[0] if decisions else None
    signal = await integration.repository.latest_signal(symbol.upper().replace("/", ""), timeframe.value)
    event_metrics = app.state.pipeline_manager.event_bus.metrics()
    feature_history = await app.state.pipeline_manager.feature_store.history(mode="live", instrument=symbol.upper().replace("/", ""), timeframe=timeframe.value, limit=1)
    provider_stats = market.manager.statistics.get(settings.market_data_provider)
    replay_enabled = settings.replay_worker_enabled
    database_healthy = isinstance(market.repository, SqlAlchemyMarketDataRepository)
    market_worker = app.state.market_data_worker.status()
    integration_worker = app.state.integration_worker.status(settings.integration_worker_enabled)
    history_initialized = candle_count >= settings.market_data_bootstrap_candles
    candle_fresh = candle is not None and (now - candle.timestamp).total_seconds() <= settings.max_candle_staleness_seconds
    if not database_healthy:
        operational_state = "DEGRADED_DATABASE"
    elif not market_worker["enabled"] or not integration_worker["enabled"]:
        operational_state = "DEGRADED_WORKER_DISABLED"
    elif provider_stats is None and candle is None:
        operational_state = "PROVIDER_UNAVAILABLE"
    elif not history_initialized:
        operational_state = "INITIALIZING_MARKET_HISTORY"
    elif snapshot is None:
        operational_state = "PIPELINE_INITIALIZING"
    elif not schedule.market_open:
        operational_state = "HEALTHY_MARKET_CLOSED"
    elif not candle_fresh:
        operational_state = "STALE_MARKET_DATA"
    elif integration_worker["last_error"]:
        operational_state = "PIPELINE_DEGRADED"
    else:
        operational_state = "HEALTHY_MARKET_OPEN"
    return {
        "application_version": "0.1.0",
        "operational_state": operational_state,
        "database": {
            "status": "healthy" if database_healthy else "degraded",
            "mode": "postgresql" if database_healthy else "memory",
        },
        "provider": {
            "name": settings.market_data_provider,
            "configured_symbol": symbol,
            "provider_symbol": provider_symbol(settings.market_data_provider, symbol),
            "status": "healthy" if provider_stats and provider_stats.healthy else "unavailable" if provider_stats is None else "degraded",
            "authentication_configured": settings.market_data_provider in market.manager.statistics,
            "last_success_at": provider_stats.last_success_at if provider_stats else None,
            "last_failure_at": provider_stats.last_failure_at if provider_stats else None,
            "last_error": provider_stats.last_error if provider_stats else "ProviderNotConfigured",
            "rate_limit": provider_stats.quota.model_dump(mode="json") if provider_stats else None,
        },
        "market": {
            "symbol": "XAU/USD",
            **schedule.model_dump(mode="json"),
            "latest_candle_at": candle.timestamp if candle else None,
            "latest_candle_age_seconds": max(0, (now - candle.timestamp).total_seconds()) if candle else None,
            "freshness": "absent" if candle is None else "fresh" if (now - candle.timestamp).total_seconds() <= settings.max_candle_staleness_seconds else "stale",
        },
        "history": {"candle_count": candle_count, "required_candle_count": settings.market_data_bootstrap_candles, "initialized": history_initialized},
        "workers": {
            "market_data_worker": market_worker,
            "integration_worker": integration_worker,
        },
        "event_bus": event_metrics,
        "feature_store": {"status": "healthy", "latest_feature_at": (feature_history[0].created_at if feature_history else None)},
        "pipeline": {
            "status": integration.health()["status"],
            "latest_snapshot": snapshot.model_dump(mode="json") if snapshot else None,
            "latest_decision": decision.model_dump(mode="json") if decision else None,
            "latest_scenario": signal.model_dump(mode="json") if signal and signal.state == "eligible" else None,
        },
        "ai": {
            "configured": bool(settings.openrouter_api_key),
            "base_url_configured": bool(settings.openrouter_base_url),
            "model": settings.openrouter_model,
            "latest_status": app.state.ai_scoring_service.metrics.snapshot(),
        },
        "replay": {"enabled": replay_enabled, "status": "enabled" if replay_enabled else "disabled"},
    }
