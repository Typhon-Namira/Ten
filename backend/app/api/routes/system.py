from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Request

from backend.app.api.market_intelligence import build_market_intelligence
from backend.app.api.safe import safe_call
from backend.app.engines.economic_calendar_engine.analyzer import staged_diagnostics
from backend.app.engines.market_data_engine import Timeframe
from backend.app.engines.market_data_engine.repository import SqlAlchemyMarketDataRepository
from backend.app.engines.market_data_engine.symbols import provider_symbol

router = APIRouter(prefix="/api/v1/system", tags=["system-diagnostics"])


def _default_selection(request: Request) -> tuple[str, str]:
    """The one authoritative (instrument, timeframe) pair the pipeline actually runs — the first
    entries of `settings.market_data_symbols`/`market_data_timeframes`. Every dashboard-facing
    aggregation endpoint below defaults to this instead of a hardcoded "XAUUSD"/"M15" literal, so
    a deployment that configures a different primary timeframe (e.g. M1) doesn't silently leave
    every widget querying a pair the pipeline never actually produces data for."""
    settings = request.app.state.settings
    return settings.market_data_symbols[0].upper(), settings.market_data_timeframes[0]


@router.get("/selection")
async def selection(request: Request) -> dict[str, object]:
    """The single source of truth for "which instrument/timeframe is the dashboard about" — every
    frontend data source must derive its query params from this instead of hardcoding its own
    default, which is what previously let different widgets silently disagree."""
    settings = request.app.state.settings
    instrument, timeframe = _default_selection(request)
    return {
        "instrument": instrument,
        "timeframe": timeframe,
        "configured_instruments": list(settings.market_data_symbols),
        "configured_timeframes": list(settings.market_data_timeframes),
    }


@router.get("/diagnostics")
async def diagnostics(request: Request) -> dict[str, object]:
    app = request.app
    settings = app.state.settings
    market = app.state.market_data_service
    integration = app.state.integration_service
    symbol, timeframe_value = _default_selection(request)
    timeframe = Timeframe(timeframe_value)
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
    runtime_provider = market.manager.current_provider or settings.market_data_provider
    provider_stats = market.manager.statistics.get(runtime_provider)
    replay_enabled = settings.replay_worker_enabled
    database_healthy = isinstance(market.repository, SqlAlchemyMarketDataRepository)
    market_worker = app.state.market_data_worker.status()
    integration_worker = app.state.integration_worker.status(settings.integration_worker_enabled)
    provider_capabilities = market.manager.registry.get(runtime_provider).capabilities
    required_candle_count = min(
        settings.market_data_bootstrap_candles,
        provider_capabilities.maximum_history_candles
        or settings.market_data_bootstrap_candles,
    )
    history_initialized = candle_count >= required_candle_count
    # Measured from the candle's CLOSE time, matching the integration layer's own freshness gate
    # (`FullSystemIntegrationService.process()`'s `age = clock() - envelope.payload.close_time`).
    # Measuring from open time (the previous formula) understated a candle's real age by a full
    # bar duration — for M15 that's 900 extra seconds of apparent freshness, enough to mask this
    # dashboard ever reporting `STALE_MARKET_DATA` while the integration layer was already
    # silently blocking the same candle as stale internally.
    candle_age_seconds = max(0, (now - (candle.timestamp + candle.timeframe.duration)).total_seconds()) if candle else None
    candle_fresh = candle_age_seconds is not None and candle_age_seconds <= settings.max_candle_staleness_seconds
    if not database_healthy:
        operational_state = "DEGRADED_DATABASE"
    elif market_worker["crashed"] or integration_worker["crashed"]:
        # Distinct from "disabled": the worker is configured on but its background task is not
        # actually running (crashed, or never started) — a real outage that `enabled` alone could
        # never reveal, since `enabled` only reflects static config. This is what let a genuinely
        # dead worker report "Healthy" indefinitely: `NewCandle` stops firing (or the outbox stops
        # draining) with zero visible signal anywhere else in this response.
        operational_state = "DEGRADED_WORKER_DEAD"
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
            "name": runtime_provider,
            "preferred_provider": settings.market_data_provider,
            "configured_symbol": symbol,
            "provider_symbol": provider_symbol(runtime_provider, symbol),
            "status": "healthy" if provider_stats and provider_stats.healthy else "unavailable" if provider_stats is None else "degraded",
            "authentication_configured": runtime_provider in market.manager.statistics,
            "last_success_at": provider_stats.last_success_at if provider_stats else None,
            "last_failure_at": provider_stats.last_failure_at if provider_stats else None,
            "last_error": provider_stats.last_error if provider_stats else "ProviderNotConfigured",
            "rate_limit": provider_stats.quota.model_dump(mode="json") if provider_stats else None,
        },
        "market": {
            "symbol": symbol,
            "timeframe": timeframe.value,
            **schedule.model_dump(mode="json"),
            "latest_candle_at": candle.timestamp if candle else None,
            # Measured from close time, not open time — see the comment above `candle_age_seconds`.
            "latest_candle_age_seconds": candle_age_seconds,
            "freshness": "absent" if candle is None else "fresh" if candle_fresh else "stale",
        },
        "history": {
            "candle_count": candle_count,
            "required_candle_count": required_candle_count,
            "requested_bootstrap_candle_count": settings.market_data_bootstrap_candles,
            "initialized": history_initialized,
        },
        "workers": {
            "market_data_worker": market_worker,
            "integration_worker": integration_worker,
            "retention_worker": app.state.retention_worker.status(),
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
            "configured": settings.groq_pool_enabled and any(
                key is not None
                for index, key in enumerate(
                    settings.groq_pool_api_keys,
                    start=1,
                )
                if index <= settings.groq_pool_size
            ),
            "primary_provider": "Groq pool",
            "configured_account_count": sum(
                settings.groq_pool_enabled
                and index <= settings.groq_pool_size
                and key is not None
                for index, key in enumerate(
                    settings.groq_pool_api_keys,
                    start=1,
                )
            ),
            "providers": {
                f"groq_{index}": {
                    "configured": (
                        settings.groq_pool_enabled
                        and index <= settings.groq_pool_size
                        and key is not None
                    ),
                    "base_url_configured": bool(settings.groq_base_url),
                    "model": settings.groq_model,
                }
                for index, key in enumerate(
                    settings.groq_pool_api_keys,
                    start=1,
                )
            },
            "latest_status": app.state.ai_scoring_service.metrics.snapshot(),
        },
        "replay": {"enabled": replay_enabled, "status": "enabled" if replay_enabled else "disabled"},
    }


@router.get("/market-intelligence")
async def market_intelligence(request: Request, instrument: str | None = None, timeframe: str | None = None) -> dict[str, object]:
    """Every live field for the market-intelligence panel. Never 500s: a failing source is
    reported per-field in `source_errors` and every other field is still returned."""
    app = request.app
    market = app.state.market_data_service
    default_instrument, default_timeframe = _default_selection(request)
    symbol = (instrument or default_instrument).upper()
    timeframe = timeframe or default_timeframe
    tf = Timeframe(timeframe)
    now = datetime.now(UTC)
    errors: dict[str, str | None] = {}

    candle, errors["candle"] = await safe_call(lambda: market.latest(symbol, tf))
    spread_supported: bool | None = None
    if candle is not None:
        try:
            spread_supported = market.manager.registry.get(candle.provider).capabilities.spread
        except Exception:
            spread_supported = None
    smc, errors["smc"] = await safe_call(lambda: app.state.smc_service.state(symbol, tf))
    liquidity, errors["liquidity"] = await safe_call(lambda: app.state.liquidity_service.state(symbol, tf))
    volume_profile, errors["volume_profile"] = await safe_call(lambda: app.state.volume_profile_service.state(symbol, tf))
    institutional_flow, errors["institutional_flow"] = await safe_call(lambda: app.state.institutional_flow_service.state(symbol, tf))
    market_regime, errors["market_regime"] = await safe_call(lambda: app.state.market_regime_service.state(symbol, tf))
    economic_context, errors["economic_calendar"] = await safe_call(lambda: app.state.economic_calendar_service.context(symbol, as_of=now, publish=False))
    economic_stages: dict[str, object] | None = None
    if economic_context is not None:
        calendar = app.state.economic_calendar_service
        economic_snapshot, _ = await safe_call(lambda: calendar.snapshot(now - timedelta(days=2), now + timedelta(days=7), as_of=now))
        if economic_snapshot is not None:
            economic_stages = staged_diagnostics(economic_snapshot, economic_context)
    ai_score, errors["ai_scoring"] = await safe_call(lambda: app.state.ai_scoring_service.repository.get_latest_snapshot(symbol, timeframe))
    # The most recently *made* decision, not the most recently *active* one — a decision whose
    # validity window has since lapsed is still the correct thing to show on a "current state"
    # panel (labeled via `decision_active`), matching how every other source here reports its
    # latest-ever snapshot rather than requiring it to still be within some window.
    decisions, errors["signal_decision"] = await safe_call(lambda: app.state.signal_decision_service.repository.find_recent_decisions(symbol, timeframe, now, 1))
    decision = decisions[0] if decisions else None

    # A stale-but-successful snapshot must not read as "healthy" if the pipeline's latest attempt
    # at that stage actually failed — cross-reference the stage tracker (which knows about the
    # current/most recent run, success or not) independently of whatever was last persisted.
    tracker = app.state.pipeline_stage_tracker
    stage_attempts = {
        "smc": tracker.stage_attempt(symbol, timeframe, "smc_analysis"),
        "liquidity": tracker.stage_attempt(symbol, timeframe, "liquidity_analysis"),
        "volume_profile": tracker.stage_attempt(symbol, timeframe, "volume_profile"),
        "institutional_flow": tracker.stage_attempt(symbol, timeframe, "institutional_flow"),
        "market_regime": tracker.stage_attempt(symbol, timeframe, "market_regime"),
        "ai_scoring": tracker.stage_attempt(symbol, timeframe, "ai_scoring"),
        "signal_decision": tracker.stage_attempt(symbol, timeframe, "scenario_decision"),
    }

    try:
        session_status = market.sessions.status_at(now)
        errors["session"] = None
    except Exception as exc:
        session_status, errors["session"] = None, type(exc).__name__
    return build_market_intelligence(
        instrument=symbol,
        timeframe=timeframe,
        now=now,
        session=session_status,
        candle=candle,
        spread_supported=spread_supported,
        smc=smc,
        liquidity=liquidity,
        volume_profile=volume_profile,
        institutional_flow=institutional_flow,
        market_regime=market_regime,
        economic_context=economic_context,
        ai_score=ai_score,
        decision=decision,
        errors={key: value for key, value in errors.items() if value is not None},
        stage_attempts=stage_attempts,
        economic_stages=economic_stages,
    )


@router.get("/performance")
async def performance(request: Request, instrument: str | None = None, timeframe: str | None = None) -> dict[str, object]:
    """Pipeline/provider/database/analysis latency, last successful/failed provider call,
    queue length, and worker health — never 500s, unavailable metrics are reported as null."""
    app = request.app
    settings = app.state.settings
    market = app.state.market_data_service
    integration = app.state.integration_service
    default_instrument, default_timeframe = _default_selection(request)
    symbol = (instrument or default_instrument).upper()
    timeframe = timeframe or default_timeframe
    tf = Timeframe(timeframe)
    provider_stats = market.manager.statistics.get(settings.market_data_provider)
    integration_metrics = integration.repository.metrics()
    stage = app.state.pipeline_stage_tracker.latest(symbol, timeframe)
    now = datetime.now(UTC)
    # `pipeline_latency_ms` (last COMPLETED cycle's duration) and `pipeline_in_flight_ms` (the
    # CURRENT cycle's elapsed time so far, if one is running) are deliberately separate metrics —
    # collapsing them into one field is what previously left latency reported as unavailable
    # exactly when the pipeline was busiest: a cycle that is still running has no completed
    # duration yet, but "elapsed so far" is exactly the number an operator wants at that moment.
    pipeline_latency_ms = None
    pipeline_in_flight_ms = None
    if stage and stage.get("started_at"):
        if stage.get("complete"):
            pipeline_latency_ms = (stage["updated_at"] - stage["started_at"]).total_seconds() * 1000
        else:
            pipeline_in_flight_ms = (now - stage["started_at"]).total_seconds() * 1000
    # If queued work exists but no cycle has even started yet (worker stalled, or the tracker has
    # no record for this exact instrument/timeframe), the oldest pending outbox item's age is the
    # only true measure of "how long has this been waiting" — never leave it null while backlog > 0.
    queue_oldest_pending_age_seconds = None
    if integration_metrics.get("outbox_backlog"):
        oldest_pending, _ = await safe_call(integration.repository.oldest_pending)
        if oldest_pending is not None:
            queue_oldest_pending_age_seconds = max(0.0, (now - oldest_pending.available_at).total_seconds())
    # Database, cache, and provider are independent sources of truth that can legitimately
    # disagree (e.g. provider rate-limited but the database still serves the last good candle) —
    # reported separately rather than collapsed into one "market data" status.
    database_candle, _ = await safe_call(lambda: market.repository.candle_at(symbol, tf, now))
    database_engine = getattr(app.state, "smc_database_engine", None)
    pool = getattr(database_engine, "pool", None)
    from backend.app.storage.scoped_session import session_runtime_metrics

    def pool_value(name: str) -> int | None:
        value = getattr(pool, name, None)
        return int(value()) if callable(value) else None

    return {
        "instrument": symbol,
        "timeframe": timeframe,
        "pipeline_latency_ms": pipeline_latency_ms,
        "pipeline_in_flight_ms": pipeline_in_flight_ms,
        "queue_oldest_pending_age_seconds": queue_oldest_pending_age_seconds,
        "provider": {
            "name": settings.market_data_provider,
            "last_latency_ms": getattr(provider_stats, "last_latency_ms", None) if provider_stats else None,
            # `last_latency_ms` is the true single-HTTP-call bottleneck; `last_total_latency_ms`
            # is the full wall-clock time including retry backoff/rate-gate waits, and
            # `last_retry_count` says how many of those retries happened — together they explain
            # a large total instead of it reading as an unexplained slow API response.
            "last_total_latency_ms": provider_stats.last_total_latency_ms if provider_stats else None,
            "last_retry_count": provider_stats.last_retry_count if provider_stats else None,
            "last_provider_success": provider_stats.last_success_at if provider_stats else None,
            "last_provider_failure": provider_stats.last_failure_at if provider_stats else None,
            "last_success_at": provider_stats.last_success_at if provider_stats else None,
            "last_failure_at": provider_stats.last_failure_at if provider_stats else None,
            "last_error": provider_stats.last_error if provider_stats else None,
            "healthy": bool(provider_stats and provider_stats.healthy),
            "consecutive_failures": provider_stats.consecutive_failures if provider_stats else None,
            "provider_backoff_until": provider_stats.backoff_until if provider_stats else None,
            "provider_rate_limit_remaining": provider_stats.quota.remaining if provider_stats else None,
            "provider_rate_limit": provider_stats.quota.limit if provider_stats else None,
        },
        "database": {
            "mode": integration.repository_mode,
            "events": integration_metrics.get("events"),
            "outbox_backlog": integration_metrics.get("outbox_backlog"),
            "processed": integration_metrics.get("processed"),
            "last_database_update": database_candle.ingestion_timestamp if database_candle else None,
            "latest_market_candle": database_candle.timestamp if database_candle else None,
            "pool_size": pool_value("size"),
            "checked_in": pool_value("checkedin"),
            "checked_out": pool_value("checkedout"),
            "overflow": pool_value("overflow"),
            "sessions": session_runtime_metrics.snapshot(),
        },
        "cache": {
            "last_cache_update": market.cache.statistics.last_write_at,
            "hit_ratio": market.cache.statistics.hit_ratio,
            "writes": market.cache.statistics.writes,
            **market.cache.metrics(),
        },
        "analysis": {
            "ai_scoring": app.state.ai_scoring_service.metrics.snapshot(),
            "signal_decision": app.state.signal_decision_service.metrics.snapshot(),
            "ai_reasoning": app.state.ai_reasoning_service.health()["call_control"],
        },
        "queue_length": integration_metrics.get("outbox_backlog"),
        "workers": {
            "market_data_worker": app.state.market_data_worker.status(),
            "integration_worker": app.state.integration_worker.status(settings.integration_worker_enabled),
            "retention_worker": app.state.retention_worker.status(),
        },
        "event_bus": app.state.pipeline_manager.event_bus.metrics(),
        "feature_store": app.state.pipeline_manager.feature_store.metrics(),
        "dashboard_events": app.state.pipeline_activity_log.metrics(),
        "market_polling": dict(market.poll_metrics),
    }
