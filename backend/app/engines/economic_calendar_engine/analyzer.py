from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .config import EconomicCalendarConfig
from .models import (
    CalendarContextState,
    ConflictState,
    ConnectionState,
    DegradationState,
    EconomicCalendarExplanation,
    EconomicCalendarSnapshot,
    EconomicEvent,
    EconomicEventRevision,
    EconomicEventStatus,
    EventCluster,
    EventImportance,
    FreshnessState,
    GENUINELY_UNAVAILABLE_STATES,
    InstrumentEventContext,
    ProviderStatus,
    RevisionType,
    RiskWindowPhase,
    payload_hash,
    stable_id,
)

_CONNECTION_STATE_TO_CATEGORY = {
    ConnectionState.UNREACHABLE: CalendarContextState.PROVIDER_UNREACHABLE.value,
    ConnectionState.TIMEOUT: CalendarContextState.PROVIDER_TIMEOUT.value,
    ConnectionState.UNAUTHORIZED: CalendarContextState.PROVIDER_AUTH_FAILED.value,
    ConnectionState.RATE_LIMITED: CalendarContextState.PROVIDER_RATE_LIMITED.value,
    ConnectionState.DISABLED: CalendarContextState.NO_CALENDAR_DATA.value,
    ConnectionState.UNKNOWN: CalendarContextState.NO_CALENDAR_DATA.value,
}


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def event_order(item: EconomicEvent) -> tuple[datetime, int, str, str]:
    importance = {EventImportance.CRITICAL: 0, EventImportance.HIGH: 1, EventImportance.MEDIUM: 2, EventImportance.LOW: 3, EventImportance.UNKNOWN: 4}
    return (item.scheduled_at_utc or datetime.max.replace(tzinfo=UTC), importance[item.importance], item.canonical_name, str(item.event_id))


def reconcile(events: tuple[EconomicEvent, ...], provider_priority: tuple[str, ...]) -> tuple[EconomicEvent, ...]:
    grouped: dict[object, list[EconomicEvent]] = {}
    for item in events:
        grouped.setdefault(item.event_id, []).append(item)
    rank = {name: index for index, name in enumerate(provider_priority)}
    result = []
    for group in grouped.values():
        values = sorted(group, key=lambda item: (rank.get(str(item.metadata.get("provider")), 9999), item.available_at, str(item.provider_records)))
        selected = values[0]
        actuals = {item.actual_value for item in values if item.actual_value is not None}
        schedules = {item.scheduled_at_utc for item in values if item.scheduled_at_utc is not None}
        conflict = ConflictState.MATERIAL if len(actuals) > 1 or len(schedules) > 1 else ConflictState.NONE
        providers = tuple(identifier for item in values for identifier in item.provider_records)
        selected = selected.model_copy(
            update={
                "provider_records": providers,
                "conflict_state": conflict,
                "source_quality": clamp(sum(item.source_quality for item in values) / len(values) - (0.2 if conflict != ConflictState.NONE else 0)),
                "metadata": {
                    **selected.metadata,
                    "reconciliation": {
                        "selected_provider": selected.metadata.get("provider"),
                        "alternatives": [item.metadata.get("provider") for item in values[1:]],
                        "rule": "configured_provider_priority",
                    },
                },
            }
        )
        result.append(selected)
    return tuple(sorted(result, key=event_order))


def revision_between(previous: EconomicEvent | None, current: EconomicEvent, number: int) -> EconomicEventRevision | None:
    if previous is None:
        changed: tuple[str, ...] = ("initial_discovery",)
        kind = RevisionType.INITIAL_DISCOVERY
    else:
        tracked = ("status", "scheduled_at_utc", "actual_value", "forecast_value", "previous_value", "revised_previous_value", "importance", "conflict_state")
        changed = tuple(name for name in tracked if getattr(previous, name) != getattr(current, name))
        if not changed:
            return None
        if current.is_corrected and ("actual_value" in changed or "revised_previous_value" in changed):
            kind = RevisionType.CORRECTION
        elif previous.actual_value is None and current.actual_value is not None:
            kind = RevisionType.FIRST_RELEASE
        elif "actual_value" in changed or "revised_previous_value" in changed:
            kind = RevisionType.VALUE_REVISION
        elif current.status == EconomicEventStatus.CANCELLED:
            kind = RevisionType.CANCELLATION
        elif current.status == EconomicEventStatus.POSTPONED:
            kind = RevisionType.POSTPONEMENT
        elif current.is_rescheduled or "scheduled_at_utc" in changed:
            kind = RevisionType.RESCHEDULE
        elif "status" in changed:
            kind = RevisionType.STATUS_CHANGE
        elif "conflict_state" in changed:
            kind = RevisionType.PROVIDER_CONFLICT
        else:
            kind = RevisionType.METADATA_UPDATE
    digest = payload_hash({"event": str(current.event_id), "number": number, "changed": changed, "available": current.available_at.isoformat()})
    return EconomicEventRevision(
        revision_id=stable_id("revision", current.event_id, digest),
        event_id=current.event_id,
        revision_number=number,
        revision_type=kind,
        observed_at=current.last_updated_at,
        available_at=current.available_at,
        provider_published_at=current.first_published_at,
        ingested_at=current.ingested_at,
        previous_status=previous.status if previous else None,
        new_status=current.status,
        previous_scheduled_at=previous.scheduled_at_utc if previous else None,
        new_scheduled_at=current.scheduled_at_utc,
        previous_actual_value=previous.actual_value if previous else None,
        new_actual_value=current.actual_value,
        previous_forecast_value=previous.forecast_value if previous else None,
        new_forecast_value=current.forecast_value,
        previous_previous_value=previous.previous_value if previous else None,
        new_previous_value=current.previous_value,
        previous_revised_previous_value=previous.revised_previous_value if previous else None,
        new_revised_previous_value=current.revised_previous_value,
        changed_fields=changed,
        provider_observation_ids=current.provider_records,
        reason=kind.value.replace("_", " "),
        payload_hash=digest,
    )


def freshness(now: datetime, statuses: tuple[ProviderStatus, ...], config: EconomicCalendarConfig) -> FreshnessState:
    successes = [item.last_success for item in statuses if item.last_success]
    if not successes:
        return FreshnessState.UNKNOWN
    age = (now - max(successes)).total_seconds() / 60
    if age < config.freshness.aging_minutes:
        return FreshnessState.FRESH
    if age < config.freshness.stale_minutes:
        return FreshnessState.AGING
    if age < config.freshness.critical_minutes:
        return FreshnessState.STALE
    return FreshnessState.CRITICAL


def build_snapshot(
    events: tuple[EconomicEvent, ...], boundary: datetime, start: datetime, end: datetime, statuses: tuple[ProviderStatus, ...], config: EconomicCalendarConfig
) -> EconomicCalendarSnapshot:
    visible = tuple(item for item in events if item.available_at <= boundary and (item.scheduled_at_utc is None or start <= item.scheduled_at_utc <= end))
    visible = tuple(sorted(visible, key=event_order))
    upcoming = tuple(item for item in visible if item.scheduled_at_utc and item.scheduled_at_utc > boundary and not item.is_cancelled)
    recent = tuple(item for item in visible if item.scheduled_at_utc and item.scheduled_at_utc <= boundary)
    active = tuple(
        item for item in visible if _phase(item, boundary, config) not in {RiskWindowPhase.OUTSIDE, RiskWindowPhase.COOLDOWN, RiskWindowPhase.UNKNOWN}
    )
    unavailable = tuple(item.provider_name for item in statuses if not item.enabled or not item.reachable)
    quality = sum(item.source_quality * item.normalization_confidence * item.data_completeness for item in visible) / len(visible) if visible else 0
    # Degraded only if NONE of the configured providers are reachable — a working fallback (e.g.
    # Finnhub) covering for a failed primary (FMP) must never read as degraded. When it IS
    # degraded, the category/reason come from the highest-priority *enabled* provider's actual
    # connection state (`statuses` is already in priority order), so the reported reason matches
    # what actually failed rather than a generic message.
    degraded = not any(item.enabled and item.reachable for item in statuses)
    category = "healthy"
    reasons: tuple[str, ...] = ()
    if degraded:
        candidates = [item for item in statuses if item.enabled] or list(statuses)
        primary = candidates[0] if candidates else None
        category = _CONNECTION_STATE_TO_CATEGORY.get(primary.connection_state, CalendarContextState.NO_CALENDAR_DATA.value) if primary else CalendarContextState.NO_CALENDAR_DATA.value
        reasons = (primary.failure_reason if primary and primary.failure_reason else "no reachable production provider",)
    return EconomicCalendarSnapshot(
        snapshot_id=stable_id("snapshot", boundary.isoformat(), start.isoformat(), end.isoformat(), *(item.event_id for item in visible)),
        configuration_version=config.version,
        schema_version=config.versions.schema_version,
        normalization_version=config.versions.normalization_version,
        analysis_timestamp=boundary,
        historical_boundary=boundary,
        created_at=boundary,
        window_start=start,
        window_end=end,
        events=visible,
        upcoming_events=upcoming,
        active_events=active,
        recent_events=recent,
        event_count=len(visible),
        high_importance_count=sum(item.importance == EventImportance.HIGH for item in visible),
        critical_importance_count=sum(item.importance == EventImportance.CRITICAL for item in visible),
        released_count=sum(item.status in {EconomicEventStatus.RELEASED, EconomicEventStatus.REVISED, EconomicEventStatus.CORRECTED} for item in visible),
        revised_count=sum(item.is_revised for item in visible),
        conflicting_count=sum(item.conflict_state != ConflictState.NONE for item in visible),
        unavailable_count=sum(item.scheduled_at_utc is None for item in visible),
        provider_status=statuses,
        degradation=DegradationState(
            is_degraded=degraded,
            category=category,
            reasons=reasons,
            unavailable_providers=unavailable,
            partial_results=bool(visible) and degraded,
        ),
        quality=clamp(quality),
        freshness=freshness(boundary, statuses, config),
    )


def symbol_currencies(symbol: str, config: EconomicCalendarConfig) -> tuple[str, ...]:
    clean = "".join(character for character in symbol.upper() if character.isalnum())
    if clean in config.symbol_overrides:
        return config.symbol_overrides[clean]
    known = set(config.country_currency.values())
    return tuple(code for code in known if code in clean)


def _phase(event: EconomicEvent, now: datetime, config: EconomicCalendarConfig) -> RiskWindowPhase:
    if event.is_cancelled or event.is_postponed:
        return RiskWindowPhase.OUTSIDE
    if event.scheduled_at_utc is None:
        return RiskWindowPhase.UNKNOWN
    window = config.windows[event.importance.value]
    minutes = (event.scheduled_at_utc - now).total_seconds() / 60
    if abs(minutes) < 1:
        return RiskWindowPhase.AT_EVENT
    if 0 < minutes <= window.imminent_minutes:
        return RiskWindowPhase.IMMINENT
    if window.imminent_minutes < minutes <= window.pre_minutes:
        return RiskWindowPhase.PRE_EVENT
    if -window.post_minutes <= minutes < 0:
        return RiskWindowPhase.POST_EVENT
    if -(window.post_minutes + window.cooldown_minutes) <= minutes < -window.post_minutes:
        return RiskWindowPhase.COOLDOWN
    return RiskWindowPhase.OUTSIDE


def clusters(events: tuple[EconomicEvent, ...], config: EconomicCalendarConfig) -> tuple[EventCluster, ...]:
    scheduled = [item for item in sorted(events, key=event_order) if item.scheduled_at_utc and not item.is_cancelled]
    groups: list[list[EconomicEvent]] = []
    for item in scheduled:
        if groups and (item.scheduled_at_utc - groups[-1][-1].scheduled_at_utc).total_seconds() <= config.cluster_window_minutes * 60:  # type: ignore[operator]
            groups[-1].append(item)
        else:
            groups.append([item])
    result = []
    for group in groups:
        if len(group) < 2:
            continue
        start, end = group[0].scheduled_at_utc, group[-1].scheduled_at_utc
        assert start is not None and end is not None
        high = sum(item.importance == EventImportance.HIGH for item in group)
        critical = sum(item.importance == EventImportance.CRITICAL for item in group)
        score = clamp((len(group) + high + critical * 2) / 10)
        result.append(
            EventCluster(
                cluster_id=stable_id("cluster", start.isoformat(), *(item.event_id for item in group)),
                window_start=start,
                window_end=end,
                event_ids=tuple(item.event_id for item in group),
                countries=tuple(sorted({item.country_code for item in group if item.country_code})),
                currencies=tuple(sorted({code for item in group for code in item.currency_codes})),
                categories=tuple(sorted({item.category for item in group}, key=str)),
                event_count=len(group),
                high_importance_count=high,
                critical_importance_count=critical,
                cluster_density=clamp(len(group) / 10),
                cluster_importance=score,
                overlap_score=score,
                conflict_score=clamp(sum(item.conflict_state != ConflictState.NONE for item in group) / len(group)),
                affected_instruments=tuple(sorted({symbol for item in group for symbol in item.affected_instruments})),
                explanation=f"{len(group)} time-adjacent events; correlation is discounted and component counts remain visible.",
            )
        )
    return tuple(result)


def instrument_context(symbol: str, snapshot: EconomicCalendarSnapshot, config: EconomicCalendarConfig) -> InstrumentEventContext:
    currencies = symbol_currencies(symbol, config)
    relevant = tuple(item for item in snapshot.events if set(item.currency_codes) & set(currencies) or symbol.upper() in item.affected_instruments)
    previous = [item for item in relevant if item.scheduled_at_utc and item.scheduled_at_utc <= snapshot.historical_boundary]
    upcoming = [item for item in relevant if item.scheduled_at_utc and item.scheduled_at_utc > snapshot.historical_boundary and not item.is_cancelled]
    phases = [(item, _phase(item, snapshot.historical_boundary, config)) for item in relevant]
    active = tuple(item for item, phase in phases if phase not in {RiskWindowPhase.OUTSIDE, RiskWindowPhase.COOLDOWN, RiskWindowPhase.UNKNOWN})
    phase_values = [phase for _, phase in phases if phase != RiskWindowPhase.OUTSIDE]
    phase = RiskWindowPhase.OVERLAPPING if len(active) > 1 else phase_values[0] if phase_values else RiskWindowPhase.OUTSIDE
    nearest = active[0] if active else upcoming[0] if upcoming else previous[-1] if previous else None
    window = config.windows[nearest.importance.value] if nearest else config.windows["unknown"]
    importance = max((config.importance_weights[item.importance.value] for item in active or relevant), default=0)
    relevance = 1.0 if relevant else 0
    cluster_score = max((item.cluster_importance for item in clusters(relevant, config)), default=0)
    conflict = sum(item.conflict_state != ConflictState.NONE for item in active)
    timing = {
        RiskWindowPhase.AT_EVENT: 1.0,
        RiskWindowPhase.IMMINENT: 0.9,
        RiskWindowPhase.OVERLAPPING: 1.0,
        RiskWindowPhase.PRE_EVENT: 0.7,
        RiskWindowPhase.POST_EVENT: 0.6,
        RiskWindowPhase.COOLDOWN: 0.3,
    }.get(phase, 0)
    risk = clamp(0.4 * importance + 0.25 * relevance + 0.2 * timing + 0.15 * cluster_score - 0.1 * conflict)
    # The single categorical state everything downstream (signal_decision_engine, explainability,
    # market_intelligence, diagnostics) reads instead of each re-deriving its own notion of
    # "available." Genuine provider/data unavailability always wins over phase/relevance — a
    # symbol can't be meaningfully "outside its risk window" if we don't actually know what the
    # calendar looks like right now.
    if snapshot.degradation.is_degraded:
        # `snapshot.degradation.category` is already one of `CalendarContextState`'s genuine-
        # failure values (set in `build_snapshot`) — "healthy" is the only non-member default,
        # which only occurs if `is_degraded` and `category` disagree (defensive fallback only).
        context_state = CalendarContextState(snapshot.degradation.category) if snapshot.degradation.category != "healthy" else CalendarContextState.NO_CALENDAR_DATA
    elif snapshot.freshness in {FreshnessState.STALE, FreshnessState.CRITICAL, FreshnessState.UNKNOWN}:
        context_state = CalendarContextState.NO_CALENDAR_DATA
    elif phase not in {RiskWindowPhase.OUTSIDE, RiskWindowPhase.COOLDOWN, RiskWindowPhase.UNKNOWN}:
        context_state = CalendarContextState.INSIDE_RISK_WINDOW
    elif not relevant:
        context_state = CalendarContextState.NO_RELEVANT_EVENTS
    else:
        context_state = CalendarContextState.OUTSIDE_RISK_WINDOW
    if context_state in GENUINELY_UNAVAILABLE_STATES:
        unavailable_reason = snapshot.degradation.reasons[0] if snapshot.degradation.reasons else f"calendar data freshness is {snapshot.freshness.value}"
        unavailable_context: tuple[str, ...] = (unavailable_reason,)
    else:
        unavailable_context = ()
    return InstrumentEventContext(
        context_id=stable_id("context", symbol.upper(), snapshot.snapshot_id),
        symbol=symbol.upper(),
        base_currency=currencies[0] if len(currencies) > 1 else None,
        quote_currency=currencies[-1] if currencies else None,
        analysis_timestamp=snapshot.analysis_timestamp,
        historical_boundary=snapshot.historical_boundary,
        previous_relevant_event=previous[-1] if previous else None,
        next_relevant_event=upcoming[0] if upcoming else None,
        active_relevant_events=active,
        minutes_since_previous_event=(snapshot.historical_boundary - previous[-1].scheduled_at_utc).total_seconds() / 60 if previous else None,  # type: ignore[operator]
        minutes_until_next_event=(upcoming[0].scheduled_at_utc - snapshot.historical_boundary).total_seconds() / 60 if upcoming else None,  # type: ignore[operator]
        risk_window_phase=phase,
        risk_score=risk,
        cluster_score=cluster_score,
        importance_score=importance,
        relevance_score=relevance,
        freshness_score={
            FreshnessState.FRESH: 1,
            FreshnessState.AGING: 0.75,
            FreshnessState.STALE: 0.4,
            FreshnessState.CRITICAL: 0.1,
            FreshnessState.UNKNOWN: 0,
        }[snapshot.freshness],
        quality_score=snapshot.quality,
        pre_event_window_minutes=window.pre_minutes,
        post_event_window_minutes=window.post_minutes,
        cooldown_window_minutes=window.cooldown_minutes,
        direct_currency_matches=tuple(sorted(set(currencies) & {code for item in relevant for code in item.currency_codes})),
        conflicting_events=tuple(item.event_id for item in relevant if item.conflict_state != ConflictState.NONE),
        # "No relevant event right now" / "outside the risk window" are routine, expected states
        # most of the time — they must NOT be conflated with genuine unavailability (the provider
        # being unreachable, or the calendar sync being stale/never-synced), which is what
        # fail-closed trading logic (signal_decision_engine's economic-event rule) actually needs
        # to react to. `context_state` (computed above) is the one place this distinction is made;
        # `unavailable_context` is just its human-readable projection for older/display consumers.
        context_state=context_state,
        unavailable_context=unavailable_context,
        primary_explanation=f"Calendar context is {phase.value} with bounded risk score {risk:.3f}; it is probabilistic context, not a trading instruction.",
        limitations=("importance is provider/config context, not guaranteed market impact",),
    )


def staged_diagnostics(snapshot: EconomicCalendarSnapshot, context: InstrumentEventContext) -> dict[str, Any]:
    """Break `degraded`/`unavailable_context` into the five independent stages of the calendar
    pipeline for display, so "0 relevant events right now" (routine, expected most of the time) is
    visibly distinguishable from "the provider is actually unreachable" (a real failure) — see
    `instrument_context()` for the fix that decoupled these in `unavailable_context` itself."""
    reachable_providers = [item for item in snapshot.provider_status if item.enabled and item.reachable]
    mapped_count = snapshot.event_count - snapshot.unavailable_count
    return {
        "provider_health": {
            "status": "healthy" if reachable_providers else "unavailable",
            "reachable_providers": [item.provider_name for item in reachable_providers],
            "providers": [item.model_dump(mode="json") for item in snapshot.provider_status],
        },
        "downloaded_events": {
            "status": "ok" if snapshot.event_count > 0 else "empty",
            "count": snapshot.event_count,
            "window_start": snapshot.window_start,
            "window_end": snapshot.window_end,
        },
        "mapped_events": {
            "status": "ok" if snapshot.event_count == 0 or mapped_count > 0 else "degraded",
            "mapped_count": mapped_count,
            "unmapped_count": snapshot.unavailable_count,
        },
        "relevant_events": {
            # `relevance_score` (not `unavailable_context`) is the correct signal here — it's
            # purely "did any event match this symbol's currencies," independent of whether the
            # calendar sync itself is healthy.
            "status": "available" if context.relevance_score > 0 else "none_relevant",
            "symbol": context.symbol,
            "active_count": len(context.active_relevant_events),
            "has_previous_event": context.previous_relevant_event is not None,
            "has_next_event": context.next_relevant_event is not None,
        },
        "trading_context": {
            "status": "ready" if not context.unavailable_context else "unavailable",
            # The exact categorical state (see `CalendarContextState`) — the same value
            # signal_decision_engine and the explainability layer read, so the dashboard can
            # distinguish e.g. "outside_risk_window" from "provider_rate_limited" instead of a
            # collapsed ready/unavailable boolean.
            "context_state": context.context_state.value,
            "risk_window_phase": context.risk_window_phase.value,
            "risk_score": context.risk_score,
            "reason": context.unavailable_context[0] if context.unavailable_context else None,
        },
    }


def surprise(event: EconomicEvent, config: EconomicCalendarConfig) -> dict[str, Any]:
    if event.actual_value is None or event.forecast_value is None or event.value_type.value in {"text", "not_applicable", "unknown"}:
        return {"available": False, "raw": None, "normalized": None, "direction": "unavailable", "trading_instruction": False}
    raw = event.actual_value - event.forecast_value
    denominator = abs(event.forecast_value)
    normalized = raw / denominator if denominator else None
    direction = "positive" if raw > 0 else "negative" if raw < 0 else "neutral"
    if event.category.value in config.inverted_surprise_categories and direction != "neutral":
        direction = "negative" if direction == "positive" else "positive"
    return {"available": True, "raw": raw, "normalized": normalized, "direction": direction, "asset_direction": None, "trading_instruction": False}


def explain(context: InstrumentEventContext) -> EconomicCalendarExplanation:
    relevant = tuple(item.event_id for item in context.active_relevant_events)
    return EconomicCalendarExplanation(
        headline=f"Economic calendar context for {context.symbol}",
        summary=context.primary_explanation,
        relevant_events=relevant,
        conflicting_events=context.conflicting_events,
        mapping_explanation=(f"Direct currency matches: {', '.join(context.direct_currency_matches) or 'none'}",),
        risk_components={
            "risk": context.risk_score,
            "importance": context.importance_score,
            "relevance": context.relevance_score,
            "cluster": context.cluster_score,
        },
        quality_components={"quality": context.quality_score},
        freshness_components={"freshness": context.freshness_score},
        limitations=context.limitations,
    )
