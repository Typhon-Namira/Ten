from datetime import datetime, timedelta
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.dependencies import get_economic_calendar_service
from backend.app.engines.economic_calendar_engine import EconomicCalendarService
from backend.app.engines.economic_calendar_engine.analyzer import staged_diagnostics

router = APIRouter(prefix="/economic-calendar", tags=["economic-calendar"])
Service = Annotated[EconomicCalendarService, Depends(get_economic_calendar_service)]


def _aware(value: datetime | None, name: str) -> datetime | None:
    if value is not None and value.tzinfo is None:
        raise HTTPException(422, f"{name} must include a timezone")
    return value


def _window(
    service: EconomicCalendarService, start: datetime | None, end: datetime | None, as_of: datetime | None = None
) -> tuple[datetime, datetime, datetime]:
    boundary = _aware(as_of, "as_of") or service.clock.now()
    resolved_start = _aware(start, "start") or boundary - timedelta(days=2)
    resolved_end = _aware(end, "end") or boundary + timedelta(days=7)
    if resolved_start >= resolved_end:
        raise HTTPException(422, "start must precede end")
    if resolved_end - resolved_start > timedelta(days=service.config.api.maximum_query_window_days):
        raise HTTPException(422, "requested calendar window exceeds the configured bound")
    return resolved_start, resolved_end, boundary


@router.get("/health")
async def health(service: Service) -> dict[str, object]:
    result = service.health()
    result["provider_status"] = [item.model_dump(mode="json") for item in await service.provider_status()]
    return result


@router.get("/config")
async def config(service: Service) -> dict[str, object]:
    value = service.config.model_dump(mode="json")
    for provider in value["providers"]:
        provider.pop("file_path", None)
    return {"configuration": value, "configuration_version": service.config.version, "engine_version": service.engine.version}


@router.get("/metrics")
async def metrics(service: Service) -> dict[str, int | float | str | None]:
    return service.metrics.snapshot()


@router.get("/providers")
async def providers(service: Service) -> object:
    return await service.provider_status()


@router.get("/diagnostics")
async def diagnostics(service: Service, symbol: str = "XAUUSD", as_of: datetime | None = None) -> dict[str, object]:
    """Provider health -> downloaded events -> mapped events -> relevant events -> trading
    context, each reported independently, so "no relevant events right now" (routine) is never
    confused with "the provider is actually down" (a real failure) — both used to collapse into
    the same `degraded`/`unavailable_context` flag with no way to tell them apart."""
    boundary = _aware(as_of, "as_of") or service.clock.now()
    context = await service.context(symbol, as_of=boundary, publish=False)
    snapshot = await service.snapshot(boundary - timedelta(days=2), boundary + timedelta(days=7), as_of=boundary)
    return staged_diagnostics(snapshot, context)


@router.get("/events")
async def events(
    service: Service,
    start: datetime | None = None,
    end: datetime | None = None,
    as_of: datetime | None = None,
    country: str | None = None,
    currency: str | None = None,
    category: str | None = None,
    importance: str | None = None,
    status: str | None = None,
    provider: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> object:
    start, end, boundary = _window(service, start, end, as_of)
    if country and (len(country) != 2 or not country.isalpha()):
        raise HTTPException(422, "country must be a two-letter code")
    if currency and (len(currency) != 3 or not currency.isalpha()):
        raise HTTPException(422, "currency must be a three-letter code")
    values = list(await service.events(start, end, boundary, limit))
    if country:
        values = [item for item in values if item.country_code == country.upper()]
    if currency:
        values = [item for item in values if currency.upper() in item.currency_codes]
    if category:
        values = [item for item in values if item.category.value == category]
    if importance:
        values = [item for item in values if item.importance.value == importance]
    if status:
        values = [item for item in values if item.status.value == status]
    if provider:
        values = [item for item in values if item.metadata.get("provider") == provider]
    return values


@router.get("/events/{event_id}")
async def event(service: Service, event_id: UUID, as_of: datetime | None = None) -> object:
    value = await service.event(event_id, _aware(as_of, "as_of"))
    if value is None:
        raise HTTPException(404, "economic event not found at the requested boundary")
    return {"event": value, "historical_boundary": as_of, "reconstructed": as_of is not None, "latest_state_available": await service.event(event_id)}


@router.get("/events/{event_id}/revisions")
async def revisions(service: Service, event_id: UUID, limit: Annotated[int, Query(ge=1, le=500)] = 100) -> object:
    return await service.revisions(event_id, limit)


@router.get("/events/{event_id}/observations")
async def observations(service: Service, event_id: UUID, limit: Annotated[int, Query(ge=1, le=500)] = 100) -> object:
    return await service.observations(event_id, limit)


async def _filtered(service: EconomicCalendarService, mode: str, start: datetime | None, end: datetime | None, as_of: datetime | None, limit: int) -> object:
    start, end, boundary = _window(service, start, end, as_of)
    snapshot = await service.snapshot(start, end, as_of=boundary)
    return getattr(snapshot, f"{mode}_events")[:limit]


@router.get("/upcoming")
async def upcoming(
    service: Service, start: datetime | None = None, end: datetime | None = None, as_of: datetime | None = None, limit: Annotated[int, Query(ge=1, le=500)] = 50
) -> object:
    return await _filtered(service, "upcoming", start, end, as_of, limit)


@router.get("/recent")
async def recent(
    service: Service, start: datetime | None = None, end: datetime | None = None, as_of: datetime | None = None, limit: Annotated[int, Query(ge=1, le=500)] = 50
) -> object:
    return await _filtered(service, "recent", start, end, as_of, limit)


@router.get("/active")
async def active(
    service: Service, start: datetime | None = None, end: datetime | None = None, as_of: datetime | None = None, limit: Annotated[int, Query(ge=1, le=500)] = 50
) -> object:
    return await _filtered(service, "active", start, end, as_of, limit)


@router.get("/snapshot")
async def snapshot(service: Service, start: datetime | None = None, end: datetime | None = None, as_of: datetime | None = None) -> object:
    start, end, boundary = _window(service, start, end, as_of)
    return await service.snapshot(start, end, as_of=boundary)


@router.get("/history")
async def history(service: Service, limit: Annotated[int, Query(ge=1, le=500)] = 50) -> object:
    return await service.repository.list_snapshots(limit)


@router.get("/context")
@router.get("/context/{symbol}")
async def context(service: Service, symbol: str = "XAUUSD", as_of: datetime | None = None) -> object:
    clean = "".join(character for character in symbol.upper() if character.isalnum())
    if clean != symbol.upper() or not 3 <= len(clean) <= 20:
        raise HTTPException(422, "invalid symbol")
    return await service.context(clean, as_of=_aware(as_of, "as_of"))


@router.get("/clusters")
async def clusters(service: Service, start: datetime | None = None, end: datetime | None = None, as_of: datetime | None = None) -> object:
    start, end, boundary = _window(service, start, end, as_of)
    return await service.event_clusters(start, end, boundary)


@router.get("/conflicts")
async def conflicts(service: Service, start: datetime | None = None, end: datetime | None = None, as_of: datetime | None = None) -> object:
    start, end, boundary = _window(service, start, end, as_of)
    return [item for item in await service.events(start, end, boundary, service.config.api.maximum_page_size) if item.conflict_state.value != "none"]


@router.get("/explanations")
async def explanations(service: Service, symbol: str = "XAUUSD", as_of: datetime | None = None) -> object:
    return await service.explanation(symbol, _aware(as_of, "as_of"))
