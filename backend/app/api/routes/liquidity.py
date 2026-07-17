from datetime import datetime
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.dependencies import get_liquidity_service
from backend.app.engines.liquidity_engine import LiquidityAnalysisSnapshot, LiquidityService
from backend.app.engines.market_data_engine import Timeframe

router = APIRouter(prefix="/liquidity", tags=["liquidity"])
Service = Annotated[LiquidityService, Depends(get_liquidity_service)]


async def _snapshot(service: LiquidityService, symbol: str, timeframe: Timeframe, timestamp: datetime | None, limit: int) -> LiquidityAnalysisSnapshot:
    existing = await service.state(symbol, timeframe, timestamp)
    if existing:
        return existing
    try:
        return await service.replay(symbol, timeframe, timestamp, limit) if timestamp else await service.analyze(symbol, timeframe, limit=limit)
    except Exception as exc:
        raise HTTPException(503, "Liquidity source data is unavailable") from exc


@router.get("/health")
async def health(service: Service) -> dict[str, object]:
    return service.health()


@router.get("/metrics")
async def metrics(service: Service) -> dict[str, int | float | str | None]:
    return service.metrics.snapshot()


@router.get("/config")
async def config(service: Service) -> dict[str, object]:
    return {
        "configuration": service.config.model_dump(mode="json"),
        "configuration_version": service.config.version,
        "engine_version": service.analyzer.version,
    }


@router.get("/snapshot", response_model=LiquidityAnalysisSnapshot)
async def snapshot(
    service: Service,
    symbol: str = "XAUUSD",
    timeframe: Timeframe = Timeframe.M15,
    timestamp: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
) -> LiquidityAnalysisSnapshot:
    return await _snapshot(service, symbol, timeframe, timestamp, limit)


@router.get("/state")
async def state(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None) -> object:
    return (await _snapshot(service, symbol, timeframe, timestamp, 500)).state


@router.get("/replay", response_model=LiquidityAnalysisSnapshot)
async def replay(
    service: Service, timestamp: datetime, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, limit: Annotated[int, Query(ge=1, le=5000)] = 500
) -> LiquidityAnalysisSnapshot:
    return await service.replay(symbol, timeframe, timestamp, limit)


def _value(item: object, *names: str) -> object | None:
    for name in names:
        value = getattr(item, name, None)
        if value is not None:
            return cast(object, getattr(value, "value", value))
    return None


def _score(item: object, name: str) -> float:
    value = _value(item, name)
    return float(value) if isinstance(value, int | float) else 0.0


def _items(
    snapshot: LiquidityAnalysisSnapshot,
    name: str,
    offset: int,
    limit: int,
    *,
    side: str | None = None,
    source: str | None = None,
    scope: str | None = None,
    object_type: str | None = None,
    lifecycle_state: str | None = None,
    session: str | None = None,
    min_confidence: float = 0,
    min_strength: float = 0,
    active_only: bool = False,
) -> list[object]:
    active_states = {"active", "approached", "touched", "partially_swept", "reclaimed"}
    items = list(getattr(snapshot, name))
    filters = (
        (("side",), side),
        (("source",), source),
        (("scope",), scope),
        (("level_type", "pool_type", "event_type", "classification", "reference_type", "status"), object_type),
        (("lifecycle_state",), lifecycle_state),
        (("session",), session),
    )
    for names, expected in filters:
        if expected is not None:
            items = [item for item in items if str(_value(item, *names)) == expected]
    items = [item for item in items if _score(item, "confidence_score") >= min_confidence]
    items = [item for item in items if _score(item, "strength_score") >= min_strength]
    if active_only:
        items = [item for item in items if _value(item, "lifecycle_state") in active_states]
    return items[offset : offset + limit]


def collection(path: str, attribute: str) -> None:
    async def endpoint(
        service: Service,
        symbol: str = "XAUUSD",
        timeframe: Timeframe = Timeframe.M15,
        timestamp: datetime | None = None,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=5000)] = 500,
        side: str | None = None,
        source: str | None = None,
        scope: str | None = None,
        object_type: Annotated[str | None, Query(alias="type")] = None,
        lifecycle_state: str | None = None,
        session: str | None = None,
        min_confidence: Annotated[float, Query(ge=0, le=100)] = 0,
        min_strength: Annotated[float, Query(ge=0, le=100)] = 0,
        active_only: bool = False,
    ) -> list[object]:
        snapshot = await _snapshot(service, symbol, timeframe, timestamp, min(5000, offset + limit))
        return _items(
            snapshot,
            attribute,
            offset,
            limit,
            side=side,
            source=source,
            scope=scope,
            object_type=object_type,
            lifecycle_state=lifecycle_state,
            session=session,
            min_confidence=min_confidence,
            min_strength=min_strength,
            active_only=active_only,
        )

    endpoint.__name__ = f"liquidity_{attribute}"
    router.add_api_route(path, endpoint, methods=["GET"])


for _path, _attribute in (
    ("/levels", "levels"),
    ("/equal-levels", "equal_levels"),
    ("/pools", "pools"),
    ("/events", "events"),
    ("/sweeps", "sweeps"),
    ("/grabs", "grabs"),
    ("/raids", "raids"),
    ("/stop-hunts", "stop_hunts"),
    ("/false-breaks", "false_breaks"),
    ("/sessions", "sessions"),
    ("/reference-levels", "reference_levels"),
    ("/confluences", "confluences"),
    ("/targets", "targets"),
    ("/map", "map_bands"),
):
    collection(_path, _attribute)


@router.get("/mtf")
async def mtf(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None) -> object:
    return await service.multi_timeframe(symbol, timeframe, timestamp)
