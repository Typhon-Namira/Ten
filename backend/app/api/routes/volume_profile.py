from datetime import datetime
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.dependencies import get_volume_profile_service
from backend.app.engines.market_data_engine import Timeframe
from backend.app.engines.volume_profile_engine import ProcessingMode, ProfileType, VolumeProfileAnalysisSnapshot, VolumeProfileService

router = APIRouter(prefix="/volume-profile", tags=["volume-profile"])
Service = Annotated[VolumeProfileService, Depends(get_volume_profile_service)]


async def _snapshot(service: VolumeProfileService, symbol: str, timeframe: Timeframe, timestamp: datetime | None, limit: int) -> VolumeProfileAnalysisSnapshot:
    existing = await service.state(symbol, timeframe, timestamp)
    if existing:
        return existing
    try:
        return await service.replay(symbol, timeframe, timestamp, limit) if timestamp else await service.analyze(symbol, timeframe, limit=limit)
    except Exception as exc:
        raise HTTPException(503, "Volume Profile source data is unavailable") from exc


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


@router.get("/snapshot", response_model=VolumeProfileAnalysisSnapshot)
async def snapshot(
    service: Service,
    symbol: str = "XAUUSD",
    timeframe: Timeframe = Timeframe.M15,
    timestamp: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100000)] = 500,
) -> VolumeProfileAnalysisSnapshot:
    return await _snapshot(service, symbol, timeframe, timestamp, limit)


@router.get("/state")
async def state(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None) -> object:
    return await _snapshot(service, symbol, timeframe, timestamp, 500)


@router.get("/replay", response_model=VolumeProfileAnalysisSnapshot)
async def replay(
    service: Service, timestamp: datetime, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, limit: Annotated[int, Query(ge=1, le=100000)] = 500
) -> VolumeProfileAnalysisSnapshot:
    return await service.replay(symbol, timeframe, timestamp, limit)


@router.get("/fixed-range")
async def fixed_range(
    service: Service,
    start: datetime,
    end: datetime,
    symbol: str = "XAUUSD",
    timeframe: Timeframe = Timeframe.M15,
    limit: Annotated[int, Query(ge=1, le=100000)] = 5000,
) -> list[object]:
    if end <= start:
        raise HTTPException(422, "end must be later than start")
    if (end - start).days > service.config.processing.maximum_range_days:
        raise HTTPException(422, "fixed range exceeds configured maximum")
    snapshot = await service.analyze(symbol, timeframe, start=start, end=end, limit=limit, mode=ProcessingMode.FIXED_RANGE)
    return [x for x in snapshot.profiles if x.profile_type == ProfileType.FIXED_RANGE]


def _value(item: object, name: str) -> object | None:
    value = getattr(item, name, None)
    return cast(object, getattr(value, "value", value)) if value is not None else None


def collection(path: str, attribute: str, profile_type: str | None = None) -> None:
    async def endpoint(
        service: Service,
        symbol: str = "XAUUSD",
        timeframe: Timeframe = Timeframe.M15,
        timestamp: datetime | None = None,
        requested_profile_type: Annotated[str | None, Query(alias="profile_type")] = None,
        lifecycle_state: str | None = None,
        anchor_type: str | None = None,
        session: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        tested: bool | None = None,
        min_confidence: Annotated[float, Query(ge=0, le=100)] = 0,
        min_quality: Annotated[float, Query(ge=0, le=100)] = 0,
        completed_only: bool = False,
        developing_only: bool = False,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=5000)] = 500,
    ) -> list[object]:
        snapshot = await _snapshot(service, symbol, timeframe, timestamp, min(100000, offset + limit))
        values = list(getattr(snapshot, attribute))
        if profile_type:
            values = [x for x in values if _value(x, "profile_type") == profile_type]
        if requested_profile_type:
            values = [x for x in values if _value(x, "profile_type") == requested_profile_type]
        if lifecycle_state:
            values = [x for x in values if _value(x, "lifecycle_state") == lifecycle_state]
        if anchor_type:
            values = [x for x in values if _value(getattr(x, "anchor", None), "anchor_type") == anchor_type]
        if session:
            values = [x for x in values if _value(x, "session") == session]
        if start:
            values = [x for x in values if getattr(x, "start_timestamp", snapshot.analysis_timestamp) >= start]
        if end:
            values = [x for x in values if getattr(x, "end_timestamp", snapshot.analysis_timestamp) <= end]
        if tested is not None:
            values = [x for x in values if bool(getattr(getattr(x, "poc", None), "tested", False)) == tested]
        values = [x for x in values if float(getattr(x, "confidence_score", 100)) >= min_confidence and float(getattr(x, "quality_score", 100)) >= min_quality]
        if completed_only:
            values = [x for x in values if _value(x, "status") == "completed"]
        if developing_only:
            values = [x for x in values if _value(x, "status") == "developing"]
        return values[offset : offset + limit]

    endpoint.__name__ = "volume_profile_" + path.replace("/", "_").strip("_")
    router.add_api_route(path, endpoint, methods=["GET"])


for _path, _attr, _type in (
    ("/profiles", "profiles", None),
    ("/developing", "developing", None),
    ("/completed", "completed", None),
    ("/sessions", "profiles", "session"),
    ("/daily", "profiles", "daily"),
    ("/weekly", "profiles", "weekly"),
    ("/monthly", "profiles", "monthly"),
    ("/composite", "profiles", "composite"),
    ("/anchored", "profiles", "anchored"),
    ("/migrations", "migrations", None),
    ("/confluences", "confluences", None),
):
    collection(_path, _attr, _type)


def derived(path: str, name: str) -> None:
    async def endpoint(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None) -> list[object]:
        snapshot = await _snapshot(service, symbol, timeframe, timestamp, 500)
        if name in {"poc", "value_area", "shape"}:
            return [value for profile in snapshot.profiles if (value := getattr(profile, name)) is not None]
        return [value for profile in snapshot.profiles for value in getattr(profile, name)]

    endpoint.__name__ = "volume_profile_derived_" + name
    router.add_api_route(path, endpoint, methods=["GET"])


for _path, _name in (
    ("/poc", "poc"),
    ("/value-area", "value_area"),
    ("/hvn", "hvns"),
    ("/lvn", "lvns"),
    ("/shelves", "shelves"),
    ("/gaps", "gaps"),
    ("/shapes", "shape"),
):
    derived(_path, _name)


@router.get("/mtf")
async def mtf(service: Service, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M15, timestamp: datetime | None = None) -> object:
    return await service.multi_timeframe(symbol, timeframe, timestamp)
