from __future__ import annotations

import asyncio
import csv
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .models import ProviderCapabilities, ProviderEventObservation, ProviderMode, ProviderStatus, payload_hash, stable_id


def provider_api_key(environment_name: str) -> str:
    return os.getenv(environment_name, "")


class ProviderFetchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    start: datetime
    end: datetime
    countries: tuple[str, ...] = ()
    currencies: tuple[str, ...] = ()
    importance: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    cursor: str | None = None
    limit: int = Field(default=1000, ge=1, le=10000)


class ProviderFetchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    observations: tuple[ProviderEventObservation, ...]
    cursor: str | None = None
    update_token: str | None = None
    success_count: int = 0
    failure_count: int = 0
    warnings: tuple[str, ...] = ()


class EconomicCalendarProvider(ABC):
    name: str
    version: str
    timezone: str
    mode: ProviderMode
    capabilities: ProviderCapabilities

    @abstractmethod
    async def fetch_events(self, request: ProviderFetchRequest) -> ProviderFetchResult: ...

    async def fetch_event(self, provider_event_id: str) -> ProviderEventObservation | None:
        return None

    async def fetch_updates(self, request: ProviderFetchRequest, update_token: str | None) -> ProviderFetchResult:
        return await self.fetch_events(request)

    @abstractmethod
    async def health(self) -> ProviderStatus: ...


def observation_from_mapping(provider: str, version: str, row: Mapping[str, Any], now: datetime, index: int = 0) -> ProviderEventObservation:
    provider_id = str(row.get("provider_event_id") or row.get("id") or stable_id("provider-row", provider, index, payload_hash(row)))
    response_at = _datetime(row.get("response_received_at")) or now
    published = _datetime(row.get("provider_published_at") or row.get("published_at"))
    available = _datetime(row.get("available_at")) or published or response_at
    ingested = _datetime(row.get("ingested_at")) or now
    sanitized = {str(key): value for key, value in row.items() if str(key).lower() not in {"authorization", "api_key", "token", "password", "secret"}}
    raw_name = str(row.get("raw_name") or row.get("name") or "").strip()
    if not raw_name:
        raise ValueError("provider observation requires a name")
    return ProviderEventObservation(
        observation_id=stable_id("observation", provider, provider_id, available.isoformat(), payload_hash(sanitized)),
        provider_name=provider,
        provider_version=version,
        provider_event_id=provider_id,
        provider_event_url=_optional(row.get("provider_event_url")),
        request_id=_optional(row.get("request_id")),
        request_started_at=_datetime(row.get("request_started_at")),
        response_received_at=response_at,
        raw_name=raw_name,
        raw_category=_optional(row.get("raw_category") or row.get("category")),
        raw_country=_optional(row.get("raw_country") or row.get("country")),
        raw_currency=_optional(row.get("raw_currency") or row.get("currency")),
        raw_importance=_optional(row.get("raw_importance") or row.get("importance")),
        raw_status=_optional(row.get("raw_status") or row.get("status")),
        raw_scheduled_time=_optional(row.get("raw_scheduled_time") or row.get("scheduled_at")),
        raw_timezone=_optional(row.get("raw_timezone") or row.get("timezone")),
        raw_actual=row.get("raw_actual", row.get("actual")),
        raw_forecast=row.get("raw_forecast", row.get("forecast")),
        raw_previous=row.get("raw_previous", row.get("previous")),
        raw_revised_previous=row.get("raw_revised_previous", row.get("revised_previous")),
        raw_unit=_optional(row.get("raw_unit") or row.get("unit")),
        provider_published_at=published,
        provider_updated_at=_datetime(row.get("provider_updated_at") or row.get("updated_at")),
        available_at=available,
        ingested_at=ingested,
        payload_hash=payload_hash(sanitized),
        raw_payload_reference=_optional(row.get("raw_payload_reference")),
        metadata={"sanitized_payload": sanitized},
    )


def _optional(value: object) -> str | None:
    return str(value) if value not in {None, ""} else None


def _datetime(value: object) -> datetime | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("provider timestamps must be timezone-aware")
        return value.astimezone(UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("provider timestamps must include a timezone")
    return parsed.astimezone(UTC)


class InMemoryProvider(EconomicCalendarProvider):
    def __init__(
        self, name: str, rows: Sequence[Mapping[str, Any]], *, version: str = "fixture-1", mode: ProviderMode = ProviderMode.IN_MEMORY_TEST_PROVIDER
    ) -> None:
        self.name = name
        self.version = version
        self.timezone = "UTC"
        self.mode = mode
        self.capabilities = ProviderCapabilities(historical_events=True, revisions=True, incremental_updates=True, publication_time=True)
        self._rows = tuple(rows)
        self._last_success: datetime | None = None

    async def fetch_events(self, request: ProviderFetchRequest) -> ProviderFetchResult:
        now = datetime.now(UTC)
        failures = 0
        values: list[ProviderEventObservation] = []
        for index, row in enumerate(self._rows):
            try:
                item = observation_from_mapping(self.name, self.version, row, now, index)
                if request.start <= item.available_at <= request.end and len(values) < request.limit:
                    values.append(item)
            except (TypeError, ValueError):
                failures += 1
        self._last_success = now
        return ProviderFetchResult(observations=tuple(values), cursor=str(len(values)), success_count=len(values), failure_count=failures)

    async def fetch_event(self, provider_event_id: str) -> ProviderEventObservation | None:
        now = datetime.now(UTC)
        for index, row in enumerate(self._rows):
            try:
                item = observation_from_mapping(self.name, self.version, row, now, index)
            except (TypeError, ValueError):
                continue
            if item.provider_event_id == provider_event_id:
                return item
        return None

    async def health(self) -> ProviderStatus:
        return ProviderStatus(
            provider_name=self.name,
            mode=self.mode,
            enabled=True,
            authenticated=True,
            reachable=True,
            last_success=self._last_success,
            capabilities=self.capabilities,
        )


class FileImportProvider(InMemoryProvider):
    def __init__(self, name: str, path: Path, *, import_root: Path, version: str = "file-1") -> None:
        resolved = path.resolve()
        root = import_root.resolve()
        if root not in resolved.parents or resolved.suffix.lower() not in {".json", ".csv"}:
            raise ValueError("file import path or type is not allowed")
        if resolved.stat().st_size > 10_000_000:
            raise ValueError("file import exceeds size limit")
        if resolved.suffix.lower() == ".json":
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else payload.get("events", [])
        else:
            with resolved.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("file import must contain event objects")
        super().__init__(name, rows, version=version, mode=ProviderMode.FILE_IMPORT)


class DisabledProvider(EconomicCalendarProvider):
    def __init__(self, name: str = "disabled", version: str = "1") -> None:
        self.name, self.version, self.timezone, self.mode = name, version, "UTC", ProviderMode.DISABLED
        self.capabilities = ProviderCapabilities(
            future_events=False,
            actual_values=False,
            forecast_values=False,
            previous_values=False,
            statuses=False,
            importance=False,
            country=False,
            currency=False,
            unit=False,
        )

    async def fetch_events(self, request: ProviderFetchRequest) -> ProviderFetchResult:
        await asyncio.sleep(0)
        return ProviderFetchResult(observations=(), warnings=("provider is disabled",))

    async def health(self) -> ProviderStatus:
        return ProviderStatus(
            provider_name=self.name,
            mode=self.mode,
            enabled=False,
            message="No live provider is configured; safe degraded mode is active.",
            capabilities=self.capabilities,
        )


class HttpEconomicCalendarProvider(EconomicCalendarProvider):
    """Base for HTTP-backed live providers; concrete subclasses parse the vendor payload."""

    def __init__(self, name: str, *, api_key: str, base_url: str, timeout_seconds: float = 10, version: str = "1") -> None:
        if not api_key:
            raise ValueError(f"{name} economic calendar provider requires an API key")
        self.name = name
        self.version = version
        self.timezone = "UTC"
        self.mode = ProviderMode.LIVE_PROVIDER
        self.capabilities = ProviderCapabilities(
            historical_events=True, future_events=True, actual_values=True, forecast_values=True,
            previous_values=True, statuses=True, importance=True, country=True, currency=True, unit=True,
        )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_seconds)
        self.last_latency_ms = 0.0
        self._last_success: datetime | None = None
        self._last_failure: datetime | None = None
        self._last_error: str | None = None

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> ProviderStatus:
        return ProviderStatus(
            provider_name=self.name,
            mode=self.mode,
            enabled=True,
            authenticated=bool(self.api_key),
            reachable=self._last_error is None,
            last_success=self._last_success,
            last_failure=self._last_failure,
            capabilities=self.capabilities,
            message=self._last_error or "",
        )


class FinnhubEconomicCalendarProvider(HttpEconomicCalendarProvider):
    """Live adapter for Finnhub's `/calendar/economic` endpoint."""

    def __init__(self, *, api_key: str, base_url: str = "https://finnhub.io/api/v1", timeout_seconds: float = 10, version: str = "1") -> None:
        super().__init__("finnhub", api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds, version=version)

    async def fetch_events(self, request: ProviderFetchRequest) -> ProviderFetchResult:
        now = datetime.now(UTC)
        started = perf_counter()
        try:
            response = await self._client.get(
                f"{self.base_url}/calendar/economic",
                params={"from": request.start.date().isoformat(), "to": request.end.date().isoformat(), "token": self.api_key},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            self._last_failure = now
            self._last_error = type(exc).__name__
            return ProviderFetchResult(observations=(), warnings=(f"finnhub_request_failed:{type(exc).__name__}",))
        finally:
            self.last_latency_ms = (perf_counter() - started) * 1000
        rows = payload.get("economicCalendar") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            self._last_failure = now
            self._last_error = "unexpected_response_shape"
            return ProviderFetchResult(observations=(), warnings=("finnhub_unexpected_response_shape",))
        values: list[ProviderEventObservation] = []
        failures = 0
        for index, row in enumerate(rows):
            try:
                item = observation_from_mapping(self.name, self.version, self._map_row(row), now, index)
                if request.start <= item.available_at <= request.end and len(values) < request.limit:
                    values.append(item)
            except (TypeError, ValueError, KeyError):
                failures += 1
        self._last_success = now
        self._last_error = None
        return ProviderFetchResult(observations=tuple(values), cursor=str(len(values)), success_count=len(values), failure_count=failures)

    @staticmethod
    def _map_row(row: Mapping[str, Any]) -> dict[str, Any]:
        event = str(row.get("event") or "").strip()
        country = str(row.get("country") or "").strip().upper()
        scheduled = str(row.get("time") or "").strip()
        return {
            "provider_event_id": stable_id("finnhub-event", country, event, scheduled, row.get("unit")),
            "raw_name": event,
            "raw_country": country or None,
            "raw_importance": row.get("impact"),
            "raw_status": "released" if row.get("actual") not in (None, "") else "scheduled",
            "raw_scheduled_time": scheduled or None,
            "raw_timezone": "UTC",
            "raw_actual": row.get("actual"),
            "raw_forecast": row.get("estimate"),
            "raw_previous": row.get("prev"),
            "raw_unit": row.get("unit"),
        }


def build_providers(
    configs: Sequence[Any],
    *,
    import_root: Path = Path("data/economic_calendar_imports"),
    fixtures: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[EconomicCalendarProvider, ...]:
    """Build only explicitly configured safe adapters; live mode requires a concrete adapter."""
    result: list[EconomicCalendarProvider] = []
    fixtures = fixtures or {}
    for config in configs:
        if config.mode == ProviderMode.FILE_IMPORT and config.enabled:
            if not config.file_path:
                raise ValueError(f"file provider {config.name} requires file_path")
            result.append(FileImportProvider(config.name, Path(config.file_path), import_root=import_root, version=config.version))
        elif config.mode in {ProviderMode.STATIC_FIXTURE, ProviderMode.IN_MEMORY_TEST_PROVIDER} and config.enabled:
            result.append(InMemoryProvider(config.name, fixtures.get(config.name, ()), version=config.version, mode=config.mode))
        elif config.mode == ProviderMode.LIVE_PROVIDER and config.enabled and config.name == "finnhub":
            key = provider_api_key(config.api_key_env or "TEN_FINNHUB_API_KEY")
            if key:
                result.append(FinnhubEconomicCalendarProvider(api_key=key, base_url=config.base_url or "https://finnhub.io/api/v1", timeout_seconds=config.timeout_seconds, version=config.version))
            else:
                result.append(DisabledProvider(config.name, config.version))
        else:
            result.append(DisabledProvider(config.name, config.version))
    return tuple(result) or (DisabledProvider(),)
