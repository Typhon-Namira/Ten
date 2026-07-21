from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .models import ConnectionState, ProviderCapabilities, ProviderEventObservation, ProviderMode, ProviderStatus, payload_hash, stable_id

logger = logging.getLogger(__name__)


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
            provider_version=self.version,
            mode=self.mode,
            enabled=True,
            api_key_configured=True,
            authenticated=True,
            reachable=True,
            connection_state=ConnectionState.CONNECTED,
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
            provider_version=self.version,
            mode=self.mode,
            enabled=False,
            connection_state=ConnectionState.DISABLED,
            failure_reason="provider is disabled",
            message="No live provider is configured; safe degraded mode is active.",
            capabilities=self.capabilities,
        )


def _sanitized_body(response: httpx.Response, limit: int = 500) -> str:
    """The raw response body, truncated and stripped of anything that looks like a credential —
    surfaced to operators/AI explainability as `raw_error`, never logged with the API key."""
    text = response.text
    for needle in ("apikey", "api_key", "token", "key"):
        # Best-effort scrub: FMP/Finnhub echo the query string in some error bodies. This is a
        # diagnostic aid, not a security boundary — the key itself is never sent in the body.
        if needle in text.lower():
            text = "(response body withheld: may echo request parameters)"
            break
    return text[:limit]


def _int_header(response: httpx.Response, *names: str) -> int | None:
    for name in names:
        value = response.headers.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue
    return None


class HttpEconomicCalendarProvider(EconomicCalendarProvider):
    """Base for HTTP-backed live providers. Owns one shared, retrying GET helper so every live
    provider reports identical, comparable telemetry (latency, retries, backoff, rate limit,
    connection state) instead of each adapter tracking its own ad-hoc subset."""

    def __init__(
        self, name: str, *, api_key: str, base_url: str, timeout_seconds: float = 10, version: str = "1", max_retries: int = 2, retry_backoff_seconds: float = 0.5
    ) -> None:
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
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._client = httpx.AsyncClient(timeout=timeout_seconds)
        self.last_latency_ms = 0.0
        self._last_request: datetime | None = None
        self._last_success: datetime | None = None
        self._last_failure: datetime | None = None
        self._connection_state = ConnectionState.UNKNOWN
        self._failure_reason: str | None = None
        self._http_status: int | None = None
        self._retry_count = 0
        self._backoff_until: datetime | None = None
        self._rate_limit_remaining: int | None = None
        self._rate_limit_limit: int | None = None
        self._raw_error: str | None = None

    async def close(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: Mapping[str, Any]) -> httpx.Response | None:
        """GET with retry/backoff, updating every telemetry field this provider reports. Never
        raises for an HTTP-level failure — returns `None` and leaves the failure fully described
        in `self._failure_reason`/`self._raw_error`/`self._connection_state` for `health()`."""
        now = datetime.now(UTC)
        self._last_request = now
        started = perf_counter()
        attempt = 0
        last_exception: Exception | None = None
        while attempt <= self.max_retries:
            try:
                response = await self._client.get(f"{self.base_url}{path}", params=params)
            except httpx.TimeoutException as exc:
                last_exception, self._connection_state, self._failure_reason = exc, ConnectionState.TIMEOUT, "request timed out"
            except httpx.HTTPError as exc:
                last_exception, self._connection_state, self._failure_reason = exc, ConnectionState.UNREACHABLE, f"{type(exc).__name__}: {exc}"[:300]
            else:
                self.last_latency_ms = (perf_counter() - started) * 1000
                self._http_status = response.status_code
                self._rate_limit_remaining = _int_header(response, "x-ratelimit-remaining", "x-ratelimit-remaining-day")
                self._rate_limit_limit = _int_header(response, "x-ratelimit-limit", "x-ratelimit-limit-day")
                if response.status_code in (401, 403):
                    self._connection_state, self._failure_reason = ConnectionState.UNAUTHORIZED, f"HTTP {response.status_code}: authentication failed"
                    self._raw_error, self._last_failure, self._retry_count = _sanitized_body(response), now, attempt
                    # Never silently fail an auth error — log provider, endpoint (no query string,
                    # so the API key is never written to logs), status code, and the provider's own
                    # error message so this is diagnosable from logs alone.
                    logger.error("economic_calendar provider authentication failed: provider=%s endpoint=%s status=%s message=%s", self.name, path, response.status_code, self._raw_error)
                    return None
                if response.status_code == 429:
                    self._connection_state, self._failure_reason = ConnectionState.RATE_LIMITED, "HTTP 429: rate limited"
                    retry_after = response.headers.get("retry-after", "")
                    self._backoff_until = now + timedelta(seconds=float(retry_after)) if retry_after.isdigit() else now + timedelta(seconds=60)
                    self._raw_error, self._last_failure, self._retry_count = _sanitized_body(response), now, attempt
                    logger.warning("economic_calendar provider rate limited: provider=%s endpoint=%s status=%s backoff_until=%s", self.name, path, response.status_code, self._backoff_until)
                    return None
                if response.status_code >= 400:
                    retryable = response.status_code >= 500
                    self._connection_state, self._failure_reason = ConnectionState.UNREACHABLE, f"HTTP {response.status_code}"
                    self._raw_error = _sanitized_body(response)
                    if retryable and attempt < self.max_retries:
                        attempt += 1
                        await asyncio.sleep(self.retry_backoff_seconds * attempt)
                        continue
                    self._last_failure, self._retry_count = now, attempt
                    logger.warning("economic_calendar provider request failed: provider=%s endpoint=%s status=%s message=%s", self.name, path, response.status_code, self._raw_error)
                    return None
                self._connection_state, self._failure_reason, self._raw_error, self._backoff_until = ConnectionState.CONNECTED, None, None, None
                self._last_success, self._retry_count = now, attempt
                return response
            attempt += 1
            if attempt <= self.max_retries:
                await asyncio.sleep(self.retry_backoff_seconds * attempt)
        self.last_latency_ms = (perf_counter() - started) * 1000
        self._raw_error = f"{type(last_exception).__name__}: {last_exception}"[:300] if last_exception else self._failure_reason
        logger.warning("economic_calendar provider unreachable: provider=%s endpoint=%s connection_state=%s message=%s", self.name, path, self._connection_state.value, self._raw_error)
        self._last_failure, self._retry_count = now, attempt
        return None

    async def health(self) -> ProviderStatus:
        return ProviderStatus(
            provider_name=self.name,
            provider_version=self.version,
            base_url=self.base_url,
            mode=self.mode,
            enabled=True,
            api_key_configured=bool(self.api_key),
            authenticated=self._connection_state not in {ConnectionState.UNAUTHORIZED, ConnectionState.UNKNOWN},
            reachable=self._connection_state == ConnectionState.CONNECTED,
            rate_limited=self._connection_state == ConnectionState.RATE_LIMITED,
            connection_state=self._connection_state,
            failure_reason=self._failure_reason,
            http_status=self._http_status,
            last_request=self._last_request,
            last_success=self._last_success,
            last_failure=self._last_failure,
            response_time_ms=self.last_latency_ms or None,
            retry_count=self._retry_count,
            backoff_until=self._backoff_until,
            rate_limit_remaining=self._rate_limit_remaining,
            rate_limit_limit=self._rate_limit_limit,
            raw_error=self._raw_error,
            capabilities=self.capabilities,
            message=self._failure_reason or "",
        )


class FinancialModelingPrepProvider(HttpEconomicCalendarProvider):
    """Live adapter for Financial Modeling Prep's stable `/stable/economics-calendar` endpoint —
    TEN's primary economic calendar provider.

    FMP retired its legacy `/api/v3/*` endpoints for accounts created/authorized after 2025-08-31
    (they now return `403 Forbidden` with a "Legacy Endpoint" error body) — every request here
    must go through the `/stable` base URL instead."""

    def __init__(
        self, *, api_key: str, base_url: str = "https://financialmodelingprep.com/stable", timeout_seconds: float = 10, version: str = "1", max_retries: int = 2, retry_backoff_seconds: float = 0.5
    ) -> None:
        super().__init__("financial_modeling_prep", api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds, version=version, max_retries=max_retries, retry_backoff_seconds=retry_backoff_seconds)

    async def fetch_events(self, request: ProviderFetchRequest) -> ProviderFetchResult:
        now = datetime.now(UTC)
        endpoint = "/economics-calendar"
        response = await self._get(endpoint, {"from": request.start.date().isoformat(), "to": request.end.date().isoformat(), "apikey": self.api_key})
        if response is None:
            return ProviderFetchResult(observations=(), warnings=(f"fmp_request_failed:{self._connection_state.value}",))
        try:
            payload = response.json()
        except ValueError:
            self._connection_state, self._failure_reason = ConnectionState.UNREACHABLE, "response was not valid JSON"
            return ProviderFetchResult(observations=(), warnings=("fmp_unexpected_response_shape",))
        if isinstance(payload, dict) and ("Error Message" in payload or "error" in payload):
            # FMP returns HTTP 200 with an error body for some invalid-key/invalid-plan/legacy-
            # endpoint cases — a real authentication/authorization failure, not "zero events
            # today". Treated identically to a 401/403: logged loudly, never silently swallowed.
            self._connection_state = ConnectionState.UNAUTHORIZED
            self._failure_reason = str(payload.get("Error Message") or payload.get("error"))[:300]
            self._raw_error = self._failure_reason
            self._last_failure = now
            logger.error("economic_calendar provider authentication failed: provider=%s endpoint=%s status=%s message=%s", self.name, endpoint, response.status_code, self._failure_reason)
            return ProviderFetchResult(observations=(), warnings=("fmp_error_response",))
        if not isinstance(payload, list):
            self._connection_state, self._failure_reason = ConnectionState.UNREACHABLE, "unexpected response shape"
            return ProviderFetchResult(observations=(), warnings=("fmp_unexpected_response_shape",))
        values: list[ProviderEventObservation] = []
        failures = 0
        for index, row in enumerate(payload):
            try:
                item = observation_from_mapping(self.name, self.version, self._map_row(row), now, index)
                if request.start <= item.available_at <= request.end and len(values) < request.limit:
                    values.append(item)
            except (TypeError, ValueError, KeyError):
                failures += 1
        return ProviderFetchResult(observations=tuple(values), cursor=str(len(values)), success_count=len(values), failure_count=failures)

    @staticmethod
    def _map_row(row: Mapping[str, Any]) -> dict[str, Any]:
        event = str(row.get("event") or "").strip()
        country = str(row.get("country") or "").strip().upper()
        currency = str(row.get("currency") or "").strip().upper()
        scheduled = str(row.get("date") or "").strip()
        return {
            "provider_event_id": stable_id("fmp-event", country, currency, event, scheduled),
            "raw_name": event,
            "raw_country": country or None,
            "raw_currency": currency or None,
            "raw_importance": row.get("impact"),
            "raw_status": "released" if row.get("actual") not in (None, "") else "scheduled",
            "raw_scheduled_time": scheduled or None,
            "raw_timezone": "UTC",
            "raw_actual": row.get("actual"),
            "raw_forecast": row.get("estimate"),
            "raw_previous": row.get("previous"),
            "raw_unit": row.get("unit"),
        }


class FinnhubEconomicCalendarProvider(HttpEconomicCalendarProvider):
    """Live adapter for Finnhub's `/calendar/economic` endpoint — optional fallback provider,
    used only when Financial Modeling Prep is unavailable."""

    def __init__(
        self, *, api_key: str, base_url: str = "https://finnhub.io/api/v1", timeout_seconds: float = 10, version: str = "1", max_retries: int = 2, retry_backoff_seconds: float = 0.5
    ) -> None:
        super().__init__("finnhub", api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds, version=version, max_retries=max_retries, retry_backoff_seconds=retry_backoff_seconds)

    async def fetch_events(self, request: ProviderFetchRequest) -> ProviderFetchResult:
        now = datetime.now(UTC)
        response = await self._get("/calendar/economic", {"from": request.start.date().isoformat(), "to": request.end.date().isoformat(), "token": self.api_key})
        if response is None:
            return ProviderFetchResult(observations=(), warnings=(f"finnhub_request_failed:{self._connection_state.value}",))
        try:
            payload = response.json()
        except ValueError:
            self._connection_state, self._failure_reason = ConnectionState.UNREACHABLE, "response was not valid JSON"
            return ProviderFetchResult(observations=(), warnings=("finnhub_unexpected_response_shape",))
        rows = payload.get("economicCalendar") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            self._connection_state, self._failure_reason = ConnectionState.UNREACHABLE, "unexpected response shape"
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


_LIVE_PROVIDER_CLASSES: dict[str, Callable[..., HttpEconomicCalendarProvider]] = {
    "financial_modeling_prep": FinancialModelingPrepProvider,
    "fmp": FinancialModelingPrepProvider,
    "finnhub": FinnhubEconomicCalendarProvider,
}
_LIVE_PROVIDER_DEFAULT_KEY_ENV = {
    "financial_modeling_prep": "TEN_FMP_API_KEY",
    "fmp": "TEN_FMP_API_KEY",
    "finnhub": "TEN_FINNHUB_API_KEY",
}
_LIVE_PROVIDER_DEFAULT_BASE_URL = {
    "financial_modeling_prep": "https://financialmodelingprep.com/stable",
    "fmp": "https://financialmodelingprep.com/stable",
    "finnhub": "https://finnhub.io/api/v1",
}


def build_providers(
    configs: Sequence[Any],
    *,
    import_root: Path = Path("data/economic_calendar_imports"),
    fixtures: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[EconomicCalendarProvider, ...]:
    """Build only explicitly configured safe adapters; live mode requires a concrete adapter.

    Providers are built in the order given in `configs` — callers must pass them in priority
    order (Financial Modeling Prep first, Finnhub as an optional fallback, then any static/disabled
    entries), since `EconomicCalendarService` queries every returned provider on each sync and
    `reconcile()` ranks conflicting data by that same order. A `live_provider` entry with no
    configured API key degrades to `DisabledProvider` rather than raising, so a missing optional
    fallback key never prevents the primary provider from starting.
    """
    result: list[EconomicCalendarProvider] = []
    fixtures = fixtures or {}
    for config in configs:
        if config.mode == ProviderMode.FILE_IMPORT and config.enabled:
            if not config.file_path:
                raise ValueError(f"file provider {config.name} requires file_path")
            result.append(FileImportProvider(config.name, Path(config.file_path), import_root=import_root, version=config.version))
        elif config.mode in {ProviderMode.STATIC_FIXTURE, ProviderMode.IN_MEMORY_TEST_PROVIDER} and config.enabled:
            result.append(InMemoryProvider(config.name, fixtures.get(config.name, ()), version=config.version, mode=config.mode))
        elif config.mode == ProviderMode.LIVE_PROVIDER and config.enabled and config.name in _LIVE_PROVIDER_CLASSES:
            provider_class = _LIVE_PROVIDER_CLASSES[config.name]
            key = provider_api_key(config.api_key_env or _LIVE_PROVIDER_DEFAULT_KEY_ENV[config.name])
            if key:
                result.append(
                    provider_class(
                        api_key=key,
                        base_url=config.base_url or _LIVE_PROVIDER_DEFAULT_BASE_URL[config.name],
                        timeout_seconds=config.timeout_seconds,
                        version=config.version,
                        max_retries=config.retry_count,
                    )
                )
            else:
                result.append(DisabledProvider(config.name, config.version))
        else:
            result.append(DisabledProvider(config.name, config.version))
    return tuple(result) or (DisabledProvider(),)
