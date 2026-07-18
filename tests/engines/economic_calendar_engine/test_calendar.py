from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.engines.economic_calendar_engine import (
    BaselineEconomicCalendarEngine,
    DisabledProvider,
    EconomicCalendarCheckpoint,
    EconomicCalendarConfig,
    EconomicCalendarService,
    EconomicEventStatus,
    EventCategory,
    EventImportance,
    FileImportProvider,
    FixedClock,
    InMemoryEconomicCalendarRepository,
    InMemoryProvider,
    ProviderConfig,
    ProviderMode,
    RiskWindowConfig,
    TimingPrecision,
)
from backend.app.engines.economic_calendar_engine.analyzer import (
    _phase,
    build_snapshot,
    clusters,
    explain,
    instrument_context,
    reconcile,
    revision_between,
    surprise,
    symbol_currencies,
)
from backend.app.engines.economic_calendar_engine.models import ConflictState, FreshnessState, RevisionType, RiskWindowPhase
from backend.app.engines.economic_calendar_engine.normalization import canonical_name, normalize_observation, parse_schedule, parse_value
from backend.app.engines.economic_calendar_engine.providers import ProviderFetchRequest, observation_from_mapping
from backend.app.engines.economic_calendar_engine.repository import _checkpoint_bytes
from backend.app.events import InMemoryEventBus
from backend.app.features import InMemoryFeatureStore

NOW = datetime(2026, 7, 18, 12, tzinfo=UTC)


def row(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": "us-cpi-2026-07",
        "name": "US CPI m/m",
        "category": "inflation",
        "country": "US",
        "currency": "USD",
        "importance": "high",
        "status": "scheduled",
        "scheduled_at": (NOW + timedelta(minutes=10)).isoformat(),
        "timezone": "UTC",
        "available_at": (NOW - timedelta(hours=1)).isoformat(),
        "response_received_at": (NOW - timedelta(hours=1)).isoformat(),
        "ingested_at": (NOW - timedelta(hours=1)).isoformat(),
        "forecast": "3.2%",
        "previous": "3.0%",
        "unit": "%",
    }
    value.update(updates)
    return value


def observation(provider: str = "alpha", **updates: object):
    return observation_from_mapping(provider, "1", row(**updates), NOW)


def event(provider: str = "alpha", **updates: object):
    return normalize_observation(
        observation(provider, **updates),
        EconomicCalendarConfig(
            providers=(ProviderConfig(name=provider, mode=ProviderMode.IN_MEMORY_TEST_PROVIDER, enabled=True),), provider_priority=(provider,)
        ),
    )


@pytest.mark.parametrize(
    ("raw", "value", "kind", "scale"),
    [
        ("3.2%", 3.2, "percent", 1),
        ("-0.4%", -0.4, "percent", 1),
        ("1,250K", 1_250_000, "number", 1000),
        ("2.5M", 2_500_000, "number", 1_000_000),
        ("1.2B", 1_200_000_000, "number", 1_000_000_000),
        ("$4.3B", 4_300_000_000, "currency", 1_000_000_000),
        ("<1.0%", 1.0, "percent", 1),
        ("(2.5)", -2.5, "number", 1),
        ("1.234,5", 1234.5, "number", 1),
        ("1,25", 1.25, "number", 1),
    ],
)
def test_value_parser(raw: str, value: float, kind: str, scale: float) -> None:
    parsed = parse_value(raw)
    assert parsed.value == value
    assert parsed.value_type.value == kind
    assert parsed.scale == scale


@pytest.mark.parametrize("raw", [None, "", "N/A", "-", "Tentative", "TBD", "--"])
def test_unavailable_value_is_never_zero(raw: str | None) -> None:
    parsed = parse_value(raw)
    assert parsed.value is None


def test_numeric_and_text_value_parsing() -> None:
    assert parse_value(5).value == 5
    assert parse_value("not numeric").value is None
    assert parse_value("€12").unit == "€"


def test_schedule_parsing_and_timezone_safety() -> None:
    exact, precision, warnings = parse_schedule("2026-07-18T08:30:00", "America/New_York")
    assert exact == datetime(2026, 7, 18, 12, 30, tzinfo=UTC)
    assert precision == TimingPrecision.MINUTE and not warnings
    date_only, precision, _ = parse_schedule("2026-07-18", "Europe/London")
    assert date_only is not None and precision == TimingPrecision.DATE_ONLY
    assert parse_schedule("Tentative", None)[1] == TimingPrecision.TENTATIVE
    assert parse_schedule("bad", "UTC")[2]
    fallback, _, warnings = parse_schedule("2026-07-18T10:00:00", "Invalid/Zone")
    assert fallback == datetime(2026, 7, 18, 10, tzinfo=UTC) and warnings


def test_models_and_configuration_are_strict() -> None:
    assert canonical_name("  U.S. — Core CPI! ") == "u s core cpi"
    with pytest.raises(ValidationError):
        ProviderConfig(name="bad", timezone="Mars/Base")
    with pytest.raises(ValidationError):
        RiskWindowConfig(pre_minutes=5, imminent_minutes=6, post_minutes=5, cooldown_minutes=5)
    with pytest.raises(ValidationError):
        EconomicCalendarConfig(providers=(ProviderConfig(name="x"), ProviderConfig(name="x")), provider_priority=("x", "x"))
    with pytest.raises(ValidationError):
        EconomicCalendarConfig(providers=(ProviderConfig(name="x"),), provider_priority=("other",))
    with pytest.raises(ValidationError):
        EconomicCalendarConfig(importance_weights={"high": -1})
    with pytest.raises(ValidationError):
        EconomicCalendarConfig(repository_mode="sqlite")
    with pytest.raises(ValidationError):
        EconomicCalendarConfig(windows={})


def test_normalization_preserves_semantics() -> None:
    item = event(actual="3.4%", status="published", importance="red", category="rates")
    assert item.status == EconomicEventStatus.RELEASED
    assert item.actual_value == 3.4
    assert item.forecast_value == 3.2
    assert item.category == EventCategory.INTEREST_RATE
    assert item.importance == EventImportance.HIGH
    assert item.publication_state.value == "first_release"
    assert item.probabilistic_context is True and item.trading_instruction is False
    assert item.name == item.display_name and item.currency == "USD"
    assert item.actual == item.actual_value and item.forecast == item.forecast_value and item.previous == item.previous_value


def test_normalization_unknown_and_tentative() -> None:
    item = event(country="Unknown", currency="", importance="mystery", category="mystery", status="mystery", scheduled_at="Tentative", actual="text")
    assert item.timing_precision == TimingPrecision.TENTATIVE
    assert item.status == EconomicEventStatus.TENTATIVE
    assert item.importance == EventImportance.UNKNOWN
    assert item.category == EventCategory.UNKNOWN
    assert item.normalization_confidence < 1


def test_reconciliation_priority_and_conflicts() -> None:
    first = event("alpha", actual="3.4%")
    second = event("beta", actual="3.5%")
    combined = reconcile((second, first), ("alpha", "beta"))
    assert len(combined) == 1
    assert combined[0].actual_value == 3.4
    assert combined[0].conflict_state == ConflictState.MATERIAL
    assert len(combined[0].provider_records) == 2
    assert combined == reconcile((first, second), ("alpha", "beta"))


def test_revision_classification_and_deduplication() -> None:
    initial = event()
    discovery = revision_between(None, initial, 1)
    assert discovery and discovery.revision_type == RevisionType.INITIAL_DISCOVERY
    assert revision_between(initial, initial, 2) is None
    released = event(
        actual="3.4%", status="released", available_at=(NOW + timedelta(minutes=11)).isoformat(), response_received_at=(NOW + timedelta(minutes=11)).isoformat()
    )
    release = revision_between(initial, released, 2)
    assert release and release.revision_type == RevisionType.FIRST_RELEASE
    revised = released.model_copy(update={"actual_value": 3.5, "available_at": NOW + timedelta(minutes=20)})
    assert revision_between(released, revised, 3).revision_type == RevisionType.VALUE_REVISION  # type: ignore[union-attr]
    cancelled = initial.model_copy(update={"status": EconomicEventStatus.CANCELLED, "is_cancelled": True})
    assert revision_between(initial, cancelled, 2).revision_type == RevisionType.CANCELLATION  # type: ignore[union-attr]
    postponed = initial.model_copy(update={"status": EconomicEventStatus.POSTPONED, "is_postponed": True})
    assert revision_between(initial, postponed, 2).revision_type == RevisionType.POSTPONEMENT  # type: ignore[union-attr]
    changed_status = initial.model_copy(update={"status": EconomicEventStatus.CONFIRMED})
    assert revision_between(initial, changed_status, 2).revision_type == RevisionType.STATUS_CHANGE  # type: ignore[union-attr]
    conflict = initial.model_copy(update={"conflict_state": ConflictState.MATERIAL})
    assert revision_between(initial, conflict, 2).revision_type == RevisionType.PROVIDER_CONFLICT  # type: ignore[union-attr]
    metadata = initial.model_copy(update={"importance": EventImportance.CRITICAL})
    assert revision_between(initial, metadata, 2).revision_type == RevisionType.METADATA_UPDATE  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [(90, "outside"), (20, "pre_event"), (5, "imminent"), (0, "at_event"), (-5, "post_event"), (-50, "cooldown"), (-100, "outside")],
)
def test_risk_window_phases(minutes: int, expected: str) -> None:
    item = event(scheduled_at=(NOW + timedelta(minutes=minutes)).isoformat())
    assert _phase(item, NOW, EconomicCalendarConfig()) == RiskWindowPhase(expected)


def test_cancelled_tentative_and_clusters() -> None:
    cancelled = event(status="cancelled").model_copy(update={"is_cancelled": True})
    assert _phase(cancelled, NOW, EconomicCalendarConfig()) == RiskWindowPhase.OUTSIDE
    tentative = event(scheduled_at="Tentative")
    assert _phase(tentative, NOW, EconomicCalendarConfig()) == RiskWindowPhase.UNKNOWN
    first = event(id="a", name="CPI", scheduled_at=NOW.isoformat())
    second = event(id="b", name="Core CPI", scheduled_at=(NOW + timedelta(minutes=5)).isoformat())
    values = clusters((first, second, cancelled), EconomicCalendarConfig())
    assert len(values) == 1 and values[0].event_count == 2
    assert clusters((first,), EconomicCalendarConfig()) == ()


def test_snapshot_context_explanation_and_surprise() -> None:
    statuses = (
        pytest.importorskip("backend.app.engines.economic_calendar_engine.models").ProviderStatus(
            provider_name="alpha", mode=ProviderMode.IN_MEMORY_TEST_PROVIDER, enabled=True, authenticated=True, reachable=True, last_success=NOW
        ),
    )
    values = (
        event(id="old", name="Payrolls", category="employment", actual="200K", forecast="180K", scheduled_at=(NOW - timedelta(minutes=20)).isoformat()),
        event(id="new", name="CPI", scheduled_at=(NOW + timedelta(minutes=5)).isoformat()),
    )
    snapshot = build_snapshot(values, NOW, NOW - timedelta(days=1), NOW + timedelta(days=1), statuses, EconomicCalendarConfig())
    assert snapshot.event_count == 2 and snapshot.freshness == FreshnessState.FRESH
    context = instrument_context("EURUSD", snapshot, EconomicCalendarConfig())
    assert context.next_relevant_event is not None
    assert context.risk_score <= 1 and context.trading_instruction is False
    assert explain(context).risk_components["risk"] == context.risk_score
    released = values[0]
    result = surprise(released, EconomicCalendarConfig())
    assert result["available"] is True and result["direction"] == "negative"
    assert surprise(event(actual=None), EconomicCalendarConfig())["available"] is False
    assert symbol_currencies("GBPJPY", EconomicCalendarConfig()) == ("JPY", "GBP") or set(symbol_currencies("GBPJPY", EconomicCalendarConfig())) == {
        "GBP",
        "JPY",
    }
    assert symbol_currencies("XAUUSD", EconomicCalendarConfig()) == ("USD",)


@pytest.mark.asyncio
async def test_in_memory_provider_contract_and_file_import(tmp_path: Path) -> None:
    provider = InMemoryProvider("fixture", (row(), {"id": "bad"}))
    request = ProviderFetchRequest(start=NOW - timedelta(days=1), end=NOW + timedelta(days=1), limit=10)
    result = await provider.fetch_events(request)
    assert result.success_count == 1 and result.failure_count == 1
    assert await provider.fetch_event("us-cpi-2026-07") is not None
    assert await provider.fetch_event("missing") is None
    assert (await provider.fetch_updates(request, None)).success_count == 1
    assert (await provider.health()).reachable is True
    disabled = DisabledProvider()
    assert not (await disabled.health()).enabled
    assert (await disabled.fetch_events(request)).warnings
    import_root = tmp_path / "imports"
    import_root.mkdir()
    json_path = import_root / "events.json"
    json_path.write_text(json.dumps({"events": [row()]}), encoding="utf-8")
    file_provider = FileImportProvider("file", json_path, import_root=import_root)
    assert (await file_provider.fetch_events(request)).success_count == 1
    csv_path = import_root / "events.csv"
    csv_path.write_text("id,name,available_at,response_received_at\n1,CPI,2026-07-18T11:00:00+00:00,2026-07-18T11:00:00+00:00\n", encoding="utf-8")
    assert (await FileImportProvider("csv", csv_path, import_root=import_root).fetch_events(request)).success_count == 1
    with pytest.raises(ValueError):
        FileImportProvider("bad", tmp_path / "outside.json", import_root=import_root)


@pytest.mark.asyncio
async def test_repository_point_in_time_checkpoint_and_retention() -> None:
    repository = InMemoryEconomicCalendarRepository()
    initial = event()
    later = initial.model_copy(update={"actual_value": 3.4, "available_at": NOW + timedelta(minutes=20), "last_updated_at": NOW + timedelta(minutes=20)})
    await repository.save_provider_observations((observation(), observation()))
    assert await repository.get_provider_observation(observation().observation_id)
    await repository.save_event(initial)
    await repository.save_event(initial)
    await repository.save_event(later)
    assert await repository.get_event(initial.event_id) == later
    assert await repository.get_event_at_boundary(initial.event_id, NOW) == initial
    assert await repository.get_event_at_boundary(initial.event_id, NOW - timedelta(days=2)) is None
    revision = revision_between(initial, later, 1)
    assert revision
    await repository.save_revision(revision)
    await repository.save_revision(revision)
    assert len(await repository.list_revisions(initial.event_id)) == 1
    statuses = ((await DisabledProvider().health()),)
    snapshot = build_snapshot((initial,), NOW, NOW - timedelta(days=1), NOW + timedelta(days=1), statuses, EconomicCalendarConfig())
    await repository.save_snapshot(snapshot)
    assert await repository.get_snapshot(snapshot.snapshot_id) == snapshot
    context = instrument_context("XAUUSD", snapshot, EconomicCalendarConfig())
    await repository.save_instrument_context(context)
    assert await repository.get_instrument_context("XAUUSD") == context
    await repository.save_sync_state("x", {"cursor": "1"})
    assert await repository.load_sync_state("x") == {"cursor": "1"}
    state = {"identity": {}}
    checkpoint = EconomicCalendarCheckpoint(
        checkpoint_id=uuid4(), state_payload=state, payload_hash=sha256(_checkpoint_bytes(state)).hexdigest(), created_at=NOW
    )
    await repository.save_checkpoint(checkpoint)
    assert await repository.load_checkpoint() == checkpoint
    with pytest.raises(ValueError):
        await repository.save_checkpoint(checkpoint.model_copy(update={"payload_hash": "0" * 64}))
    result = await repository.prune_history(0, 0, 0)
    assert result["events"] == 1


@pytest.mark.asyncio
async def test_service_sync_replay_features_events_and_recovery() -> None:
    config = EconomicCalendarConfig(
        providers=(ProviderConfig(name="fixture", mode=ProviderMode.IN_MEMORY_TEST_PROVIDER, enabled=True),), provider_priority=("fixture",)
    )
    repository = InMemoryEconomicCalendarRepository()
    bus, store = InMemoryEventBus(), InMemoryFeatureStore()
    service = EconomicCalendarService(bus, store, config, repository, (InMemoryProvider("fixture", (row(),)),), clock=FixedClock(NOW))
    assert await service.restore() is False
    snapshot = await service.synchronize(NOW - timedelta(days=1), NOW + timedelta(days=1), boundary=NOW)
    assert snapshot.event_count == 1
    assert service.metrics.events_inserted == 1
    context = await service.context("XAUUSD", as_of=NOW)
    assert context.next_relevant_event
    features = await store.snapshot(context.context_id)
    assert features.features["economic_calendar"]["trading_instruction"] is False
    replayed = await service.replay("XAUUSD", NOW)
    assert replayed.context_id == context.context_id
    assert await service.event(snapshot.events[0].event_id, NOW)
    assert await service.revisions(snapshot.events[0].event_id)
    assert await service.observations(snapshot.events[0].event_id)
    assert await service.event_clusters(NOW - timedelta(days=1), NOW + timedelta(days=1), NOW) == ()
    assert (await service.explanation("XAUUSD", NOW)).headline
    assert service.health()["initialized"] is True
    assert (await service.provider_status())[0].reachable
    recovered = EconomicCalendarService(bus, store, config, repository, (InMemoryProvider("fixture", (row(),)),), clock=FixedClock(NOW))
    assert await recovered.restore() is True
    assert recovered.recovery_state == "recovered"
    assert bus.history()


@pytest.mark.asyncio
async def test_service_reschedule_and_incremental_consistency() -> None:
    config = EconomicCalendarConfig(
        providers=(ProviderConfig(name="fixture", mode=ProviderMode.IN_MEMORY_TEST_PROVIDER, enabled=True),), provider_priority=("fixture",)
    )
    repository = InMemoryEconomicCalendarRepository()
    service = EconomicCalendarService(
        InMemoryEventBus(), InMemoryFeatureStore(), config, repository, (InMemoryProvider("fixture", (row(),)),), clock=FixedClock(NOW)
    )
    await service.restore()
    full = await service.synchronize(NOW - timedelta(days=1), NOW + timedelta(days=2), boundary=NOW)
    service.providers = (
        InMemoryProvider(
            "fixture",
            (
                row(
                    scheduled_at=(NOW + timedelta(days=1)).isoformat(),
                    available_at=(NOW + timedelta(hours=1)).isoformat(),
                    response_received_at=(NOW + timedelta(hours=1)).isoformat(),
                ),
            ),
        ),
    )
    incremental = await service.synchronize(NOW - timedelta(days=1), NOW + timedelta(days=2), boundary=NOW + timedelta(hours=1), incremental=True)
    assert incremental.events[0].event_id == full.events[0].event_id
    assert incremental.events[0].is_rescheduled


def test_baseline_pipeline_contract_has_no_instruction() -> None:
    item = event()
    result = BaselineEconomicCalendarEngine().analyze((NOW, [item]))
    assert result.risk_level == EventImportance.HIGH
    assert result.no_trade is False
    assert result.trading_instruction is False


def test_fixed_clock_rejects_naive_time() -> None:
    with pytest.raises(ValueError):
        FixedClock(datetime(2026, 1, 1))
