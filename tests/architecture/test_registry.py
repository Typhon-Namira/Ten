from backend.app.services import EngineFactory, EngineLoader, build_engine_registry


def test_loader_discovers_versioned_engine_registrations() -> None:
    factory = EngineFactory()
    loaded = EngineLoader(factory).discover()
    names = {definition.metadata.name for definition in factory.definitions()}
    assert "backend.app.engines.smc_engine" in loaded
    assert {"market_data", "smc", "liquidity", "institutional_flow", "volume_profile", "economic_calendar", "ai_scoring", "signal", "market_regime", "replay"} <= names
    assert all(definition.metadata.version == "1.0.0" for definition in factory.definitions())
    assert all(definition.metadata.compatibility_version == "1.0" for definition in factory.definitions())


def test_registry_respects_disabled_infrastructure_engines() -> None:
    registry = build_engine_registry()
    statuses = {status.name: status for status in registry.statuses()}
    assert statuses["smc"].enabled is True
    assert statuses["market_regime"].enabled is False
    assert statuses["replay"].enabled is False
