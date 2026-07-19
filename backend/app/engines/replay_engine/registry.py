from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .exceptions import ReplayConfigurationError


@dataclass(frozen=True)
class ReplayEngineRegistration:
    engine_name: str
    engine_version: str
    compatibility_version: str
    supports_replay: bool
    supports_point_in_time: bool
    supports_isolated_feature_store: bool
    supports_isolated_events: bool
    deterministic_under_fixed_input: bool
    required_sources: tuple[str, ...] = ()
    required_engine_dependencies: tuple[str, ...] = ()

    def validate_safe(self) -> None:
        flags = (
            self.supports_replay,
            self.supports_point_in_time,
            self.supports_isolated_feature_store,
            self.supports_isolated_events,
            self.deterministic_under_fixed_input,
        )
        if not all(flags):
            raise ReplayConfigurationError(f"engine is not replay-safe: {self.engine_name}")


class ReplayCompatibilityRegistry:
    def __init__(self, registrations: Iterable[ReplayEngineRegistration] = ()) -> None:
        self._items: dict[tuple[str, str], ReplayEngineRegistration] = {}
        for item in registrations:
            self.register(item)

    def register(self, registration: ReplayEngineRegistration) -> None:
        key = (registration.engine_name, registration.engine_version)
        if key in self._items:
            raise ReplayConfigurationError(f"duplicate replay engine registration: {registration.engine_name}@{registration.engine_version}")
        registration.validate_safe()
        self._items[key] = registration

    def resolve(self, selected: tuple[str, ...], versions: dict[str, str]) -> tuple[ReplayEngineRegistration, ...]:
        resolved = []
        for name in selected:
            key = (name, versions.get(name, ""))
            if key not in self._items:
                raise ReplayConfigurationError(f"unknown or incompatible replay engine: {name}@{key[1]}")
            resolved.append(self._items[key])
        names = {item.engine_name for item in resolved}
        for item in resolved:
            missing = set(item.required_engine_dependencies) - names
            if missing:
                raise ReplayConfigurationError(f"missing replay engine dependency for {item.engine_name}: {sorted(missing)}")
        return self._topological(resolved)

    @staticmethod
    def _topological(items: list[ReplayEngineRegistration]) -> tuple[ReplayEngineRegistration, ...]:
        by_name = {item.engine_name: item for item in items}
        visiting: set[str] = set()
        visited: set[str] = set()
        ordered: list[ReplayEngineRegistration] = []

        def visit(name: str) -> None:
            if name in visiting:
                raise ReplayConfigurationError("replay engine dependency cycle detected")
            if name in visited:
                return
            visiting.add(name)
            item = by_name[name]
            for dependency in sorted(item.required_engine_dependencies):
                if dependency in by_name:
                    visit(dependency)
            visiting.remove(name)
            visited.add(name)
            ordered.append(item)

        for name in sorted(by_name):
            visit(name)
        return tuple(ordered)

    def registrations(self) -> tuple[ReplayEngineRegistration, ...]:
        return tuple(sorted(self._items.values(), key=lambda item: (item.engine_name, item.engine_version)))


def production_replay_registry() -> ReplayCompatibilityRegistry:
    dependencies = {
        "market_data": (),
        "smc": ("market_data",),
        "liquidity": ("market_data", "smc"),
        "volume_profile": ("market_data",),
        "institutional_flow": ("market_data", "smc", "liquidity", "volume_profile"),
        "market_regime": ("market_data", "smc", "liquidity", "volume_profile", "institutional_flow"),
        "economic_calendar": (),
        "ai_scoring": ("market_data", "smc", "liquidity", "volume_profile", "institutional_flow", "market_regime", "economic_calendar"),
        "signal_decision": ("ai_scoring", "market_regime", "economic_calendar"),
    }
    items = [
        ReplayEngineRegistration(
            engine_name=name,
            engine_version="1.0.0",
            compatibility_version="1.0",
            supports_replay=True,
            supports_point_in_time=True,
            supports_isolated_feature_store=True,
            supports_isolated_events=True,
            deterministic_under_fixed_input=True,
            required_sources=("historical_candles",) if name != "economic_calendar" else ("economic_calendar",),
            required_engine_dependencies=required,
        )
        for name, required in dependencies.items()
    ]
    return ReplayCompatibilityRegistry(items)
