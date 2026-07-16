"""Dynamic engine registration discovery."""

import importlib
import pkgutil
from types import ModuleType

import backend.app.engines as engines_package

from .engine_factory import EngineFactory


class EngineLoader:
    """Discovers engine packages that expose a `registration.register(factory)` hook."""

    def __init__(self, factory: EngineFactory, package: ModuleType = engines_package) -> None:
        self.factory = factory
        self.package = package

    def discover(self) -> tuple[str, ...]:
        loaded: list[str] = []
        prefix = f"{self.package.__name__}."
        for module in sorted(pkgutil.iter_modules(self.package.__path__, prefix), key=lambda item: item.name):
            if not module.ispkg:
                continue
            registration_name = f"{module.name}.registration"
            try:
                registration = importlib.import_module(registration_name)
            except ModuleNotFoundError as exc:
                if exc.name == registration_name:
                    continue
                raise
            register = getattr(registration, "register", None)
            if callable(register):
                register(self.factory)
                loaded.append(module.name)
        return tuple(loaded)
