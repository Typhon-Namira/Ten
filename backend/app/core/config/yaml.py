"""Validated YAML configuration repository."""

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from backend.app.core.exceptions import ConfigurationError

ConfigT = TypeVar("ConfigT", bound=BaseModel)


class YamlConfigRepository:
    """Loads named configuration documents from a single controlled directory."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = (directory or Path("configs")).resolve()

    def load(self, name: str) -> dict[str, Any]:
        """Load a YAML mapping by safe file stem."""

        if not name.replace("_", "").replace("-", "").isalnum():
            raise ConfigurationError(f"Invalid configuration name: {name}")
        path = (self.directory / f"{name}.yaml").resolve()
        if self.directory not in path.parents or not path.is_file():
            raise ConfigurationError(f"Configuration not found: {name}")
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Invalid YAML configuration: {path.name}") from exc
        if not isinstance(value, dict):
            raise ConfigurationError(f"Configuration root must be a mapping: {path.name}")
        return value

    def load_model(self, name: str, model: type[ConfigT]) -> ConfigT:
        """Load and validate a YAML document into a typed model."""

        return model.model_validate(self.load(name))
