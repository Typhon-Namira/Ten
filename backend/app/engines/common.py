"""Shared engine contracts and health models."""

from abc import ABC, abstractmethod
from datetime import UTC, date, datetime
from enum import StrEnum
from pydantic import BaseModel, Field


class EngineState(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class EngineLifecycleStatus(StrEnum):
    EXPERIMENTAL = "experimental"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"


class EngineMetadata(BaseModel):
    """Immutable identity and compatibility contract for an engine version."""

    name: str
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    compatibility_version: str = Field(pattern=r"^\d+\.\d+$")
    created_date: date
    status: EngineLifecycleStatus = EngineLifecycleStatus.STABLE
    dependencies: tuple[str, ...] = ()
    description: str
    enabled: bool = True
    config_key: str
    feature_flag: str | None = None


class EngineStatus(BaseModel):
    name: str
    version: str
    state: EngineState = EngineState.READY
    details: str = "operational"
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    compatibility_version: str = "1.0"
    created_date: date = date(2026, 7, 16)
    lifecycle_status: EngineLifecycleStatus = EngineLifecycleStatus.STABLE
    dependencies: tuple[str, ...] = ()
    description: str = ""
    enabled: bool = True


class AnalysisEngine[InputT, OutputT](ABC):
    """Stable contract implemented by every deterministic engine."""

    name: str
    version: str
    compatibility_version: str = "1.0"

    @abstractmethod
    def analyze(self, data: InputT) -> OutputT:
        """Analyze normalized input without side effects."""

    def status(self) -> EngineStatus:
        return EngineStatus(name=self.name, version=self.version)
