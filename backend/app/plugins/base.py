"""Stable plugin lifecycle and metadata contracts."""

from abc import ABC, abstractmethod
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class PluginType(StrEnum):
    AI_PROVIDER = "ai_provider"
    MARKET_DATA_PROVIDER = "market_data_provider"
    ANALYSIS_ENGINE = "analysis_engine"
    NOTIFICATION_PROVIDER = "notification_provider"


class PluginStatus(StrEnum):
    EXPERIMENTAL = "experimental"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"


class PluginMetadata(BaseModel):
    name: str
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    compatibility_version: str = Field(pattern=r"^\d+\.\d+$")
    created_date: date
    plugin_type: PluginType
    description: str
    status: PluginStatus = PluginStatus.STABLE


class Plugin(ABC):
    metadata: PluginMetadata

    @abstractmethod
    async def start(self) -> None:
        """Allocate provider resources."""

    @abstractmethod
    async def stop(self) -> None:
        """Release provider resources."""
