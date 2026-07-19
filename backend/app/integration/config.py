from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class IntegrationConfigModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class IntegrationGraphNode(IntegrationConfigModel):
    engine: str
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    dependencies: tuple[str, ...] = ()
    required: bool = True


def production_graph() -> tuple[IntegrationGraphNode, ...]:
    return (
        IntegrationGraphNode(engine="market_data", version="1.0.0"),
        IntegrationGraphNode(engine="smc", version="1.0.0", dependencies=("market_data",)),
        IntegrationGraphNode(engine="liquidity", version="1.0.0", dependencies=("market_data", "smc")),
        IntegrationGraphNode(engine="volume_profile", version="1.0.0", dependencies=("market_data", "smc", "liquidity")),
        IntegrationGraphNode(engine="institutional_flow", version="1.0.0", dependencies=("market_data", "smc", "liquidity", "volume_profile")),
        IntegrationGraphNode(engine="market_regime", version="1.0.0", dependencies=("market_data", "smc", "liquidity", "volume_profile", "institutional_flow")),
        IntegrationGraphNode(engine="economic_calendar", version="1.0.0"),
        IntegrationGraphNode(
            engine="ai_scoring",
            version="1.0.0",
            dependencies=("market_data", "smc", "liquidity", "volume_profile", "institutional_flow", "market_regime", "economic_calendar"),
        ),
        IntegrationGraphNode(engine="signal_decision", version="1.0.0", dependencies=("ai_scoring",)),
        IntegrationGraphNode(engine="replay", version="1.0.0", dependencies=("market_data", "economic_calendar")),
    )


class InstrumentConfig(IntegrationConfigModel):
    instrument_id: str = "XAUUSD"
    provider_symbols: dict[str, str] = Field(
        default_factory=lambda: {
            "twelve_data": "XAU/USD",
            "alpha_vantage": "XAUUSD",
            "financial_modeling_prep": "XAUUSD",
            "oanda": "XAU_USD",
            "golden": "XAU/USD",
        }
    )
    asset_class: str = "precious_metal"
    quote_currency: str = "USD"
    venue: str = "otc_reference"
    timeframes: tuple[str, ...] = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")


class IntegrationLimits(IntegrationConfigModel):
    event_bus_queue_size: int = Field(default=1000, ge=1, le=100_000)
    event_processing_timeout_seconds: float = Field(default=60, gt=0, le=3600)
    outbox_batch_size: int = Field(default=100, ge=1, le=1000)
    outbox_poll_seconds: float = Field(default=1, gt=0, le=300)
    max_concurrent_keys: int = Field(default=16, ge=1, le=256)
    maximum_candles: int = Field(default=500, ge=20, le=10_000)
    maximum_page_size: int = Field(default=200, ge=1, le=1000)
    maximum_history_days: int = Field(default=365, ge=1, le=3650)
    trace_retention_days: int = Field(default=30, ge=1, le=3650)


class IntegrationPolicy(IntegrationConfigModel):
    schema_version: str = Field(default="1.0", pattern=r"^\d+\.\d+$")
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    stale_after_seconds: int = Field(default=300, ge=1, le=604800)
    future_tolerance_seconds: int = Field(default=5, ge=0, le=300)
    snapshot_max_timestamp_skew_seconds: int = Field(default=300, ge=0, le=86400)
    minimum_quality_score: float = Field(default=50, ge=0, le=100)
    required_evidence: tuple[str, ...] = ("market_data", "smc", "liquidity", "volume_profile", "institutional_flow", "market_regime", "economic_calendar")
    optional_evidence: tuple[str, ...] = ()
    late_event_policy: str = Field(default="reject", pattern=r"^(reject|rebuild)$")


class IntegrationWorkerConfig(IntegrationConfigModel):
    enabled: bool = True
    embedded_api_worker: bool = False
    lease_seconds: int = Field(default=30, ge=3, le=3600)
    heartbeat_seconds: int = Field(default=10, ge=1, le=1200)

    @model_validator(mode="after")
    def lease_order(self) -> IntegrationWorkerConfig:
        if self.heartbeat_seconds >= self.lease_seconds:
            raise ValueError("integration worker heartbeat must precede lease expiry")
        return self


class IntegrationConfig(IntegrationConfigModel):
    enabled: bool = True
    live_pipeline_enabled: bool = True
    graph: tuple[IntegrationGraphNode, ...] = Field(default_factory=production_graph)
    instruments: tuple[InstrumentConfig, ...] = (InstrumentConfig(),)
    limits: IntegrationLimits = IntegrationLimits()
    policy: IntegrationPolicy = IntegrationPolicy()
    worker: IntegrationWorkerConfig = IntegrationWorkerConfig()

    @model_validator(mode="after")
    def validate_graph(self) -> IntegrationConfig:
        names = [item.engine for item in self.graph]
        expected = {
            "market_data",
            "smc",
            "liquidity",
            "volume_profile",
            "institutional_flow",
            "market_regime",
            "economic_calendar",
            "ai_scoring",
            "signal_decision",
            "replay",
        }
        if set(names) != expected or len(names) != len(expected):
            raise ValueError("integration graph must contain exactly the ten production engines")
        positions = {name: index for index, name in enumerate(names)}
        for index, node in enumerate(self.graph):
            if any(dependency not in positions for dependency in node.dependencies):
                raise ValueError(f"unknown dependency for {node.engine}")
            if any(positions[dependency] >= index for dependency in node.dependencies):
                raise ValueError(f"dependency order or cycle detected for {node.engine}")
        if not self.instruments:
            raise ValueError("at least one integration instrument is required")
        if set(self.policy.required_evidence) & set(self.policy.optional_evidence):
            raise ValueError("required and optional evidence cannot overlap")
        return self

    def instrument(self, value: str) -> InstrumentConfig:
        normalized = value.upper().replace("/", "").replace("-", "").replace("_", "")
        for item in self.instruments:
            if item.instrument_id == normalized:
                return item
        raise ValueError(f"unsupported integration instrument: {value}")
