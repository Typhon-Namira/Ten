"""Configuration-driven, event-publishing analysis pipeline manager."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from backend.app.core.config import YamlConfigRepository
from backend.app.core.exceptions import ConfigurationError
from backend.app.engines.ai_scoring_engine import AIScoringEngine
from backend.app.engines.economic_calendar_engine import EconomicEvent
from backend.app.engines.institutional_flow_engine import BaselineInstitutionalFlowEngine
from backend.app.engines.liquidity_engine import BaselineLiquidityAnalyzer
from backend.app.engines.market_data_engine import Candle
from backend.app.engines.signal_engine import BaselineSignalEngine, Signal
from backend.app.engines.smc_engine import BaselineSMCAnalyzer
from backend.app.engines.volume_profile_engine import BaselineVolumeProfileAnalyzer
from backend.app.events import DashboardUpdated, EventBus, InMemoryEventBus
from backend.app.features import FeatureRecord, FeatureSnapshot, FeatureStore, InMemoryFeatureStore

from .confidence import ConfidenceCalculator, ConfidenceFactors, ConfidenceResult
from .pipeline_contracts import PipelineExecutionContext
from .registry import EngineRegistry, build_engine_registry


class PipelineStep(BaseModel):
    name: str
    enabled: bool = True
    required: bool = True
    confidence_factor: str | None = None


class PipelineConfig(BaseModel):
    steps: list[PipelineStep] = Field(min_length=1)


class PipelineRunResult(BaseModel):
    correlation_id: UUID
    signal: Signal
    executed_steps: list[str]
    features: FeatureSnapshot
    confidence: ConfidenceResult


class PipelineManager:
    """Executes registered engines in YAML order and communicates through events/features."""

    def __init__(
        self,
        registry: EngineRegistry,
        config: PipelineConfig,
        event_bus: EventBus,
        feature_store: FeatureStore,
        confidence: ConfidenceCalculator,
    ) -> None:
        self.registry = registry
        self.config = config
        self.event_bus = event_bus
        self.feature_store = feature_store
        self.confidence = confidence
        self._validate_configuration()

    @classmethod
    def from_yaml(
        cls,
        registry: EngineRegistry,
        configs: YamlConfigRepository,
        *,
        event_bus: EventBus | None = None,
        feature_store: FeatureStore | None = None,
    ) -> "PipelineManager":
        return cls(
            registry=registry,
            config=configs.load_model("pipeline", PipelineConfig),
            event_bus=event_bus or InMemoryEventBus(),
            feature_store=feature_store or InMemoryFeatureStore(),
            confidence=ConfidenceCalculator.from_yaml(configs),
        )

    def _validate_configuration(self) -> None:
        names = [step.name for step in self.config.steps if step.enabled]
        if len(names) != len(set(names)):
            raise ConfigurationError("Pipeline cannot contain duplicate enabled steps")
        known = set(self.registry.names()) | {"dashboard"}
        unknown = set(names) - known
        if unknown:
            raise ConfigurationError(f"Pipeline contains unknown steps: {', '.join(sorted(unknown))}")
        positions = {name: index for index, name in enumerate(names)}
        for name in names:
            if name == "dashboard":
                continue
            registration = self.registry.registration(name)
            for dependency in registration.definition.metadata.dependencies:
                if dependency not in positions or positions[dependency] >= positions[name]:
                    raise ConfigurationError(f"Pipeline dependency order invalid: {dependency} must precede {name}")

    async def run(self, candles: list[Candle], events: list[EconomicEvent], now: datetime | None = None) -> PipelineRunResult:
        if len(candles) < 2:
            raise ValueError("At least two candles are required for the analysis pipeline")
        correlation_id = uuid4()
        context = PipelineExecutionContext(
            correlation_id=correlation_id,
            candles=sorted(candles, key=lambda item: item.timestamp),
            events=events,
            feature_store=self.feature_store,
            now=now or datetime.now(UTC),
        )
        executed: list[str] = []
        factor_values: dict[str, float] = {}
        confidence_result: ConfidenceResult | None = None
        signal: Signal | None = None

        for step in self.config.steps:
            if not step.enabled:
                continue
            if step.name == "dashboard":
                await self.event_bus.publish(DashboardUpdated(correlation_id=correlation_id, source="pipeline", payload={"signal_available": signal is not None}))
                executed.append(step.name)
                continue
            registration = self.registry.registration(step.name)
            if not registration.enabled:
                if step.required:
                    raise ConfigurationError(f"Required pipeline engine is disabled: {step.name}")
                continue
            if step.name == "signal":
                confidence_result = self.confidence.calculate(ConfidenceFactors.model_validate(factor_values))
                context.calculated_confidence = confidence_result.confidence
                context.confidence_breakdown = confidence_result.breakdown
            result = await self.registry.execute(step.name, context)
            metadata = registration.definition.metadata
            context.results[step.name] = result.output
            await self.feature_store.write(
                FeatureRecord(
                    correlation_id=correlation_id,
                    namespace=result.namespace,
                    engine_name=metadata.name,
                    engine_version=metadata.version,
                    compatibility_version=metadata.compatibility_version,
                    values=result.features,
                )
            )
            if step.confidence_factor and result.confidence_factor is not None:
                factor_values[step.confidence_factor] = result.confidence_factor
            await self.event_bus.publish(result.event_type(correlation_id=correlation_id, source=step.name, payload={"feature_namespace": result.namespace}))
            if isinstance(result.output, Signal):
                signal = result.output
            executed.append(step.name)

        if signal is None or confidence_result is None:
            raise ConfigurationError("Configured pipeline did not produce a signal and deterministic confidence")
        snapshot = await self.feature_store.snapshot(correlation_id)
        return PipelineRunResult(correlation_id=correlation_id, signal=signal, executed_steps=executed, features=snapshot, confidence=confidence_result)


class AnalysisPipeline:
    """Backward-compatible facade over the registry-driven PipelineManager."""

    def __init__(
        self,
        ai_scoring: AIScoringEngine,
        *,
        smc: BaselineSMCAnalyzer | None = None,
        liquidity: BaselineLiquidityAnalyzer | None = None,
        flow: BaselineInstitutionalFlowEngine | None = None,
        volume_profile: BaselineVolumeProfileAnalyzer | None = None,
        calendar: object | None = None,
        signal: BaselineSignalEngine | None = None,
        configs: YamlConfigRepository | None = None,
    ) -> None:
        resolved_configs = configs or YamlConfigRepository()
        registry = build_engine_registry(configs=resolved_configs)
        registry.override("ai_scoring", ai_scoring)
        for name, instance in (("smc", smc), ("liquidity", liquidity), ("institutional_flow", flow), ("volume_profile", volume_profile), ("economic_calendar", calendar), ("signal", signal)):
            if instance is not None:
                registry.override(name, instance)
        self.manager = PipelineManager.from_yaml(registry, resolved_configs)

    async def analyze(self, candles: list[Candle], events: list[EconomicEvent], now: datetime | None = None) -> Signal:
        return (await self.manager.run(candles, events, now)).signal
