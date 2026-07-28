"""Compact provider-wire schemas and faithful evidence-reference resolution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .analysis import (
    AIAnalysisOutput,
    AlternativeAnalysisScenario,
    AnalysisBias,
    AnalysisEvidence,
    EvidenceKind,
    HigherTimeframeAnalysis,
    LiquidityAnalysis,
    MarketRegimeAnalysis,
    MarketStructureAnalysis,
    MomentumAnalysis,
    MomentumTrend,
    RegimeClassification,
    SupplyDemandAnalysis,
    VolatilityAnalysis,
    VolatilityState,
    VolatilityTrend,
)


class CompactStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceCatalogItem(CompactStrictModel):
    evidence_id: str = Field(pattern=r"^E[1-9][0-9]?$")
    fact: str = Field(min_length=1, max_length=140)
    kind: EvidenceKind
    source_type: str = Field(min_length=1, max_length=48)
    source_reference: str = Field(min_length=1, max_length=128)
    timeframe: str | None = Field(default=None, max_length=8)
    observed_value: str | float | int | bool | None = None


EvidenceRefs = tuple[str, ...]


class CompactRegime(CompactStrictModel):
    classification: RegimeClassification
    strength: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    evidence_refs: EvidenceRefs = Field(max_length=2)


class CompactHigherTimeframe(CompactStrictModel):
    bias: AnalysisBias
    summary: str = Field(min_length=1, max_length=180)
    evidence_refs: EvidenceRefs = Field(max_length=2)


class CompactStructure(CompactStrictModel):
    short_term: str = Field(min_length=1, max_length=120)
    medium_term: str = Field(min_length=1, max_length=180)
    recent_change: str = Field(min_length=1, max_length=120)
    evidence_refs: EvidenceRefs = Field(max_length=2)


class CompactLiquidity(CompactStrictModel):
    summary: str = Field(min_length=1, max_length=180)
    events: tuple[str, ...] = Field(max_length=2)
    unresolved: tuple[str, ...] = Field(max_length=2)
    evidence_refs: EvidenceRefs = Field(max_length=2)

    @field_validator("events", "unresolved")
    @classmethod
    def bounded_items(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value or len(value) > 100 for value in values):
            raise ValueError("liquidity text items must contain 1-100 characters")
        return values


class CompactSupplyDemand(CompactStrictModel):
    summary: str = Field(min_length=1, max_length=160)
    nearest_supply: float | None = Field(gt=0)
    nearest_demand: float | None = Field(gt=0)
    evidence_refs: EvidenceRefs = Field(max_length=2)


class CompactMomentum(CompactStrictModel):
    direction: AnalysisBias
    strength: float = Field(ge=0, le=100)
    trend: MomentumTrend
    evidence_refs: EvidenceRefs = Field(max_length=2)


class CompactVolatility(CompactStrictModel):
    state: VolatilityState
    trend: VolatilityTrend
    evidence_refs: EvidenceRefs = Field(max_length=2)


class CompactScenario(CompactStrictModel):
    name: str = Field(min_length=1, max_length=60)
    description: str = Field(min_length=1, max_length=180)
    probability: float = Field(ge=0, le=1)
    evidence_refs: EvidenceRefs = Field(max_length=2)


class CompactAIAnalysisOutput(CompactStrictModel):
    analysis_schema_version: Literal["compact-1.0"]
    output_profile: Literal["compact"]
    market_regime: CompactRegime
    higher_timeframe_context: CompactHigherTimeframe
    market_structure: CompactStructure
    liquidity_analysis: CompactLiquidity
    supply_demand_analysis: CompactSupplyDemand
    momentum_analysis: CompactMomentum
    volatility_analysis: CompactVolatility
    bullish_evidence_refs: EvidenceRefs = Field(max_length=3)
    bearish_evidence_refs: EvidenceRefs = Field(max_length=3)
    contradiction_refs: EvidenceRefs = Field(max_length=3)
    key_risk_refs: EvidenceRefs = Field(max_length=3)
    invalidation_conditions: tuple[str, ...] = Field(max_length=2)
    data_quality_warnings: tuple[str, ...] = Field(max_length=3)
    alternative_scenarios: tuple[CompactScenario, ...] = Field(max_length=2)
    analysis_confidence: float = Field(ge=0, le=1)
    executive_summary: str = Field(min_length=1, max_length=320)

    @field_validator("invalidation_conditions")
    @classmethod
    def bounded_invalidations(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value or len(value) > 160 for value in values):
            raise ValueError("invalidation conditions must contain 1-160 characters")
        return values

    @field_validator("data_quality_warnings")
    @classmethod
    def bounded_warnings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value or len(value) > 120 for value in values):
            raise ValueError("data-quality warnings must contain 1-120 characters")
        return values


class CompactRetryAIAnalysisOutput(CompactStrictModel):
    analysis_schema_version: Literal["compact-retry-1.0"]
    output_profile: Literal["compact_retry"]
    market_regime: CompactRegime
    higher_timeframe_context: CompactHigherTimeframe
    market_structure: CompactStructure
    liquidity_analysis: CompactLiquidity
    supply_demand_analysis: CompactSupplyDemand
    momentum_analysis: CompactMomentum
    volatility_analysis: CompactVolatility
    bullish_evidence_refs: EvidenceRefs = Field(max_length=2)
    bearish_evidence_refs: EvidenceRefs = Field(max_length=2)
    contradiction_refs: EvidenceRefs = Field(max_length=2)
    key_risk_refs: EvidenceRefs = Field(max_length=2)
    invalidation_conditions: tuple[str, ...] = Field(max_length=2)
    data_quality_warnings: tuple[str, ...] = Field(max_length=2)
    analysis_confidence: float = Field(ge=0, le=1)


CompactWireOutput = CompactAIAnalysisOutput | CompactRetryAIAnalysisOutput


class CompactOutputValidationError(ValueError):
    def __init__(self, code: str, path: str, message: str) -> None:
        self.code = code
        self.path = path
        super().__init__(message)


_DESCRIPTIVE_LIMITS = {
    "summary": 180,
    "short_term": 120,
    "medium_term": 180,
    "recent_change": 120,
    "name": 60,
    "description": 180,
    "executive_summary": 320,
}
_LIST_LIMITS = {
    "events": 2,
    "unresolved": 2,
    "invalidation_conditions": 2,
    "data_quality_warnings": 3,
    "alternative_scenarios": 2,
}


def normalize_descriptive_overflow(raw: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Locally bound only non-decision prose; identifiers and values are untouched."""

    changes: list[str] = []

    def visit(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                child_path = f"{path}.{key}" if path else key
                if key in _DESCRIPTIVE_LIMITS and isinstance(item, str):
                    limit = _DESCRIPTIVE_LIMITS[key]
                    if len(item) > limit:
                        changes.append(child_path)
                        item = item[:limit].rstrip()
                if key in _LIST_LIMITS and isinstance(item, list):
                    limit = _LIST_LIMITS[key]
                    if len(item) > limit:
                        changes.append(child_path)
                        item = item[:limit]
                normalized[key] = visit(item, child_path)
            return normalized
        if isinstance(value, list):
            return [visit(item, f"{path}.{index}") for index, item in enumerate(value)]
        return value

    return visit(raw, ""), tuple(changes)


def validate_evidence_references(
    output: CompactWireOutput,
    catalog: tuple[EvidenceCatalogItem, ...],
) -> None:
    allowed = {item.evidence_id for item in catalog}

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, BaseModel):
            walk(value.model_dump(mode="python"), path)
        elif isinstance(value, dict):
            for key, item in value.items():
                child = f"{path}.{key}" if path else key
                if key.endswith("_refs") or key == "evidence_refs":
                    for index, reference in enumerate(item):
                        if reference not in allowed:
                            raise CompactOutputValidationError(
                                "unknown_evidence_reference",
                                f"provider_response.{child}.{index}",
                                f"unknown evidence reference {reference}",
                            )
                else:
                    walk(item, child)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}.{index}")

    walk(output)


def resolve_compact_output(
    output: CompactWireOutput,
    catalog: tuple[EvidenceCatalogItem, ...],
) -> AIAnalysisOutput:
    by_id = {item.evidence_id: item for item in catalog}

    def evidence(refs: EvidenceRefs) -> tuple[AnalysisEvidence, ...]:
        return tuple(
            AnalysisEvidence(
                claim=by_id[reference].fact,
                kind=by_id[reference].kind,
                source_type=by_id[reference].source_type,
                source_reference=by_id[reference].source_reference,
                timeframe=by_id[reference].timeframe,
                observed_value=by_id[reference].observed_value,
            )
            for reference in refs
        )

    scenarios = (
        tuple(
            AlternativeAnalysisScenario(
                name=item.name,
                description=item.description,
                probability=item.probability,
                confirmation_evidence=item.evidence_refs,
            )
            for item in output.alternative_scenarios
        )
        if isinstance(output, CompactAIAnalysisOutput)
        else ()
    )
    summary = (
        output.executive_summary
        if isinstance(output, CompactAIAnalysisOutput)
        else (
            f"{output.market_regime.classification.value} regime; "
            f"{output.market_structure.recent_change}"
        )[:320]
    )
    return AIAnalysisOutput(
        market_regime=MarketRegimeAnalysis(
            classification=output.market_regime.classification,
            strength=output.market_regime.strength,
            confidence=output.market_regime.confidence,
            evidence=evidence(output.market_regime.evidence_refs),
        ),
        higher_timeframe_context=HigherTimeframeAnalysis(
            bias=output.higher_timeframe_context.bias,
            description=output.higher_timeframe_context.summary,
            evidence=evidence(output.higher_timeframe_context.evidence_refs),
        ),
        market_structure=MarketStructureAnalysis(
            short_term=output.market_structure.short_term,
            medium_term=output.market_structure.medium_term,
            higher_timeframe=output.higher_timeframe_context.summary,
            recent_change=output.market_structure.recent_change,
            evidence=evidence(output.market_structure.evidence_refs),
        ),
        liquidity_analysis=LiquidityAnalysis(
            summary=output.liquidity_analysis.summary,
            events=output.liquidity_analysis.events,
            unresolved_liquidity=output.liquidity_analysis.unresolved,
            evidence=evidence(output.liquidity_analysis.evidence_refs),
        ),
        supply_demand_analysis=SupplyDemandAnalysis(
            summary=output.supply_demand_analysis.summary,
            nearest_supply=output.supply_demand_analysis.nearest_supply,
            nearest_demand=output.supply_demand_analysis.nearest_demand,
            evidence=evidence(output.supply_demand_analysis.evidence_refs),
        ),
        momentum_analysis=MomentumAnalysis(
            direction=output.momentum_analysis.direction,
            strength=output.momentum_analysis.strength,
            trend=output.momentum_analysis.trend,
            evidence=evidence(output.momentum_analysis.evidence_refs),
        ),
        volatility_analysis=VolatilityAnalysis(
            state=output.volatility_analysis.state,
            trend=output.volatility_analysis.trend,
            evidence=evidence(output.volatility_analysis.evidence_refs),
        ),
        bullish_evidence=evidence(output.bullish_evidence_refs),
        bearish_evidence=evidence(output.bearish_evidence_refs),
        contradictions=evidence(output.contradiction_refs),
        key_risks=evidence(output.key_risk_refs),
        alternative_scenarios=scenarios,
        analysis_confidence=output.analysis_confidence,
        executive_summary=summary,
        invalidation_conditions=output.invalidation_conditions,
        data_quality_warnings=output.data_quality_warnings,
    )
