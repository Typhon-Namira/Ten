"""Compact provider-wire schemas and faithful evidence-reference resolution."""

from __future__ import annotations

from hashlib import sha256
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.config import JsonDict

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
HIGHER_TIMEFRAME_SUMMARY_LIMIT = 180
_STRING_FROM_LIST: JsonDict = {
    "x-ten-normalize": ["string_list_to_string"]
}
_LIST_FROM_STRING: JsonDict = {
    "x-ten-normalize": ["string_to_string_list"]
}


class CompactRegime(CompactStrictModel):
    classification: RegimeClassification
    strength: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    evidence_refs: EvidenceRefs = Field(json_schema_extra=_LIST_FROM_STRING)


class CompactHigherTimeframe(CompactStrictModel):
    bias: AnalysisBias
    summary: str = Field(
        min_length=1,
        max_length=HIGHER_TIMEFRAME_SUMMARY_LIMIT,
        json_schema_extra=_STRING_FROM_LIST,
    )
    evidence_refs: EvidenceRefs = Field(json_schema_extra=_LIST_FROM_STRING)


class CompactStructure(CompactStrictModel):
    short_term: str = Field(
        min_length=1, max_length=120, json_schema_extra=_STRING_FROM_LIST
    )
    medium_term: str = Field(
        min_length=1, max_length=180, json_schema_extra=_STRING_FROM_LIST
    )
    recent_change: str = Field(
        min_length=1, max_length=120, json_schema_extra=_STRING_FROM_LIST
    )
    evidence_refs: EvidenceRefs = Field(json_schema_extra=_LIST_FROM_STRING)


class CompactLiquidity(CompactStrictModel):
    summary: str = Field(
        min_length=1, max_length=180, json_schema_extra=_STRING_FROM_LIST
    )
    events: tuple[str, ...] = Field(json_schema_extra=_LIST_FROM_STRING)
    unresolved: tuple[str, ...] = Field(json_schema_extra=_LIST_FROM_STRING)
    evidence_refs: EvidenceRefs = Field(json_schema_extra=_LIST_FROM_STRING)

    @field_validator("events", "unresolved")
    @classmethod
    def bounded_items(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value or len(value) > 100 for value in values):
            raise ValueError("liquidity text items must contain 1-100 characters")
        return values


class CompactSupplyDemand(CompactStrictModel):
    summary: str = Field(
        min_length=1, max_length=160, json_schema_extra=_STRING_FROM_LIST
    )
    nearest_supply_ref: str | None = Field(
        pattern=r"^SZ[1-3]$",
    )
    nearest_demand_ref: str | None = Field(
        pattern=r"^DZ[1-3]$",
    )
    evidence_refs: EvidenceRefs = Field(json_schema_extra=_LIST_FROM_STRING)


class CompactMomentum(CompactStrictModel):
    direction: AnalysisBias
    strength: float = Field(ge=0, le=100)
    trend: MomentumTrend
    evidence_refs: EvidenceRefs = Field(json_schema_extra=_LIST_FROM_STRING)


class CompactVolatility(CompactStrictModel):
    state: VolatilityState
    trend: VolatilityTrend
    evidence_refs: EvidenceRefs = Field(json_schema_extra=_LIST_FROM_STRING)


class CompactScenario(CompactStrictModel):
    name: str = Field(
        min_length=1, max_length=60, json_schema_extra=_STRING_FROM_LIST
    )
    description: str = Field(
        min_length=1, max_length=180, json_schema_extra=_STRING_FROM_LIST
    )
    probability: float = Field(ge=0, le=1)
    evidence_refs: EvidenceRefs = Field(json_schema_extra=_LIST_FROM_STRING)


class CompactAIAnalysisOutput(CompactStrictModel):
    analysis_schema_version: Literal["compact-1.1"]
    output_profile: Literal["compact"]
    market_regime: CompactRegime
    higher_timeframe_context: CompactHigherTimeframe
    market_structure: CompactStructure
    liquidity_analysis: CompactLiquidity
    supply_demand_analysis: CompactSupplyDemand
    momentum_analysis: CompactMomentum
    volatility_analysis: CompactVolatility
    bullish_evidence_refs: EvidenceRefs = Field(json_schema_extra=_LIST_FROM_STRING)
    bearish_evidence_refs: EvidenceRefs = Field(json_schema_extra=_LIST_FROM_STRING)
    contradiction_refs: EvidenceRefs = Field(json_schema_extra=_LIST_FROM_STRING)
    key_risk_refs: EvidenceRefs = Field(json_schema_extra=_LIST_FROM_STRING)
    invalidation_conditions: tuple[str, ...] = Field(
        json_schema_extra=_LIST_FROM_STRING
    )
    data_quality_warnings: tuple[str, ...] = Field(
        json_schema_extra=_LIST_FROM_STRING
    )
    alternative_scenarios: tuple[CompactScenario, ...]
    analysis_confidence: float = Field(ge=0, le=1)
    executive_summary: str = Field(
        min_length=1, max_length=320, json_schema_extra=_STRING_FROM_LIST
    )

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


AuthoritativeAIResponse = CompactAIAnalysisOutput
# A retry changes the token budget and request classification, never the wire
# contract. Keep the old import name as a type alias for internal callers while
# retaining exactly one authoritative Pydantic response model.
CompactRetryAIAnalysisOutput = AuthoritativeAIResponse
CompactWireOutput = AuthoritativeAIResponse


@dataclass(frozen=True)
class CanonicalResponseValidation:
    provider_output: dict[str, Any]
    normalized_output: dict[str, Any]
    wire_output: CompactWireOutput
    resolved_output: AIAnalysisOutput
    normalization_details: tuple[dict[str, Any], ...]
    evidence_ref_truncations: tuple[str, ...]


def canonical_response_model(*, retry: bool = False) -> type[CompactStrictModel]:
    """Return the sole provider-response model for an authoritative attempt."""

    del retry
    return AuthoritativeAIResponse


class CompactOutputValidationError(ValueError):
    def __init__(self, code: str, path: str, message: str) -> None:
        self.code = code
        self.path = path
        super().__init__(message)


def _schema_branch_for_value(
    schema: dict[str, Any],
    value: Any,
) -> dict[str, Any]:
    branches = schema.get("anyOf")
    if not isinstance(branches, list):
        return schema
    value_type = (
        "null" if value is None else "array" if isinstance(value, list) else
        "object" if isinstance(value, dict) else "string" if isinstance(value, str)
        else "number" if isinstance(value, (int, float)) and not isinstance(value, bool)
        else "boolean" if isinstance(value, bool) else None
    )
    return next(
        (
            branch
            for branch in branches
            if isinstance(branch, dict) and branch.get("type") == value_type
        ),
        next((branch for branch in branches if isinstance(branch, dict)), schema),
    )


def normalize_compact_output_shapes(
    raw: dict[str, Any],
    *,
    retry: bool = False,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Normalize only lossless JSON container variants across the wire schema."""

    model = canonical_response_model(retry=retry)
    root_schema = model.model_json_schema()
    definitions = root_schema.get("$defs", {})
    changes: list[dict[str, Any]] = []

    def resolve(schema: dict[str, Any]) -> dict[str, Any]:
        reference = schema.get("$ref")
        if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
            return schema
        resolved = definitions.get(reference.removeprefix("#/$defs/"))
        return resolved if isinstance(resolved, dict) else schema

    def record(path: str, before: Any, after: Any, rule: str) -> None:
        encoded = json.dumps(
            before,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )
        changes.append(
            {
                "path": path,
                "rule": rule,
                "received_type": type(before).__name__,
                "normalized_type": type(after).__name__,
                "received_item_count": len(before) if isinstance(before, list) else None,
                "received_value_hash": sha256(encoded.encode("utf-8")).hexdigest()[:16],
            }
        )

    def visit(value: Any, schema: dict[str, Any], path: str) -> Any:
        schema = resolve(_schema_branch_for_value(resolve(schema), value))
        expected_type = schema.get("type")
        allowed_normalizations = frozenset(schema.get("x-ten-normalize", ()))
        if (
            expected_type == "string"
            and "string_list_to_string" in allowed_normalizations
            and isinstance(value, list)
            and value
            and all(isinstance(item, str) and item.strip() for item in value)
        ):
            normalized_string = " ".join(item.strip() for item in value)
            record(path, value, normalized_string, "non_empty_string_list_to_string")
            value = normalized_string
        elif (
            expected_type == "array"
            and "string_to_string_list" in allowed_normalizations
            and isinstance(value, str)
            and value.strip()
            and resolve(schema.get("items", {})).get("type") == "string"
        ):
            normalized_list = [value.strip()]
            record(path, value, normalized_list, "non_empty_string_to_string_list")
            value = normalized_list

        if expected_type == "object" and isinstance(value, dict):
            properties = schema.get("properties", {})
            return {
                key: visit(
                    item,
                    properties.get(key, {}) if isinstance(properties, dict) else {},
                    f"{path}.{key}" if path else key,
                )
                for key, item in value.items()
            }
        if expected_type == "array" and isinstance(value, list):
            item_schema = schema.get("items", {})
            if not isinstance(item_schema, dict):
                return value
            return [
                visit(item, item_schema, f"{path}.{index}")
                for index, item in enumerate(value)
            ]
        return value

    return visit(raw, root_schema, ""), tuple(changes)


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
                    seen: set[str] = set()
                    for index, reference in enumerate(item):
                        if reference in seen:
                            raise CompactOutputValidationError(
                                "duplicate_evidence_reference",
                                f"provider_response.{child}.{index}",
                                f"duplicate evidence reference {reference}",
                            )
                        seen.add(reference)
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


def validate_zone_references(
    output: CompactWireOutput,
    supply_catalog: tuple[Any, ...],
    demand_catalog: tuple[Any, ...],
) -> None:
    selections = (
        (
            "supply",
            output.supply_demand_analysis.nearest_supply_ref,
            {str(item.zone_id) for item in supply_catalog},
        ),
        (
            "demand",
            output.supply_demand_analysis.nearest_demand_ref,
            {str(item.zone_id) for item in demand_catalog},
        ),
    )
    for kind, reference, allowed in selections:
        path = (
            "provider_response.supply_demand_analysis."
            f"nearest_{kind}_ref"
        )
        if not allowed and reference is not None:
            raise CompactOutputValidationError(
                "reference_must_be_null_when_catalog_empty",
                path,
                f"allowed values: null; {kind} catalog is empty",
            )
        if allowed and reference is None:
            raise CompactOutputValidationError(
                "reference_required_but_missing",
                path,
                f"allowed values: {', '.join(sorted(allowed))}",
            )
        if reference is not None and reference not in allowed:
            raise CompactOutputValidationError(
                f"unknown_{kind}_zone_ref",
                path,
                f"allowed values: {', '.join(sorted(allowed)) or 'null'}",
            )


def resolve_compact_output(
    output: CompactWireOutput,
    catalog: tuple[EvidenceCatalogItem, ...],
    supply_catalog: tuple[Any, ...] = (),
    demand_catalog: tuple[Any, ...] = (),
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

    supply_by_id = {str(item.zone_id): item for item in supply_catalog}
    demand_by_id = {str(item.zone_id): item for item in demand_catalog}
    supply_ref = output.supply_demand_analysis.nearest_supply_ref
    demand_ref = output.supply_demand_analysis.nearest_demand_ref

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
            nearest_supply=(
                float(supply_by_id[supply_ref].midpoint)
                if supply_ref is not None
                else None
            ),
            nearest_demand=(
                float(demand_by_id[demand_ref].midpoint)
                if demand_ref is not None
                else None
            ),
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


def validate_canonical_response(
    raw: dict[str, Any],
    catalog: tuple[EvidenceCatalogItem, ...],
    supply_catalog: tuple[Any, ...] = (),
    demand_catalog: tuple[Any, ...] = (),
    *,
    retry: bool = False,
) -> CanonicalResponseValidation:
    """Normalize once, then validate the canonical wire and semantic contracts."""

    normalized, normalization_details = normalize_compact_output_shapes(
        raw,
        retry=retry,
    )
    model = canonical_response_model(retry=retry)
    validated = model.model_validate(normalized)
    if not isinstance(validated, AuthoritativeAIResponse):
        raise TypeError("canonical response model returned an unexpected type")
    wire = validated
    validate_evidence_references(wire, catalog)
    validate_zone_references(wire, supply_catalog, demand_catalog)
    resolved = resolve_compact_output(
        wire,
        catalog,
        supply_catalog,
        demand_catalog,
    )
    return CanonicalResponseValidation(
        provider_output=raw,
        normalized_output=wire.model_dump(mode="json"),
        wire_output=wire,
        resolved_output=resolved,
        normalization_details=normalization_details,
        evidence_ref_truncations=(),
    )
