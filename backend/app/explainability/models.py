"""Grounded context and explanation contracts.

`ExplainabilityContext` is the ONLY thing the LLM ever sees — it is assembled entirely from
already-persisted TEN state (see context.py). `Explanation` is the ONLY shape the LLM is allowed
to return (enforced via provider `response_format: json_object` plus Pydantic validation on
the way back in) — free-form prose replies are never trusted or rendered as-is.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """One citable fact TEN actually computed — never an LLM-invented reference."""

    source: str
    """Which engine/record this came from, e.g. "smc_snapshot", "ai_score", "decision"."""
    reference_id: str
    timestamp: datetime | None = None


class EngineFact(BaseModel):
    """One engine's contribution to the grounding bundle. `available=False` is itself a fact the
    explanation must be able to state plainly ("Institutional Flow has no snapshot yet") instead
    of silently omitting the engine."""

    engine: str
    available: bool
    summary: dict[str, Any] = Field(default_factory=dict)
    evidence: Evidence | None = None
    error: str | None = None


class ExplainabilityContext(BaseModel):
    """The complete, self-contained grounding bundle for one explanation request. Nothing outside
    this object is ever available to the model — it cannot browse the database, call an engine, or
    recall anything from a previous request."""

    instrument: str
    timeframe: str
    generated_at: datetime
    engines: list[EngineFact]
    ai_score: dict[str, Any] | None
    decision: dict[str, Any] | None
    pipeline_stage: dict[str, Any] | None
    previous_ai_score: dict[str, Any] | None = None
    question: str | None = None
    """Set only for /explain/chat — the user's own question, included so the model answers what
    was actually asked instead of producing a generic summary."""

    def evidence_list(self) -> list[Evidence]:
        return [fact.evidence for fact in self.engines if fact.evidence is not None]

    def explainability_score(self) -> dict[str, Any]:
        """How much real grounding backed this explanation — computed here, deterministically,
        from the context that was ACTUALLY assembled, never self-reported by the model. This is
        exactly the number the UI's "explanation confidence" meter renders."""
        total = len(self.engines)
        available = sum(1 for item in self.engines if item.available)
        evidence_count = len(self.evidence_list())
        percent = round((available / total) * 100, 1) if total else 0.0
        return {
            "percent": percent,
            "engines_available": available,
            "engines_total": total,
            "evidence_citations": evidence_count,
            "has_ai_score": self.ai_score is not None,
            "has_decision": self.decision is not None,
        }


class EngineInfluence(BaseModel):
    engine: str
    influence: str
    """Short label, e.g. "strong support", "primary blocker", "neutral"."""
    note: str


class Explanation(BaseModel):
    """The model's structured response — validated on the way back in. If the model returns
    anything that doesn't fit this shape, `ExplainabilityService` surfaces that as a degraded
    explanation (`status: "error"`), never a raw/unvalidated blob."""

    summary: str
    primary_reasons: list[str] = Field(default_factory=list)
    opposing_factors: list[str] = Field(default_factory=list)
    engine_breakdown: list[EngineInfluence] = Field(default_factory=list)
    required_for_change: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ChatTurn(BaseModel):
    role: str
    """"user" or "assistant"."""
    content: str
