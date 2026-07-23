"""Bounded structured market memory used by the LLM and monitoring service."""

from __future__ import annotations

from .models import MarketMemoryEntry, MarketMemorySummary


class MarketMemory:
    def __init__(self, maximum_entries: int) -> None:
        if maximum_entries < 1:
            raise ValueError("market memory must retain at least one entry")
        self.maximum_entries = maximum_entries

    def summarize(self, entries: tuple[MarketMemoryEntry, ...]) -> MarketMemorySummary:
        bounded = tuple(sorted(entries, key=lambda item: item.occurred_at)[-self.maximum_entries :])

        def summaries(*categories: str) -> tuple[str, ...]:
            return tuple(item.summary for item in bounded if item.category in categories)

        active = next((item for item in reversed(bounded) if item.opportunity_key or item.signal_id), None)
        latest_payload = active.structured_payload if active else {}
        return MarketMemorySummary(
            entry_count=len(bounded),
            window_started_at=bounded[0].occurred_at if bounded else None,
            window_ended_at=bounded[-1].occurred_at if bounded else None,
            regime_transitions=summaries("regime_transition"),
            structure_changes=summaries("structure_change", "bos", "choch", "mss"),
            liquidity_events=summaries("liquidity_event"),
            forecast_changes=summaries("quant_forecast", "ai_forecast"),
            evidence_changes=summaries("evidence_change"),
            signal_state_changes=summaries("signal_state_change"),
            completed_outcomes=summaries("signal_outcome", "forecast_outcome"),
            repeated_model_mistakes=summaries("model_mistake"),
            active_opportunity_key=active.opportunity_key if active else None,
            active_signal_state=str(latest_payload.get("signal_state")) if latest_payload.get("signal_state") else None,
            previous_levels=dict(latest_payload.get("levels", {})) if isinstance(latest_payload.get("levels"), dict) else {},
            session_context=str(latest_payload.get("session")) if latest_payload.get("session") else None,
        )
