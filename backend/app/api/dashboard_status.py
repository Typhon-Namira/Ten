"""Backend-authoritative derivation of AI-pipeline dashboard stage statuses.

Root cause this exists to fix: `dashboard.py` used to derive the `ai_reasoning` stage purely
from "does a persisted `AIMarketForecast` row exist for this market state" — `None` always meant
`status: "pending", reason: "ai_reasoning_not_yet_persisted_for_cycle"`, with no way to tell a
genuinely-fresh cycle apart from one whose provider attempt already failed. The same collapse-to-"pending"
problem existed one level down: historical `guardrails`/`final_action` records could show
`"awaiting_ai_proposal"` forever when a legacy AI result was non-actionable. That terminal
historical outcome was indistinguishable from a step that had not run yet.

Every function here is pure (no I/O, no `Request`/session access) so it can be unit-tested without
a database or FastAPI app; `dashboard.py` is the only caller and owns fetching the rows these
functions are handed. Attribute access uses `getattr(..., default)` throughout, matching this
route's existing convention (see `_record_status`/`_stage` in `dashboard.py`) of tolerating both
real Pydantic models and the loosely-typed `SimpleNamespace` test doubles already used across the
dashboard test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Any

TERMINAL_FAILURE_STATUS_VALUES = frozenset({"unavailable", "invalid", "failed"})
ACTIONABLE_FINAL_ACTION_VALUES = frozenset({"approved", "approved_with_reduced_risk", "published"})


@dataclass(frozen=True)
class StageResult:
    status: str
    reason: str
    error_code: str | None = None
    retryable: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _value(obj: Any) -> str:
    """Normalize an enum member, a plain string, or `None` to a lowercase string."""
    return str(getattr(obj, "value", obj) or "").lower()


def _forecast_failure_detail(forecast: Any) -> tuple[str, dict[str, Any]]:
    """The exact terminal reason for a failed/invalid/unavailable forecast.

    `StructuredAIOutputError.errors` (ai_reasoning/validation.py) is a tuple of
    `StructuredValidationIssue.encoded()` JSON strings — the *only* place field/expected/received
    survive past the log line that originally reported them — and lands unchanged in
    `AIMarketForecast.missing_evidence[0]` via `_unavailable_forecast()`. Parsing it back out here
    is what lets a structured-output failure report "Field: forecast, Expected: AIMarketForecast
    object, Received: string" instead of the bare `failure_state` code.
    """
    extra: dict[str, Any] = {}
    failure_state = getattr(forecast, "failure_state", None)
    missing_evidence = getattr(forecast, "missing_evidence", None) or ()
    if failure_state == "structured_output_invalid" and missing_evidence:
        try:
            issue = json.loads(missing_evidence[0])
        except (json.JSONDecodeError, TypeError, IndexError):
            issue = None
        if isinstance(issue, dict) and "field_path" in issue:
            extra["field"] = issue.get("field_path")
            extra["expected"] = issue.get("expected_type")
            extra["received"] = issue.get("actual_value")
            return "structured_response_validation_failed", extra
    provider_http_status = getattr(forecast, "provider_http_status", None)
    if provider_http_status is not None:
        extra["provider_http_status"] = provider_http_status
        provider = getattr(forecast, "model_provider", None) or "ai_provider"
        return f"{provider}_returned_http_{provider_http_status}", extra
    return failure_state or "ai_reasoning_failed", extra


def derive_ai_reasoning_stage(
    *,
    forecast: Any | None,
    request: Any | None,
    ai_health: dict[str, Any],
    now: datetime,
    cycle_available_at: datetime,
) -> StageResult:
    """The full terminal-state machine required for the `ai_reasoning` stage.

    Order matters: a persisted forecast (success or terminal failure) is always authoritative over
    any in-memory health signal, since the health object is process-local and resets on restart
    while the persisted row does not.
    """
    if forecast is not None:
        if _value(getattr(forecast, "status", None)) in TERMINAL_FAILURE_STATUS_VALUES:
            reason, extra = _forecast_failure_detail(forecast)
            return StageResult("failed", reason, error_code=getattr(forecast, "failure_state", None), retryable=True, extra=extra)
        if getattr(forecast, "validation_passed", True) is False:
            return StageResult("degraded", "degraded_structured_output", extra={"repaired_fields": True})
        return StageResult("available", "same_cycle_ai_reasoning_persisted")

    if not ai_health.get("enabled"):
        disabled_flags = tuple(name for name in ("shadow_enabled", "proposals_enabled", "monitoring_enabled") if not ai_health.get(name))
        return StageResult("disabled", "ai_reasoning_disabled", extra={"disabled_flags": disabled_flags})

    provider_states = ai_health.get("providers") or {}
    configured_states = [
        item
        for item in provider_states.values()
        if isinstance(item, dict) and item.get("status") != "UNCONFIGURED"
    ]
    circuit_deadlines = [
        deadline
        for item in configured_states
        for deadline in (_parse_iso(item.get("circuit_open_until")),)
        if deadline is not None and now < deadline
    ]
    if configured_states and len(circuit_deadlines) == len(configured_states):
        backoff_until = min(circuit_deadlines)
        retry_in_seconds = max(0.0, (backoff_until - now).total_seconds())
        return StageResult(
            "blocked",
            "provider_backoff_active",
            error_code=ai_health.get("failure_state"),
            retryable=True,
            extra={"retry_in_seconds": retry_in_seconds, "last_failure_state": ai_health.get("failure_state")},
        )

    if request is not None:
        if getattr(request, "compatibility_status", "compatible") == "incompatible":
            return StageResult(
                "failed",
                "ai_request_history_schema_incompatible",
                error_code=getattr(request, "compatibility_reason", None),
                retryable=False,
                extra={
                    "payload_format": getattr(request, "payload_format", "incompatible"),
                    "payload_schema_version": getattr(request, "payload_schema_version", None),
                },
            )
        created_at = getattr(request, "created_at", now)
        elapsed_seconds = max(0.0, (now - created_at).total_seconds())
        return StageResult("running", "ai_provider_request_in_progress", extra={"elapsed_seconds": elapsed_seconds, "job_state": "running"})

    elapsed_seconds = max(0.0, (now - cycle_available_at).total_seconds())
    return StageResult("pending", "queued_awaiting_worker", extra={"elapsed_seconds": elapsed_seconds, "job_state": "queued"})


def derive_ai_proposal_stage(
    *,
    forecast: Any | None,
    proposal: Any | None,
    proposals_enabled: bool,
) -> StageResult:
    if proposal is not None:
        return StageResult("available", "same_cycle_ai_proposal_persisted")
    if forecast is None:
        return StageResult("not_available", "awaiting_ai_reasoning")
    if _value(getattr(forecast, "status", None)) in TERMINAL_FAILURE_STATUS_VALUES:
        return StageResult("not_available", "ai_reasoning_failed")
    if not proposals_enabled:
        return StageResult("not_available", "ai_signal_proposals_disabled")
    return StageResult("not_available", "ai_reasoning_produced_no_proposal")


def no_actionable_proposal(proposal: Any | None) -> bool:
    """Treat missing and historical WAIT proposals as non-actionable dashboard records."""
    return proposal is None or _value(getattr(proposal, "recommended_action", None)) == "wait"


def derive_guardrails_stage(
    *,
    forecast: Any | None,
    proposal: Any | None,
    action: Any | None,
) -> StageResult:
    if action is not None:
        return StageResult("available", "same_cycle_guardrail_result_persisted")
    if forecast is None:
        return StageResult("not_evaluated", "awaiting_ai_reasoning")
    if _value(getattr(forecast, "status", None)) in TERMINAL_FAILURE_STATUS_VALUES:
        return StageResult("not_evaluated", "ai_reasoning_failed")
    if no_actionable_proposal(proposal):
        return StageResult("not_required", "no_proposal_to_evaluate")
    return StageResult("not_evaluated", "guardrail_result_not_yet_persisted")


def derive_final_action_stage(
    *,
    forecast: Any | None,
    proposal: Any | None,
    action: Any | None,
) -> StageResult:
    if action is not None:
        return StageResult("available", "same_cycle_guardrail_result_persisted")
    if forecast is None:
        return StageResult("not_available", "awaiting_ai_reasoning")
    if _value(getattr(forecast, "status", None)) in TERMINAL_FAILURE_STATUS_VALUES:
        reason, extra = _forecast_failure_detail(forecast)
        return StageResult(
            "hold",
            "ai_provider_unavailable",
            error_code=getattr(forecast, "failure_state", None),
            extra={
                "direction": "HOLD",
                "publication_eligible": False,
                "upstream_reason": reason,
                **extra,
            },
        )
    if no_actionable_proposal(proposal):
        reason = "ai_proposal_recommended_wait" if proposal is not None else "ai_reasoning_produced_no_proposal"
        return StageResult("hold", reason, extra={"direction": "HOLD"})
    return StageResult("not_available", "guardrail_result_not_yet_persisted")


def derive_publication_stage(
    *,
    publication: Any | None,
    publication_enabled: bool,
    publication_config_source: str,
) -> StageResult:
    if publication is not None:
        return StageResult("available", "analytical_signal_persisted")
    if not publication_enabled:
        return StageResult("disabled", "ai_signal_publication_disabled", extra={"config_source": publication_config_source})
    return StageResult("not_available", "final_action_not_publication_eligible")


def derive_monitoring_stage(
    *,
    signal: Any | None,
    final_action_status: str,
    action: Any | None,
    publication: Any | None,
    publication_enabled: bool,
    monitoring_enabled: bool,
) -> StageResult:
    if not monitoring_enabled:
        return StageResult("not_available", "ai_signal_monitoring_disabled")
    if signal is not None:
        signal_state = _value(getattr(signal, "state", None)) or "unknown"
        if publication is None and signal_state == "active":
            return StageResult(
                "blocked",
                "active_signal_missing_publication",
                extra={"managed_signal_state": signal_state},
            )
        if publication is None:
            reason = (
                "unpublished_signal_observed_in_shadow"
                if not publication_enabled
                else "proposed_signal_monitored_while_publication_ineligible"
            )
            return StageResult(
                "available",
                reason,
                extra={"managed_signal_state": signal_state},
            )
        return StageResult(
            "available",
            "published_managed_signal_monitored",
            extra={"managed_signal_state": signal_state},
        )
    if final_action_status == "hold":
        return StageResult("not_required", "latest_final_action_non_actionable_hold")
    if action is not None and _value(getattr(action, "action", None)) in ACTIONABLE_FINAL_ACTION_VALUES and publication is None and not publication_enabled:
        return StageResult("blocked", "signal_publication_disabled")
    return StageResult("not_available", "no_managed_signal_for_latest_cycle")


def derive_outcome_stage(
    *,
    outcome: Any | None,
    final_action_status: str,
    action: Any | None,
    publication: Any | None,
    publication_enabled: bool,
) -> StageResult:
    if outcome is not None:
        return StageResult("available", "signal_outcome_persisted")
    if final_action_status == "hold":
        return StageResult("not_applicable", "no_managed_signal_opened_for_cycle")
    if action is not None and _value(getattr(action, "action", None)) in ACTIONABLE_FINAL_ACTION_VALUES and publication is None and not publication_enabled:
        return StageResult("not_started", "no_published_signal_to_track_outcome_for")
    return StageResult("not_evaluated", "evaluation_horizon_not_complete")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
