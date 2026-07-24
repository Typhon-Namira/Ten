"""Unit coverage for backend.app.api.dashboard_status — the pure terminal-state machine that
replaced dashboard.py's old "does a row exist yet -> pending" collapse for the AI-centric stages.

No database, no FastAPI app: every function under test takes plain objects (SimpleNamespace
stands in for AIMarketForecast/AISignalProposal/FinalSystemAction, matching this route's existing
test convention) and returns a StageResult.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from backend.app.api.dashboard_status import (
    derive_ai_proposal_stage,
    derive_ai_reasoning_stage,
    derive_final_action_stage,
    derive_guardrails_stage,
    derive_monitoring_stage,
    derive_outcome_stage,
    derive_publication_stage,
    no_actionable_proposal,
)

NOW = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)


def forecast(**overrides):
    base = dict(
        status="available",
        validation_passed=True,
        failure_state=None,
        missing_evidence=(),
        provider_http_status=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def proposal(**overrides):
    base = dict(recommended_action="BUY")
    base.update(overrides)
    return SimpleNamespace(**base)


def action(**overrides):
    base = dict(action="approved")
    base.update(overrides)
    return SimpleNamespace(**base)


def healthy_ai_health(**overrides):
    base = dict(enabled=True, shadow_enabled=True, proposals_enabled=True, monitoring_enabled=True, provider_backoff_until=None, failure_state=None)
    base.update(overrides)
    return base


# --- ai_reasoning stage ------------------------------------------------------------------------


def test_fully_validated_ai_reasoning_is_reported_available_not_degraded():
    result = derive_ai_reasoning_stage(
        forecast=forecast(status="available", validation_passed=True),
        request=None,
        ai_health=healthy_ai_health(),
        now=NOW,
        cycle_available_at=NOW,
    )
    assert result.status == "available"
    assert result.reason == "same_cycle_ai_reasoning_persisted"


def test_terminal_provider_failure_reports_failed_not_pending():
    """Item 11 / primary production bug: a forecast row exists with a terminal failure status —
    the stage must report "failed", never fall back to "pending"."""
    result = derive_ai_reasoning_stage(
        forecast=forecast(status="unavailable", failure_state="openrouter_authentication_failed", provider_http_status=401),
        request=None,
        ai_health=healthy_ai_health(),
        now=NOW,
        cycle_available_at=NOW - timedelta(minutes=5),
    )
    assert result.status == "failed"
    assert result.reason == "openrouter_returned_http_401"
    assert result.error_code == "openrouter_authentication_failed"
    assert result.retryable is True
    assert result.extra["provider_http_status"] == 401


def test_structured_output_failure_exposes_exact_field_expected_received():
    """Item 4: a structured-output validation failure must expose field/expected/received, not
    just the bare failure_state code."""
    import json

    issue_json = json.dumps({"field_path": "forecast", "expected_type": "AIMarketForecast object", "actual_value": "string"})
    result = derive_ai_reasoning_stage(
        forecast=forecast(status="invalid", failure_state="structured_output_invalid", missing_evidence=(issue_json,)),
        request=None,
        ai_health=healthy_ai_health(),
        now=NOW,
        cycle_available_at=NOW - timedelta(minutes=5),
    )
    assert result.status == "failed"
    assert result.reason == "structured_response_validation_failed"
    assert result.extra["field"] == "forecast"
    assert result.extra["expected"] == "AIMarketForecast object"
    assert result.extra["received"] == "string"


def test_disabled_service_reports_disabled_with_flag_names():
    result = derive_ai_reasoning_stage(
        forecast=None,
        request=None,
        ai_health=healthy_ai_health(enabled=False, shadow_enabled=False, proposals_enabled=False, monitoring_enabled=False),
        now=NOW,
        cycle_available_at=NOW,
    )
    assert result.status == "disabled"
    assert result.reason == "ai_reasoning_disabled"
    assert set(result.extra["disabled_flags"]) == {"shadow_enabled", "proposals_enabled", "monitoring_enabled"}


def test_active_provider_backoff_reports_blocked_with_retry_time_not_pending():
    """Item 3 / the exact root cause: every cycle that lands inside an active backoff window is
    skipped by AIReasoningService.process() before it ever persists a row. Without this branch,
    that skip was indistinguishable from "hasn't started yet"."""
    backoff_until = NOW + timedelta(seconds=45)
    result = derive_ai_reasoning_stage(
        forecast=None,
        request=None,
        ai_health=healthy_ai_health(provider_backoff_until=backoff_until.isoformat(), failure_state="openrouter_authentication_failed"),
        now=NOW,
        cycle_available_at=NOW - timedelta(seconds=10),
    )
    assert result.status == "blocked"
    assert result.reason == "provider_backoff_active"
    assert result.error_code == "openrouter_authentication_failed"
    assert result.retryable is True
    assert 44 <= result.extra["retry_in_seconds"] <= 45


def test_persisted_request_without_forecast_reports_running_with_elapsed_time():
    result = derive_ai_reasoning_stage(
        forecast=None,
        request=SimpleNamespace(created_at=NOW - timedelta(seconds=7)),
        ai_health=healthy_ai_health(),
        now=NOW,
        cycle_available_at=NOW - timedelta(seconds=7),
    )
    assert result.status == "running"
    assert result.reason == "openrouter_request_in_progress"
    assert result.extra["elapsed_seconds"] == 7
    assert result.extra["job_state"] == "running"


def test_incompatible_request_history_reports_explicit_failure_not_http_breaking_exception():
    result = derive_ai_reasoning_stage(
        forecast=None,
        request=SimpleNamespace(
            created_at=NOW - timedelta(seconds=7),
            compatibility_status="incompatible",
            compatibility_reason="unsupported_persisted_request_schema_version",
            payload_format="incompatible",
            payload_schema_version="99.0",
        ),
        ai_health=healthy_ai_health(),
        now=NOW,
        cycle_available_at=NOW - timedelta(seconds=7),
    )
    assert result.status == "failed"
    assert result.reason == "ai_request_history_schema_incompatible"
    assert result.error_code == "unsupported_persisted_request_schema_version"
    assert result.extra == {
        "payload_format": "incompatible",
        "payload_schema_version": "99.0",
    }


def test_genuinely_fresh_cycle_reports_pending_with_job_state_and_elapsed_time():
    """The only case "pending" may legitimately appear — and even then it must carry job state
    and elapsed waiting time, not just a bare label."""
    result = derive_ai_reasoning_stage(
        forecast=None,
        request=None,
        ai_health=healthy_ai_health(),
        now=NOW,
        cycle_available_at=NOW - timedelta(seconds=2),
    )
    assert result.status == "pending"
    assert result.reason == "queued_awaiting_worker"
    assert result.extra["job_state"] == "queued"
    assert result.extra["elapsed_seconds"] == 2


def test_degraded_structured_output_is_not_reported_as_a_plain_failure():
    result = derive_ai_reasoning_stage(
        forecast=forecast(status="available", validation_passed=False),
        request=None,
        ai_health=healthy_ai_health(),
        now=NOW,
        cycle_available_at=NOW,
    )
    assert result.status == "degraded"


# --- ai_proposal stage -------------------------------------------------------------------------


def test_proposal_stage_does_not_claim_no_proposal_before_reasoning_even_ran():
    result = derive_ai_proposal_stage(forecast=None, proposal=None, proposals_enabled=True)
    assert result.status == "not_available"
    assert result.reason == "awaiting_ai_reasoning"


def test_proposal_stage_reports_ai_reasoning_failed_when_forecast_failed():
    result = derive_ai_proposal_stage(forecast=forecast(status="failed"), proposal=None, proposals_enabled=True)
    assert result.reason == "ai_reasoning_failed"


# --- guardrails / final_action ------------------------------------------------------------------


def test_guardrails_not_required_when_forecast_valid_and_no_proposal():
    """Items 6/7 — the core required behavior: valid forecast + no proposal -> guardrails
    not_required, final_action = wait."""
    result = derive_guardrails_stage(forecast=forecast(), proposal=None, action=None)
    assert result.status == "not_required"
    assert result.reason == "no_proposal_to_evaluate"


def test_guardrails_not_required_when_proposal_recommends_wait():
    result = derive_guardrails_stage(forecast=forecast(), proposal=proposal(recommended_action="WAIT"), action=None)
    assert result.status == "not_required"


def test_final_action_is_wait_when_forecast_valid_and_no_proposal():
    result = derive_final_action_stage(forecast=forecast(), proposal=None, action=None)
    assert result.status == "wait"
    assert result.extra["direction"] == "WAIT"


def test_final_action_is_wait_when_proposal_recommends_wait():
    result = derive_final_action_stage(forecast=forecast(), proposal=proposal(recommended_action="WAIT"), action=None)
    assert result.status == "wait"
    assert result.reason == "ai_proposal_recommended_wait"


def test_final_action_never_says_awaiting_ai_proposal_after_a_terminal_ai_failure():
    """Item 11 at this layer: once ai_reasoning has terminally failed, final_action must not sit
    in "awaiting" anything — it is blocked by a resolved upstream failure."""
    result = derive_final_action_stage(
        forecast=forecast(status="unavailable", failure_state="openrouter_authentication_failed"),
        proposal=None,
        action=None,
    )
    assert result.status == "blocked"
    assert result.reason == "ai_reasoning_failed"
    assert result.error_code == "openrouter_authentication_failed"
    assert "awaiting" not in result.reason


def test_final_action_available_when_a_real_action_was_persisted():
    result = derive_final_action_stage(forecast=forecast(), proposal=proposal(), action=action())
    assert result.status == "available"


def test_no_actionable_proposal_helper():
    assert no_actionable_proposal(None) is True
    assert no_actionable_proposal(proposal(recommended_action="WAIT")) is True
    assert no_actionable_proposal(proposal(recommended_action="BUY")) is False


# --- publication ---------------------------------------------------------------------------------


def test_publication_disabled_exposes_exact_config_source():
    result = derive_publication_stage(publication=None, publication_enabled=False, publication_config_source="environment variable TEN_AI_SIGNAL_PUBLICATION")
    assert result.status == "disabled"
    assert result.extra["config_source"] == "environment variable TEN_AI_SIGNAL_PUBLICATION"


def test_publication_disabled_status_is_independent_of_upstream_stage_results():
    """Item 8: publication being disabled is computed purely from its own inputs — it cannot,
    even accidentally, gate ai_reasoning/ai_proposal/guardrails/final_action, since those are
    derived by entirely separate functions that never read publication_enabled."""
    reasoning = derive_ai_reasoning_stage(forecast=forecast(), request=None, ai_health=healthy_ai_health(), now=NOW, cycle_available_at=NOW)
    final_action = derive_final_action_stage(forecast=forecast(), proposal=proposal(), action=action())
    assert reasoning.status == "available"
    assert final_action.status == "available"


# --- monitoring / outcome -------------------------------------------------------------------------


def test_monitoring_not_required_for_wait_final_action():
    """Item 9."""
    result = derive_monitoring_stage(
        signal=None,
        final_action_status="wait",
        action=None,
        publication=None,
        publication_enabled=False,
        monitoring_enabled=True,
    )
    assert result.status == "not_required"
    assert result.reason == "latest_final_action_non_actionable_wait"


def test_outcome_not_applicable_when_no_signal_was_ever_opened():
    """Item 10."""
    result = derive_outcome_stage(
        outcome=None,
        final_action_status="wait",
        action=None,
        publication=None,
        publication_enabled=False,
    )
    assert result.status == "not_applicable"
    assert result.reason == "no_managed_signal_opened_for_cycle"


def test_monitoring_blocked_when_actionable_action_exists_but_publication_disabled():
    result = derive_monitoring_stage(
        signal=None,
        final_action_status="available",
        action=action(action="approved"),
        publication=None,
        publication_enabled=False,
        monitoring_enabled=True,
    )
    assert result.status == "blocked"
    assert result.reason == "signal_publication_disabled"


def test_outcome_not_started_when_actionable_action_exists_but_publication_disabled():
    result = derive_outcome_stage(
        outcome=None,
        final_action_status="available",
        action=action(action="approved"),
        publication=None,
        publication_enabled=False,
    )
    assert result.status == "not_started"


def test_monitoring_active_when_signal_exists_regardless_of_final_action_status():
    result = derive_monitoring_stage(
        signal=SimpleNamespace(signal_id="s1"),
        final_action_status="available",
        action=action(),
        publication=None,
        publication_enabled=True,
        monitoring_enabled=True,
    )
    assert result.status == "available"
    assert result.reason == "managed_signal_active"


def test_monitoring_disabled_flag_takes_precedence():
    result = derive_monitoring_stage(
        signal=None,
        final_action_status="wait",
        action=None,
        publication=None,
        publication_enabled=False,
        monitoring_enabled=False,
    )
    assert result.status == "not_available"
    assert result.reason == "ai_signal_monitoring_disabled"
