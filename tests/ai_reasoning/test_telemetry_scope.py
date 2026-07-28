from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from backend.app.ai_reasoning.telemetry import (
    current_operational_usage,
    provider_attempts,
    usage_parameter,
)

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)


def row(
    *,
    deployment: str | None = "deploy-current",
    schema: str | None = "compact-1.1",
    profile: str | None = "compact",
    prompt: str = "deep_market_analysis_v2",
    created_at: datetime = NOW,
    attempts: tuple[dict[str, object], ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        created_at=created_at,
        prompt_version=prompt,
        generation_parameters={
            "telemetry_policy": "five_minute_v1",
            "deployment_id": deployment,
            "analysis_schema_version": schema,
            "output_profile": profile,
            "analysis_requests": 1,
            "initial_parse_failures": 1,
            "provider_attempts": attempts,
        },
    )


def test_current_operational_scope_excludes_incompatible_history() -> None:
    current = row()
    rows = (
        current,
        row(deployment="deploy-old"),
        row(schema="compact-1.0"),
        row(profile=None),
        row(prompt="old_prompt"),
        row(created_at=NOW - timedelta(hours=25)),
    )

    selected = current_operational_usage(
        rows,
        deployment_id="deploy-current",
        prompt_version="deep_market_analysis_v2",
        now=NOW,
    )

    assert selected == (current,)
    assert usage_parameter(selected, "analysis_requests") == 1
    assert usage_parameter(rows, "analysis_requests") == 6


def test_model_probes_do_not_enter_analysis_attempt_metrics() -> None:
    rows = (
        row(
            attempts=(
                {"request_kind": "model_probe", "schema_error_code": "json_parse_error"},
                {"request_kind": "analysis", "schema_error_code": None},
            )
        ),
    )

    assert provider_attempts(rows) == (
        {"request_kind": "analysis", "schema_error_code": None},
    )
