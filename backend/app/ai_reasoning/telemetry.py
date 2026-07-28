"""Compatible operational windows for AI provider telemetry."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

CURRENT_COMPACT_SCHEMAS = {"compact-1.1", "compact-retry-1.1"}
CURRENT_OUTPUT_PROFILES = {"compact", "compact_retry"}


def current_operational_usage(
    rows: Iterable[Any],
    *,
    deployment_id: str,
    prompt_version: str,
    now: datetime,
) -> tuple[Any, ...]:
    cutoff = now - timedelta(hours=24)
    return tuple(
        row
        for row in rows
        if row.created_at >= cutoff
        and row.prompt_version == prompt_version
        and row.generation_parameters.get("telemetry_policy") == "five_minute_v1"
        and row.generation_parameters.get("deployment_id") == deployment_id
        and row.generation_parameters.get("analysis_schema_version")
        in CURRENT_COMPACT_SCHEMAS
        and row.generation_parameters.get("output_profile")
        in CURRENT_OUTPUT_PROFILES
    )


def usage_parameter(rows: Iterable[Any], name: str) -> int:
    return sum(
        int(value)
        for row in rows
        if isinstance(
            (value := row.generation_parameters.get(name, 0)),
            (int, float),
        )
    )


def provider_attempts(rows: Iterable[Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        attempt
        for row in rows
        for attempt in row.generation_parameters.get("provider_attempts", ())
        if isinstance(attempt, dict)
        and attempt.get("request_kind") != "model_probe"
    )
