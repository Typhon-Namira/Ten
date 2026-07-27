"""Strict validation of untrusted analysis-only provider output."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import logging
from typing import Any

from pydantic import ValidationError

from .analysis import AIAnalysisOutput

logger = logging.getLogger(__name__)
_MAX_NORMALIZED_JSON_LOG_CHARACTERS = 8_000


def _normalized_json_for_log(value: dict[str, Any]) -> tuple[str, bool]:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return (
        encoded[:_MAX_NORMALIZED_JSON_LOG_CHARACTERS],
        len(encoded) > _MAX_NORMALIZED_JSON_LOG_CHARACTERS,
    )


@dataclass(frozen=True)
class StructuredValidationIssue:
    field_path: str
    expected_type: str
    actual_value: Any
    validator_name: str
    offending_json_fragment: str
    recoverable: bool = False

    def encoded(self) -> str:
        return json.dumps(
            {
                "field_path": self.field_path,
                "expected_type": self.expected_type,
                "actual_value": self.actual_value,
                "validator_name": self.validator_name,
                "offending_json_fragment": self.offending_json_fragment,
                "recoverable": self.recoverable,
            },
            default=str,
            separators=(",", ":"),
        )


class StructuredAIOutputError(ValueError):
    def __init__(
        self,
        errors: tuple[str, ...],
        *,
        first_issue: StructuredValidationIssue | None = None,
    ) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors
        self.first_issue = first_issue


class StructuredAIOutputValidator:
    """The only executable provider-output validator.

    It accepts deep market analysis and has no proposal or signal normalization
    path. Pydantic's strict ``extra='forbid'`` models reject every trading-action
    field.
    """

    def validate_analysis(self, raw: dict[str, Any]) -> AIAnalysisOutput:
        try:
            validated = AIAnalysisOutput.model_validate(raw)
        except ValidationError as exc:
            issues = tuple(
                self._pydantic_issue("provider_response", item, raw)
                for item in exc.errors()
            )
            logger.warning(
                "ai_provider.response.normalization.skipped",
                extra={
                    "normalization_status": "analysis_schema_invalid",
                    "validation_error_count": len(issues),
                    "field_path": issues[0].field_path if issues else None,
                    "expected_type": issues[0].expected_type if issues else None,
                    "actual_value": issues[0].actual_value if issues else None,
                    "validator_name": issues[0].validator_name if issues else None,
                    "offending_json_fragment": (
                        issues[0].offending_json_fragment if issues else None
                    ),
                },
            )
            raise StructuredAIOutputError(
                tuple(item.encoded() for item in issues),
                first_issue=issues[0] if issues else None,
            ) from exc
        normalized_json, truncated = _normalized_json_for_log(
            validated.model_dump(mode="python")
        )
        logger.info(
            "ai_provider.response.normalized",
            extra={
                "normalization_status": "analysis_schema_valid",
                "normalized_provider_json": normalized_json,
                "normalized_provider_json_truncated": truncated,
            },
        )
        return validated

    @staticmethod
    def _pydantic_issue(
        prefix: str,
        error: Mapping[str, Any],
        raw: dict[str, Any],
    ) -> StructuredValidationIssue:
        location = tuple(str(item) for item in error.get("loc", ()))
        current: Any = raw
        for item in error.get("loc", ()):
            if isinstance(current, dict):
                current = current.get(item)
            elif isinstance(current, (list, tuple)) and isinstance(item, int):
                current = current[item] if 0 <= item < len(current) else None
            else:
                current = None
                break
        field_path = ".".join((prefix, *location))
        fragment, _ = _normalized_json_for_log(
            {"field": field_path, "value": current}
        )
        return StructuredValidationIssue(
            field_path=field_path,
            expected_type=str(error.get("msg") or error.get("type") or "valid value"),
            actual_value=current,
            validator_name=str(error.get("type") or "pydantic"),
            offending_json_fragment=fragment,
        )
