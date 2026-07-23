"""add Phase 2 quantitative forecasting shadow infrastructure

Revision ID: 20260723_0003
Revises: 20260723_0002
Create Date: 2026-07-23
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260723_0003"
down_revision: str | None = "20260723_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "quant_forecast_model_metadata",
        sa.Column("model_name", sa.String(96), primary_key=True),
        sa.Column("model_version", sa.String(48), primary_key=True),
        sa.Column("payload", JSON, nullable=False),
    )
    op.create_table(
        "quant_feature_vectors",
        sa.Column("vector_id", sa.UUID(), primary_key=True),
        sa.Column("market_state_id", sa.UUID(), sa.ForeignKey("unified_market_states.state_id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("point_in_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("schema_version", sa.String(64), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_quant_feature_vectors_series_boundary", "quant_feature_vectors", ["instrument", "point_in_time"])
    op.create_table(
        "quant_feature_references",
        sa.Column("vector_id", sa.UUID(), sa.ForeignKey("quant_feature_vectors.vector_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("feature_name", sa.String(128), primary_key=True),
        sa.Column("evidence_id", sa.UUID(), sa.ForeignKey("evidence_items.evidence_id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("source_paths", JSON, nullable=False),
    )
    op.create_table(
        "quant_forecast_requests",
        sa.Column("request_id", sa.UUID(), primary_key=True),
        sa.Column("market_state_id", sa.UUID(), sa.ForeignKey("unified_market_states.state_id", ondelete="CASCADE"), nullable=False),
        sa.Column("cycle_id", sa.UUID(), nullable=False),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("point_in_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_name", sa.String(96), nullable=False),
        sa.Column("model_version", sa.String(48), nullable=False),
        sa.Column("mode", sa.String(16), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_quant_forecast_requests_series_boundary", "quant_forecast_requests", ["instrument", "point_in_time"])
    op.create_table(
        "quantitative_forecasts",
        sa.Column("result_id", sa.UUID(), primary_key=True),
        sa.Column("request_id", sa.UUID(), sa.ForeignKey("quant_forecast_requests.request_id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("market_state_id", sa.UUID(), sa.ForeignKey("unified_market_states.state_id", ondelete="CASCADE"), nullable=False),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("point_in_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("model_name", sa.String(96), nullable=False),
        sa.Column("model_version", sa.String(48), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_quantitative_forecasts_series_boundary", "quantitative_forecasts", ["instrument", "point_in_time"])
    op.create_table(
        "quantitative_forecast_horizons",
        sa.Column("result_id", sa.UUID(), sa.ForeignKey("quantitative_forecasts.result_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("horizon_id", sa.String(24), primary_key=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=False),
        sa.Column("payload", JSON, nullable=False),
    )
    op.create_table(
        "quant_forecast_outcomes",
        sa.Column("outcome_id", sa.UUID(), primary_key=True),
        sa.Column("result_id", sa.UUID(), sa.ForeignKey("quantitative_forecasts.result_id", ondelete="CASCADE"), nullable=False),
        sa.Column("horizon_id", sa.String(24), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("result_id", "horizon_id", name="ux_quant_forecast_outcomes_result_horizon"),
    )
    op.create_table(
        "quant_calibration_runs",
        sa.Column("report_id", sa.UUID(), primary_key=True),
        sa.Column("model_name", sa.String(96), nullable=False),
        sa.Column("model_version", sa.String(48), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "quant_calibration_buckets",
        sa.Column("report_id", sa.UUID(), sa.ForeignKey("quant_calibration_runs.report_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("ordinal", sa.Integer(), primary_key=True),
        sa.Column("horizon_id", sa.String(24), nullable=False),
        sa.Column("dimension", sa.String(48), nullable=False),
        sa.Column("payload", JSON, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("quant_calibration_buckets")
    op.drop_table("quant_calibration_runs")
    op.drop_table("quant_forecast_outcomes")
    op.drop_table("quantitative_forecast_horizons")
    op.drop_table("quantitative_forecasts")
    op.drop_table("quant_forecast_requests")
    op.drop_table("quant_feature_references")
    op.drop_table("quant_feature_vectors")
    op.drop_table("quant_forecast_model_metadata")
