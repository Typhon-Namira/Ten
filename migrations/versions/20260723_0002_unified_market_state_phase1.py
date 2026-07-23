"""add Unified Market State Phase 1 infrastructure

Revision ID: 20260723_0002
Revises: 20260719_0001
Create Date: 2026-07-23
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260723_0002"
down_revision: str | None = "20260719_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "market_evidence_frames",
        sa.Column("frame_id", sa.UUID(), nullable=False),
        sa.Column("frame_hash", sa.String(length=64), nullable=False),
        sa.Column("instrument", sa.String(length=32), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("candle_close_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("knowledge_cutoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("frame_id"),
    )
    op.create_index("ux_market_evidence_frames_hash", "market_evidence_frames", ["frame_hash"], unique=True)
    op.create_index("ix_market_evidence_frames_series_boundary", "market_evidence_frames", ["instrument", "timeframe", "candle_close_at"])
    op.create_index(op.f("ix_market_evidence_frames_instrument"), "market_evidence_frames", ["instrument"])
    op.create_index(op.f("ix_market_evidence_frames_timeframe"), "market_evidence_frames", ["timeframe"])
    op.create_index(op.f("ix_market_evidence_frames_candle_close_at"), "market_evidence_frames", ["candle_close_at"])
    op.create_index(op.f("ix_market_evidence_frames_knowledge_cutoff"), "market_evidence_frames", ["knowledge_cutoff"])
    op.create_index(op.f("ix_market_evidence_frames_created_at"), "market_evidence_frames", ["created_at"])

    op.create_table(
        "unified_market_states",
        sa.Column("state_id", sa.UUID(), nullable=False),
        sa.Column("state_hash", sa.String(length=64), nullable=False),
        sa.Column("instrument", sa.String(length=32), nullable=False),
        sa.Column("trigger_timeframe", sa.String(length=16), nullable=False),
        sa.Column("market_data_boundary", sa.DateTime(timezone=True), nullable=False),
        sa.Column("knowledge_cutoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("state_id"),
    )
    op.create_index("ux_unified_market_states_hash", "unified_market_states", ["state_hash"], unique=True)
    op.create_index("ix_unified_market_states_series_boundary", "unified_market_states", ["instrument", "market_data_boundary"])
    op.create_index(op.f("ix_unified_market_states_instrument"), "unified_market_states", ["instrument"])
    op.create_index(op.f("ix_unified_market_states_trigger_timeframe"), "unified_market_states", ["trigger_timeframe"])
    op.create_index(op.f("ix_unified_market_states_market_data_boundary"), "unified_market_states", ["market_data_boundary"])
    op.create_index(op.f("ix_unified_market_states_knowledge_cutoff"), "unified_market_states", ["knowledge_cutoff"])
    op.create_index(op.f("ix_unified_market_states_status"), "unified_market_states", ["status"])
    op.create_index(op.f("ix_unified_market_states_created_at"), "unified_market_states", ["created_at"])

    op.create_table(
        "unified_market_state_timeframes",
        sa.Column("state_id", sa.UUID(), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("frame_id", sa.UUID(), nullable=False),
        sa.Column("source_candle_close_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expected_candle_close_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stale", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["frame_id"], ["market_evidence_frames.frame_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["state_id"], ["unified_market_states.state_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("state_id", "timeframe"),
    )
    op.create_index("ix_unified_market_state_timeframes_frame", "unified_market_state_timeframes", ["frame_id"])
    op.create_index(op.f("ix_unified_market_state_timeframes_source_candle_close_at"), "unified_market_state_timeframes", ["source_candle_close_at"])

    op.create_table(
        "evidence_items",
        sa.Column("evidence_id", sa.UUID(), nullable=False),
        sa.Column("source_frame_id", sa.UUID(), nullable=False),
        sa.Column("source_engine", sa.String(length=64), nullable=False),
        sa.Column("source_timeframe", sa.String(length=16), nullable=False),
        sa.Column("availability", sa.String(length=32), nullable=False),
        sa.Column("source_candle_close_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["source_frame_id"], ["market_evidence_frames.frame_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("evidence_id"),
    )
    op.create_index("ix_evidence_items_engine_timeframe", "evidence_items", ["source_engine", "source_timeframe"])
    op.create_index("ix_evidence_items_availability_time", "evidence_items", ["availability", "available_at"])
    op.create_index(op.f("ix_evidence_items_source_frame_id"), "evidence_items", ["source_frame_id"])
    op.create_index(op.f("ix_evidence_items_source_engine"), "evidence_items", ["source_engine"])
    op.create_index(op.f("ix_evidence_items_source_timeframe"), "evidence_items", ["source_timeframe"])
    op.create_index(op.f("ix_evidence_items_availability"), "evidence_items", ["availability"])
    op.create_index(op.f("ix_evidence_items_source_candle_close_at"), "evidence_items", ["source_candle_close_at"])
    op.create_index(op.f("ix_evidence_items_available_at"), "evidence_items", ["available_at"])

    op.create_table(
        "unified_market_state_evidence_links",
        sa.Column("state_id", sa.UUID(), nullable=False),
        sa.Column("evidence_id", sa.UUID(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["evidence_id"], ["evidence_items.evidence_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["state_id"], ["unified_market_states.state_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("state_id", "evidence_id"),
    )
    op.create_index("ux_unified_market_state_evidence_ordinal", "unified_market_state_evidence_links", ["state_id", "ordinal"], unique=True)
    op.create_index("ix_unified_market_state_evidence_item", "unified_market_state_evidence_links", ["evidence_id"])


def downgrade() -> None:
    op.drop_table("unified_market_state_evidence_links")
    op.drop_table("evidence_items")
    op.drop_table("unified_market_state_timeframes")
    op.drop_table("unified_market_states")
    op.drop_table("market_evidence_frames")
