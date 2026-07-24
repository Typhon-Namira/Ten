"""Production-like schema and ownership checks for the additive storage migration."""

from importlib import import_module

from backend.app.storage.maintenance import _CLEANABLE_RELATIONS
from backend.app.storage.models import (
    EvidenceItemRecord,
    MarketEvidenceFrameRecord,
    UnifiedMarketStateCurrentRecord,
    UnifiedMarketStateEvidenceLinkRecord,
    UnifiedMarketStateRecord,
    UnifiedMarketStateTimeframeRecord,
)


def test_migration_is_additive_and_leaves_legacy_payload_rows_untouched(monkeypatch) -> None:
    migration = import_module(
        "migrations.versions.20260724_0007_dashboard_storage_control"
    )
    operations: list[tuple[str, str]] = []

    class FakeOp:
        def create_table(self, name: str, *_: object, **__: object) -> None:
            operations.append(("create_table", name))

        def create_index(
            self, name: str, table: str, *_: object, **__: object
        ) -> None:
            operations.append(("create_index", f"{table}.{name}"))

        def bulk_insert(self, table: object, rows: list[dict[str, object]]) -> None:
            operations.append(("bulk_insert", f"{getattr(table, 'name', '')}:{len(rows)}"))

    monkeypatch.setattr(migration, "op", FakeOp())
    migration.upgrade()

    assert {value for kind, value in operations if kind == "create_table"} == {
        "unified_market_state_current",
        "pipeline_stage_current",
        "pipeline_stage_history",
        "storage_retention_policies",
    }
    touched = " ".join(value for _, value in operations)
    assert "unified_market_states." not in touched
    assert "evidence_items." not in touched
    assert "market_evidence_frames." not in touched


def test_single_full_payload_owner_and_reference_foreign_keys_are_explicit() -> None:
    assert MarketEvidenceFrameRecord.__table__.c.payload.nullable is False
    # These JSONB columns remain for rolling compatibility, but new repository writes compact
    # metadata only; the semantic equivalence test proves reads rehydrate from the frame.
    assert UnifiedMarketStateRecord.__table__.c.payload.nullable is False
    assert EvidenceItemRecord.__table__.c.payload.nullable is False

    current_fk = next(iter(UnifiedMarketStateCurrentRecord.__table__.c.state_id.foreign_keys))
    timeframe_state_fk = next(
        iter(UnifiedMarketStateTimeframeRecord.__table__.c.state_id.foreign_keys)
    )
    timeframe_frame_fk = next(
        iter(UnifiedMarketStateTimeframeRecord.__table__.c.frame_id.foreign_keys)
    )
    link_state_fk = next(
        iter(UnifiedMarketStateEvidenceLinkRecord.__table__.c.state_id.foreign_keys)
    )
    link_evidence_fk = next(
        iter(UnifiedMarketStateEvidenceLinkRecord.__table__.c.evidence_id.foreign_keys)
    )
    assert current_fk.ondelete == "CASCADE"
    assert timeframe_state_fk.ondelete == "CASCADE"
    assert timeframe_frame_fk.ondelete == "RESTRICT"
    assert link_state_fk.ondelete == "CASCADE"
    assert link_evidence_fk.ondelete == "RESTRICT"


def test_retention_worker_cannot_delete_immutable_analytical_owners() -> None:
    assert _CLEANABLE_RELATIONS == {
        "realtime_candles": "received_at",
        "pipeline_stage_history": "observed_at",
    }
    assert not {
        "market_evidence_frames",
        "evidence_items",
        "unified_market_states",
    } & set(_CLEANABLE_RELATIONS)
