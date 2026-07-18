from pathlib import Path

from backend.app.core.database.base import Base


def test_migration_is_idempotent_and_metadata_is_registered() -> None:
    root = Path(__file__).resolve().parents[3]
    migration = (root / "migrations" / "20260718_economic_calendar_v1.sql").read_text(encoding="utf-8")
    tables = {
        "economic_calendar_provider_observations",
        "economic_calendar_events",
        "economic_calendar_event_revisions",
        "economic_calendar_snapshots",
        "economic_calendar_instrument_contexts",
        "economic_calendar_sync_state",
        "economic_calendar_checkpoints",
    }
    assert all(f"CREATE TABLE IF NOT EXISTS {name}" in migration for name in tables)
    assert all(name in Base.metadata.tables for name in tables)
    assert "CREATE INDEX IF NOT EXISTS" in migration
    assert "DROP TABLE" not in migration.upper()
