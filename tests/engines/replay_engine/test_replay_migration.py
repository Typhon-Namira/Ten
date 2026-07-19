from pathlib import Path

from backend.app.core.database.base import Base


def test_replay_migration_and_sqlalchemy_metadata() -> None:
    root = Path(__file__).resolve().parents[3]
    migration = (root / "migrations" / "20260719_replay_engine_v1.sql").read_text(encoding="utf-8")
    tables = {"replay_sessions", "replay_checkpoints", "replay_transitions", "replay_event_trace", "replay_outputs"}
    assert all(f"CREATE TABLE IF NOT EXISTS {table}" in migration for table in tables)
    assert tables.issubset(Base.metadata.tables)
    assert "row_version" in migration
    assert "lease_expires_at" in migration
    assert "UNIQUE (replay_id, sequence)" in migration
    assert "ON DELETE CASCADE" in migration
    assert "CREATE INDEX IF NOT EXISTS" in migration
    assert "DROP TABLE" not in migration.upper()
