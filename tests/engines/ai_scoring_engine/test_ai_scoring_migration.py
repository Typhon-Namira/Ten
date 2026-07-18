from pathlib import Path

from backend.app.core.database.base import Base


def test_ai_scoring_migration_and_metadata_are_registered() -> None:
    root = Path(__file__).resolve().parents[3]
    migration = (root / "migrations" / "20260719_ai_scoring_v1.sql").read_text(encoding="utf-8")
    tables = {"ai_score_snapshots", "ai_score_components", "ai_score_conflicts"}
    assert all(f"CREATE TABLE IF NOT EXISTS {name}" in migration for name in tables)
    assert all(name in Base.metadata.tables for name in tables)
    assert "ON DELETE CASCADE" in migration
    assert "UNIQUE (input_fingerprint, mode)" in migration
    assert "CREATE INDEX IF NOT EXISTS" in migration
    assert "DROP TABLE" not in migration.upper()
