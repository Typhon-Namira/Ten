from pathlib import Path

from backend.app.core.database.base import Base


def test_signal_decision_migration_and_metadata() -> None:
    root = Path(__file__).resolve().parents[3]
    migration = (root / "migrations" / "20260719_signal_decision_v1.sql").read_text(encoding="utf-8")
    tables = {"signal_decisions", "signal_decision_rules", "signal_decision_reasons"}
    assert all(f"CREATE TABLE IF NOT EXISTS {name}" in migration for name in tables)
    assert all(name in Base.metadata.tables for name in tables)
    assert "ON DELETE CASCADE" in migration
    assert "ON DELETE RESTRICT" in migration
    assert "UNIQUE (input_fingerprint, mode)" in migration
    assert "CHECK (valid_until >= valid_from)" in migration
    assert "CREATE INDEX IF NOT EXISTS" in migration
    assert "DROP TABLE" not in migration.upper()
