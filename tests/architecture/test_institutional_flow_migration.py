from pathlib import Path

from backend.app.core.database.base import Base


def test_institutional_flow_migration_is_safe_and_models_registered() -> None:
    sql = Path("migrations/20260718_institutional_flow_v1.sql").read_text()
    lowered = sql.lower()
    assert lowered.count("create table if not exists") == 3
    assert "drop table" not in lowered
    assert "institutional_flow_snapshots" in Base.metadata.tables
    assert "institutional_flow_evidence" in Base.metadata.tables
    assert "institutional_flow_checkpoints" in Base.metadata.tables
