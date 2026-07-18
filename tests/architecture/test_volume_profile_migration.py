from pathlib import Path

from backend.app.core.database.base import Base
from backend.app.storage import models as _models


def test_volume_profile_migration_is_idempotent_nondestructive_and_registered() -> None:
    assert _models.VolumeProfileSnapshotRecord.__tablename__ == "volume_profile_snapshots"
    sql = Path("migrations/20260718_volume_profile_v1.sql").read_text()
    assert sql.count("CREATE TABLE IF NOT EXISTS") == 3
    assert "DROP " not in sql.upper() and "TRUNCATE " not in sql.upper() and "ALTER " not in sql.upper()
    assert {"volume_profile_snapshots", "volume_profile_objects", "volume_profile_checkpoints"} <= set(Base.metadata.tables)
