from pathlib import Path

from backend.app.core.database.base import Base
from backend.app.storage import models as storage_models


ROOT = Path(__file__).parents[2]


def test_liquidity_metadata_and_idempotent_nondestructive_migration() -> None:
    assert storage_models.LiquiditySnapshotRecord.__tablename__ == "liquidity_snapshots"
    required = {"liquidity_snapshots", "liquidity_objects", "liquidity_checkpoints"}
    assert required <= set(Base.metadata.tables)
    sql = (ROOT / "migrations" / "20260717_liquidity_v1.sql").read_text()
    assert all(f"CREATE TABLE IF NOT EXISTS {name}" in sql for name in required)
    assert sql.count("CREATE INDEX IF NOT EXISTS") >= 4
    assert sql.count("CREATE UNIQUE INDEX IF NOT EXISTS") >= 2
    assert not any(statement in sql.upper() for statement in ("DROP TABLE", "TRUNCATE", "ALTER TABLE"))


def test_liquidity_provider_and_trading_boundaries() -> None:
    source = "\n".join(path.read_text() for path in (ROOT / "backend" / "app" / "engines" / "liquidity_engine").glob("*.py"))
    forbidden = (
        "TwelveData",
        "AlphaVantage",
        "FinancialModelingPrep",
        "OandaProvider",
        "import requests",
        "import httpx",
        "place_order",
        "execute_order",
        "stop_loss",
        "take_profit",
    )
    assert not any(item.lower() in source.lower() for item in forbidden)
    assert not any(item in source for item in ("BOSDetected", "CHOCHDetected", "MSSDetected", "ZoneType.BULLISH_FVG", "VolumeProfile"))


def test_liquidity_documentation_is_complete_and_honest() -> None:
    document = (ROOT / "docs" / "LIQUIDITY_ENGINE.md").read_text()
    required = (
        "SMC-confirmed",
        "price-action label",
        "DST",
        "inducement",
        "confluence",
        "target priority",
        "inferred_density",
        "no entry",
        "checkpoints",
        "100",
    )
    assert all(term.lower() in document.lower() for term in required)
    assert "not order-book heatmaps" in document
