import asyncio
from uuid import uuid4

from backend.app.features import FeatureRecord, InMemoryFeatureStore


def test_feature_store_returns_namespaced_versioned_snapshot() -> None:
    correlation_id = uuid4()
    store = InMemoryFeatureStore()
    asyncio.run(store.write(FeatureRecord(correlation_id=correlation_id, namespace="smc", engine_name="smc", engine_version="1.0.0", compatibility_version="1.0", values={"bias": "bullish"})))
    snapshot = asyncio.run(store.snapshot(correlation_id))
    assert snapshot.features == {"smc": {"bias": "bullish"}}
    assert snapshot.engine_versions == {"smc": "1.0.0"}
