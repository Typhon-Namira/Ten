"""Prevent repository conflict targets from drifting from authoritative schema boundaries."""

import ast
from pathlib import Path

import pytest
from sqlalchemy import Index, UniqueConstraint

from backend.app.storage.models import Base
from backend.app.storage.models import (
    AIMarketForecastRecord,
    EconomicCalendarRevisionRecord,
    GuardrailEvaluationRecord,
    InstitutionalFlowSnapshotRecord,
    LiquiditySnapshotRecord,
    ManagedSignalRecord,
    MarketRegimeSnapshotRecord,
    QuantFeatureVectorRecord,
    QuantForecastResultRecord,
    ReplayOutputRecord,
    SMCAnalysisSnapshotRecord,
    VolumeProfileSnapshotRecord,
)

ROOT = Path(__file__).parents[2]
SNAPSHOT_CONTRACTS = (
    (
        ROOT / "backend/app/engines/smc_engine/repository.py",
        SMCAnalysisSnapshotRecord,
        ("symbol", "timeframe", "analysis_timestamp", "configuration_version", "processing_mode"),
    ),
    (
        ROOT / "backend/app/engines/liquidity_engine/repository.py",
        LiquiditySnapshotRecord,
        ("symbol", "timeframe", "analysis_timestamp", "configuration_version", "processing_mode"),
    ),
    (
        ROOT / "backend/app/engines/volume_profile_engine/repository.py",
        VolumeProfileSnapshotRecord,
        ("symbol", "timeframe", "analysis_timestamp", "configuration_version", "processing_mode"),
    ),
    (
        ROOT / "backend/app/engines/institutional_flow_engine/repository.py",
        InstitutionalFlowSnapshotRecord,
        ("symbol", "timeframe", "analysis_timestamp", "configuration_version", "processing_mode"),
    ),
    (
        ROOT / "backend/app/engines/market_regime_engine/repository.py",
        MarketRegimeSnapshotRecord,
        ("symbol", "timeframe", "analysis_timestamp", "configuration_version"),
    ),
    (
        ROOT / "backend/app/engines/economic_calendar_engine/repository.py",
        EconomicCalendarRevisionRecord,
        ("event_id", "revision_number"),
    ),
    (
        ROOT / "backend/app/engines/replay_engine/repository.py",
        ReplayOutputRecord,
        ("replay_id", "fingerprint"),
    ),
    (
        ROOT / "backend/app/quant_forecasting/repository.py",
        QuantFeatureVectorRecord,
        ("market_state_id",),
    ),
    (
        ROOT / "backend/app/quant_forecasting/repository.py",
        QuantForecastResultRecord,
        ("request_id",),
    ),
    (
        ROOT / "backend/app/ai_reasoning/repository.py",
        AIMarketForecastRecord,
        ("request_id",),
    ),
    (
        ROOT / "backend/app/ai_reasoning/repository.py",
        ManagedSignalRecord,
        ("structural_opportunity_key",),
    ),
    (
        ROOT / "backend/app/final_decision/repository.py",
        GuardrailEvaluationRecord,
        ("final_action_id", "gate_id"),
    ),
)


def _insert_model(call: ast.Call) -> str | None:
    node: ast.AST = call.func.value  # type: ignore[union-attr]
    while isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "insert" and node.args:
            argument = node.args[0]
            return argument.id if isinstance(argument, ast.Name) else None
        if isinstance(node.func, ast.Attribute):
            node = node.func.value
        else:
            break
    return None


def _conflict_target_entries(path: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    targets: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"on_conflict_do_nothing", "on_conflict_do_update"}
        ):
            continue
        model = _insert_model(node)
        keyword = next((item for item in node.keywords if item.arg == "index_elements"), None)
        if model is None or keyword is None or not isinstance(keyword.value, (ast.List, ast.Tuple)):
            continue
        columns = tuple(
            item.value if isinstance(item, ast.Constant) else item.attr
            for item in keyword.value.elts
            if isinstance(item, (ast.Constant, ast.Attribute))
        )
        targets.append((model, columns))
    return tuple(targets)


def _conflict_targets(path: Path) -> dict[str, tuple[str, ...]]:
    return dict(_conflict_target_entries(path))


@pytest.mark.parametrize(("repository_path", "record_type", "expected"), SNAPSHOT_CONTRACTS)
def test_snapshot_conflict_target_matches_model_and_migration(
    repository_path: Path,
    record_type: type[object],
    expected: tuple[str, ...],
) -> None:
    unique_indexes = {
        tuple(column.name for column in index.columns)
        for index in record_type.__table__.indexes  # type: ignore[attr-defined]
        if isinstance(index, Index) and index.unique
    }
    assert expected in unique_indexes
    assert _conflict_targets(repository_path)[record_type.__name__] == expected

    migration = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "migrations/versions").glob("*.py"))
    )
    constraint_name = next(
        index.name
        for index in record_type.__table__.indexes  # type: ignore[attr-defined]
        if isinstance(index, Index) and index.unique and tuple(column.name for column in index.columns) == expected
    )
    assert constraint_name is not None
    if constraint_name.startswith("ux_"):
        assert constraint_name in migration
    else:
        assert record_type.__table__.name in migration  # type: ignore[attr-defined]
        assert all(column in migration for column in expected)
        assert "unique=True" in migration


def test_every_static_repository_conflict_target_is_a_real_schema_boundary() -> None:
    model_types = {mapper.class_.__name__: mapper.class_ for mapper in Base.registry.mappers}
    repository_paths = tuple((ROOT / "backend").rglob("*repository.py"))
    for path in repository_paths:
        for model_name, target in _conflict_target_entries(path):
            record_type = model_types.get(model_name)
            if record_type is None:
                continue
            table = record_type.__table__
            valid_boundaries = {tuple(column.name for column in table.primary_key.columns)}
            valid_boundaries.update(
                tuple(column.name for column in item.columns)
                for item in (*table.indexes, *table.constraints)
                if (isinstance(item, Index) and item.unique) or isinstance(item, UniqueConstraint)
            )
            assert target in valid_boundaries, f"{path.relative_to(ROOT)}: {model_name} targets {target}, schema has {sorted(valid_boundaries)}"


def test_primary_key_only_conflict_target_has_no_competing_unique_boundary() -> None:
    model_types = {mapper.class_.__name__: mapper.class_ for mapper in Base.registry.mappers}
    for path in (ROOT / "backend").rglob("*repository.py"):
        for model_name, target in _conflict_target_entries(path):
            record_type = model_types.get(model_name)
            if record_type is None:
                continue
            table = record_type.__table__
            primary_key = tuple(column.name for column in table.primary_key.columns)
            unique_boundaries = {
                tuple(column.name for column in item.columns)
                for item in (*table.indexes, *table.constraints)
                if ((isinstance(item, Index) and item.unique) or isinstance(item, UniqueConstraint))
                and tuple(column.name for column in item.columns) != primary_key
            }
            if target == primary_key:
                assert not unique_boundaries, (
                    f"{path.relative_to(ROOT)}: {model_name} handles only primary key {primary_key} "
                    f"but competing unique boundaries exist: {sorted(unique_boundaries)}"
                )
