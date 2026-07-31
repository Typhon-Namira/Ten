from asyncio import run
from io import StringIO
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from backend.app.core.database import Base, SCHEMA_HEAD_REVISION, prepare_database_schema
import backend.app.storage.models  # noqa: F401 -- registers the production metadata


ROOT = Path(__file__).resolve().parents[2]


def alembic_config(*, output_buffer: StringIO | None = None) -> Config:
    config = Config(str(ROOT / "alembic.ini"), output_buffer=output_buffer)
    config.set_main_option("script_location", str(ROOT / "migrations"))
    return config


def test_runtime_schema_revision_matches_the_actual_alembic_head() -> None:
    script = ScriptDirectory.from_config(alembic_config())

    assert script.get_current_head() == SCHEMA_HEAD_REVISION


def test_multi_timeframe_signal_migration_is_strict_idempotent_and_reversible() -> None:
    source = (
        ROOT
        / "migrations/versions/20260729_0014_multi_timeframe_signal_synthesis.py"
    ).read_text(encoding="utf-8")

    assert 'revision = "20260729_0014"' in source
    assert 'down_revision = "20260728_0013"' in source
    assert '"multi_timeframe_signal_sets"' in source
    assert '"timeframe_analytical_signals"' in source
    assert "ux_multi_timeframe_signal_sets_market_state_id" in source
    assert "ux_multi_timeframe_signal_sets_analysis_id" in source
    assert "ux_timeframe_analytical_signal_scope" in source
    assert "combined_direction IN ('BUY','SELL')" in source
    assert "analytical_direction IN ('BUY','SELL')" in source
    assert "execution_status IN ('READY','BLOCKED')" in source
    assert 'ondelete="CASCADE"' in source
    assert 'ondelete="RESTRICT"' in source
    assert 'op.drop_table("timeframe_analytical_signals")' in source
    assert 'op.drop_table("multi_timeframe_signal_sets")' in source


def test_scenario_forecasting_migration_is_traceable_and_reversible() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations/versions/20260730_0016_scenario_forecasting.py"
    )
    source = path.read_text(encoding="utf-8")

    for table in (
        "forward_market_scenarios",
        "combined_forward_scenarios",
        "scenario_outcomes",
    ):
        assert f'"{table}"' in source
        assert f'op.drop_table("{table}")' in source
    assert "ux_forward_market_scenario_boundary" in source
    assert "market_cutoff_time" in source
    assert "ondelete=\"RESTRICT\"" in source
    assert "ondelete=\"CASCADE\"" in source


def test_primary_scenario_simulation_migration_is_normalized_and_reversible() -> None:
    source = (
        ROOT
        / "migrations/versions/20260730_0017_primary_scenario_simulation.py"
    ).read_text(encoding="utf-8")
    tables = (
        "market_simulation_cycles",
        "candidate_market_scenarios",
        "scenario_path_stages",
        "scenario_score_components",
        "primary_scenario_selections",
        "primary_scenario_geometries",
        "scenario_lifecycle_transitions",
        "scenario_calibration_metrics",
        "candidate_scenario_outcomes",
    )
    for table in tables:
        assert f'"{table}"' in source
        assert f'op.drop_table("{table}")' in source
    assert 'down_revision = "20260730_0016"' in source
    assert "ux_market_simulation_boundary" in source
    assert "ux_candidate_market_scenario_diversity" in source
    assert 'ondelete="CASCADE"' in source
    assert 'ondelete="RESTRICT"' in source


def test_authoritative_simulation_attempt_migration_is_idempotent_and_reversible() -> None:
    source = (
        ROOT
        / "migrations/versions/20260731_0018_authoritative_simulation_attempts.py"
    ).read_text(encoding="utf-8")

    assert 'revision = "20260731_0018"' in source
    assert 'down_revision = "20260730_0017"' in source
    assert '"authoritative_simulation_attempts"' in source
    assert "ux_authoritative_simulation_attempt_boundary" in source
    for status in (
        "SCHEDULED",
        "RUNNING",
        "SUCCESS",
        "NO_SIGNAL",
        "ANALYTICAL_ONLY",
        "BLOCKED",
        "FAILED",
        "SKIPPED",
    ):
        assert status in source
    assert 'op.drop_table("authoritative_simulation_attempts")' in source


def test_ai_reasoning_gate_decision_migration_is_traceable_and_reversible() -> None:
    source = (
        ROOT
        / "migrations/versions/20260731_0019_ai_reasoning_gate_decisions.py"
    ).read_text(encoding="utf-8")

    assert 'revision = "20260731_0019"' in source
    assert 'down_revision = "20260731_0018"' in source
    assert '"ai_reasoning_gate_decisions"' in source
    assert "ix_ai_reasoning_gate_decision_boundary" in source
    assert "ix_ai_reasoning_gate_decisions_market_state_id" in source
    assert 'ondelete="SET NULL"' in source
    assert 'op.drop_table("ai_reasoning_gate_decisions")' in source


def test_scenario_waiting_and_email_delivery_migration_is_reversible() -> None:
    source = (
        ROOT
        / "migrations/versions/20260731_0020_scenario_waiting_and_email_delivery.py"
    ).read_text(encoding="utf-8")

    assert 'revision = "20260731_0020"' in source
    assert 'down_revision = "20260731_0019"' in source
    assert "WAITING_FOR_AI_ANALYSIS" in source
    assert "PERMANENTLY_FAILED" in source
    assert "COMMITTED" in source
    assert "ai_analysis_id" in source
    assert "quantitative_forecast_id" in source
    assert "op.drop_column" in source


def test_ai_reasoning_claim_lease_migration_is_reversible() -> None:
    source = (
        ROOT / "migrations/versions/20260731_0021_ai_reasoning_claim_leases.py"
    ).read_text(encoding="utf-8")

    assert 'revision = "20260731_0021"' in source
    assert 'down_revision = "20260731_0020"' in source
    for field in (
        "claim_id",
        "market_state_id",
        "snapshot_id",
        "claimed_by",
        "heartbeat_at",
        "lease_expires_at",
        "failure_reason",
        "released_at",
    ):
        assert field in source
    assert "ACTIVE_CLAIM" in source
    assert "RECOVERED" in source
    assert "op.drop_column" in source


def test_signal_email_outbox_migration_is_idempotent_and_reversible() -> None:
    source = (
        ROOT / "migrations/versions/20260730_0015_signal_email_outbox.py"
    ).read_text(encoding="utf-8")

    assert 'revision = "20260730_0015"' in source
    assert 'down_revision = "20260729_0014"' in source
    assert '"signal_email_outbox"' in source
    assert '"ux_signal_email_outbox_signal_id"' in source
    assert 'ForeignKey("signal_decisions.id", ondelete="CASCADE")' in source
    assert "PENDING" in source
    assert "PROCESSING" in source
    assert "SENT" in source
    assert "FAILED" in source
    assert 'op.drop_table("signal_email_outbox")' in source


def test_initial_migration_renders_every_model_as_postgresql_ddl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEN_DATABASE_URL", "postgresql+asyncpg://ten:ten@localhost:5432/ten")
    output = StringIO()

    command.upgrade(alembic_config(output_buffer=output), "head", sql=True)

    ddl = output.getvalue()
    assert SCHEMA_HEAD_REVISION in ddl
    for table_name in Base.metadata.tables:
        assert (
            f"CREATE TABLE {table_name}" in ddl
            or (
                table_name == "ai_reasoning_cycle_locks"
                and (
                    "ALTER TABLE ai_reasoning_"
                    "window_locks RENAME TO ai_reasoning_cycle_locks"
                )
                in ddl
            )
        )
    assert "FOREIGN KEY" in ddl
    assert "CREATE UNIQUE INDEX" in ddl
    assert "JSONB" in ddl


def test_alembic_accepts_railways_native_postgresql_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEN_DATABASE_URL", "postgresql://postgres:secret@postgres.railway.internal:5432/railway")
    output = StringIO()

    command.upgrade(alembic_config(output_buffer=output), "head", sql=True)

    assert SCHEMA_HEAD_REVISION in output.getvalue()


@pytest.mark.asyncio
async def test_managed_runtime_requires_the_alembic_head() -> None:
    connection = AsyncMock()
    result = Mock()
    result.scalar_one_or_none.return_value = "outdated"
    connection.run_sync.return_value = True
    connection.execute.return_value = result

    with pytest.raises(RuntimeError, match="alembic upgrade head"):
        await prepare_database_schema(connection, managed_runtime=True)

    connection.run_sync.assert_awaited_once()


@pytest.mark.asyncio
async def test_managed_runtime_rejects_an_unversioned_schema() -> None:
    connection = AsyncMock()
    connection.run_sync.return_value = False

    with pytest.raises(RuntimeError, match="alembic upgrade head"):
        await prepare_database_schema(connection, managed_runtime=True)

    connection.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_managed_runtime_accepts_the_alembic_head() -> None:
    connection = AsyncMock()
    result = Mock()
    result.scalar_one_or_none.return_value = SCHEMA_HEAD_REVISION
    connection.run_sync.return_value = True
    connection.execute.return_value = result

    await prepare_database_schema(connection, managed_runtime=True)

    connection.run_sync.assert_awaited_once()
    connection.execute.assert_awaited_once()


def test_clean_postgresql_database_can_migrate_to_head() -> None:
    database_url = os.getenv("TEN_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEN_TEST_DATABASE_URL is required for the disposable PostgreSQL migration test")
    if not database_url.startswith("postgresql+asyncpg://"):
        pytest.fail("TEN_TEST_DATABASE_URL must use postgresql+asyncpg://")

    async def table_names() -> set[str]:
        engine = create_async_engine(database_url)
        try:
            async with engine.connect() as connection:
                return set(await connection.run_sync(lambda sync_connection: inspect(sync_connection).get_table_names()))
        finally:
            await engine.dispose()

    assert run(table_names()) == set(), "TEN_TEST_DATABASE_URL must reference a clean disposable database"
    config = alembic_config()
    previous_url = os.environ.get("TEN_DATABASE_URL")
    os.environ["TEN_DATABASE_URL"] = database_url
    try:
        command.upgrade(config, "head")
        migrated_tables = run(table_names())
        assert set(Base.metadata.tables) <= migrated_tables
        assert "alembic_version" in migrated_tables
    finally:
        command.downgrade(config, "base")

        async def remove_version_table() -> None:
            engine = create_async_engine(database_url)
            try:
                async with engine.begin() as connection:
                    await connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
            finally:
                await engine.dispose()

        run(remove_version_table())
        if previous_url is None:
            os.environ.pop("TEN_DATABASE_URL", None)
        else:
            os.environ["TEN_DATABASE_URL"] = previous_url
