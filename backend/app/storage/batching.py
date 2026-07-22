"""Bounded multi-row insert planning for PostgreSQL/AsyncPG repositories."""

from collections.abc import Mapping, Sequence


DEFAULT_BIND_PARAMETER_BUDGET = 30_000
DEFAULT_OPERATIONAL_ROW_CAP = 1_000


def maximum_rows_per_insert(
    parameters_per_row: int,
    *,
    bind_parameter_budget: int = DEFAULT_BIND_PARAMETER_BUDGET,
    operational_row_cap: int = DEFAULT_OPERATIONAL_ROW_CAP,
) -> int:
    """Derive a conservative row limit without ever exceeding the bind budget."""
    if parameters_per_row < 1:
        raise ValueError("parameters_per_row must be positive")
    if bind_parameter_budget < parameters_per_row:
        raise ValueError("bind_parameter_budget must fit at least one row")
    if operational_row_cap < 1:
        raise ValueError("operational_row_cap must be positive")
    return min(bind_parameter_budget // parameters_per_row, operational_row_cap)


def bounded_insert_chunks[InsertRow: Mapping[str, object]](
    rows: Sequence[InsertRow],
    *,
    bind_parameter_budget: int = DEFAULT_BIND_PARAMETER_BUDGET,
    operational_row_cap: int = DEFAULT_OPERATIONAL_ROW_CAP,
) -> tuple[tuple[InsertRow, ...], ...]:
    """Split homogeneous insert rows into deterministic, parameter-safe chunks."""
    if not rows:
        return ()
    columns = tuple(rows[0])
    if not columns:
        raise ValueError("insert rows must contain at least one bound column")
    expected = frozenset(columns)
    if any(frozenset(row) != expected for row in rows):
        raise ValueError("all insert rows must bind the same columns")
    chunk_size = maximum_rows_per_insert(
        len(columns),
        bind_parameter_budget=bind_parameter_budget,
        operational_row_cap=operational_row_cap,
    )
    chunks = tuple(tuple(rows[index : index + chunk_size]) for index in range(0, len(rows), chunk_size))
    if any(len(chunk) * len(columns) > bind_parameter_budget for chunk in chunks):
        raise ValueError("generated insert chunk exceeds bind-parameter budget")
    return chunks
