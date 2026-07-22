import pytest

from backend.app.storage.batching import bounded_insert_chunks, maximum_rows_per_insert


def test_parameter_budget_derives_chunk_size_and_boundaries() -> None:
    assert maximum_rows_per_insert(13) == 1_000
    rows = [{"a": index, "b": index} for index in range(2_001)]
    assert [len(chunk) for chunk in bounded_insert_chunks(rows, bind_parameter_budget=2_000, operational_row_cap=750)] == [750, 750, 501]


def test_empty_and_invalid_insert_rows_are_guarded() -> None:
    assert bounded_insert_chunks([]) == ()
    with pytest.raises(ValueError, match="same columns"):
        bounded_insert_chunks([{"a": 1}, {"b": 2}])
    with pytest.raises(ValueError, match="fit at least one row"):
        maximum_rows_per_insert(13, bind_parameter_budget=12)
