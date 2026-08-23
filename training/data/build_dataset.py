from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTEXT_BARS = 96
HORIZON_BARS = 6
BAR_SECONDS = 300

FEATURE_COLUMNS = (
    "log_return_1",
    "log_return_3",
    "log_hl_range",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "rolling_vol_12",
    "rolling_vol_48",
    "ema_gap_12",
    "ema_gap_48",
    "hour_sin",
    "hour_cos",
)


def _timestamps(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        clean = pd.to_numeric(values, errors="coerce")
        median = clean.dropna().abs().median()

        if median > 1e14:
            unit = "us"
        elif median > 1e11:
            unit = "ms"
        else:
            unit = "s"

        return pd.to_datetime(clean, unit=unit, utc=True, errors="coerce")

    return pd.to_datetime(values, utc=True, errors="coerce")


def load_market_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)

    frame.columns = [str(column).strip().lower() for column in frame.columns]

    timestamp_column = next(
        (
            item
            for item in ("timestamp", "datetime", "time", "open_time")
            if item in frame.columns
        ),
        None,
    )

    if timestamp_column is None:
        raise ValueError(
            "input requires timestamp, datetime, time, or open_time column"
        )

    required = {"open", "high", "low", "close"}
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"missing OHLC columns: {sorted(missing)}")

    frame["timestamp"] = _timestamps(frame[timestamp_column])

    for column in ("open", "high", "low", "close", "volume"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    columns = ["timestamp", "open", "high", "low", "close"]

    if "volume" in frame.columns:
        columns.append("volume")

    frame = (
        frame[columns]
        .dropna(subset=["timestamp", "open", "high", "low", "close"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    positive = (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid_high = frame["high"] >= frame[["open", "close"]].max(axis=1)
    valid_low = frame["low"] <= frame[["open", "close"]].min(axis=1)

    frame = frame[positive & valid_high & valid_low].reset_index(drop=True)

    if len(frame) < CONTEXT_BARS + HORIZON_BARS:
        raise ValueError("not enough valid market data")

    return frame


def ensure_m5(frame: pd.DataFrame) -> pd.DataFrame:
    deltas = frame["timestamp"].diff().dropna().dt.total_seconds()

    if deltas.empty:
        raise ValueError("cannot infer input timeframe")

    median_delta = float(deltas.median())

    if 240 <= median_delta <= 360:
        return frame.reset_index(drop=True)

    if median_delta > 360:
        raise ValueError(
            f"input timeframe is too coarse for M5: median delta={median_delta}"
        )

    indexed = frame.set_index("timestamp")

    aggregation: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }

    if "volume" in indexed.columns:
        aggregation["volume"] = "sum"

    result = indexed.resample("5min", label="left", closed="left").agg(aggregation)
    result = result.dropna(subset=["open", "high", "low", "close"]).reset_index()

    return result


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    open_ = result["open"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    close = result["close"].astype(float)

    log_close = np.log(close)
    return_1 = log_close.diff()

    result["log_return_1"] = return_1
    result["log_return_3"] = log_close.diff(3)
    result["log_hl_range"] = np.log(high / low)

    result["body_pct"] = (close - open_) / open_

    candle_top = pd.concat([open_, close], axis=1).max(axis=1)
    candle_bottom = pd.concat([open_, close], axis=1).min(axis=1)

    result["upper_wick_pct"] = (high - candle_top) / open_
    result["lower_wick_pct"] = (candle_bottom - low) / open_

    result["rolling_vol_12"] = return_1.rolling(12).std()
    result["rolling_vol_48"] = return_1.rolling(48).std()

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_48 = close.ewm(span=48, adjust=False).mean()

    result["ema_gap_12"] = close / ema_12 - 1.0
    result["ema_gap_48"] = close / ema_48 - 1.0

    hour = (
        result["timestamp"].dt.hour
        + result["timestamp"].dt.minute / 60.0
    )

    angle = 2.0 * np.pi * hour / 24.0

    result["hour_sin"] = np.sin(angle)
    result["hour_cos"] = np.cos(angle)

    return result


def parse_boundary(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp


def build_samples(
    bars: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> pd.DataFrame:
    """Vectorized leakage-safe TEN training sample builder."""

    n = len(bars)

    if n < CONTEXT_BARS + HORIZON_BARS:
        raise ValueError("not enough bars to create training samples")

    print("building samples with vectorized engine...")

    timestamp_ns = bars["timestamp"].astype("int64").to_numpy()

    # Candidate cutoff positions.
    cutoffs = np.arange(
        CONTEXT_BARS - 1,
        n - HORIZON_BARS,
        dtype=np.int64,
    )

    context_starts = cutoffs - CONTEXT_BARS + 1
    target_ends = cutoffs + HORIZON_BARS

    # ------------------------------------------------------------
    # Continuity validation
    # ------------------------------------------------------------

    expected_delta_ns = BAR_SECONDS * 1_000_000_000

    bad_edges = (
        np.diff(timestamp_ns) != expected_delta_ns
    ).astype(np.int64)

    edge_prefix = np.concatenate(
        ([0], np.cumsum(bad_edges, dtype=np.int64))
    )

    # Edges from context_start -> target_end must all be exactly M5.
    bad_edge_count = (
        edge_prefix[target_ends]
        - edge_prefix[context_starts]
    )

    contiguous = bad_edge_count == 0

    # ------------------------------------------------------------
    # Feature validity
    # ------------------------------------------------------------

    features = bars.loc[
        :,
        list(FEATURE_COLUMNS),
    ].to_numpy(dtype=np.float32)

    bad_feature_row = (
        ~np.isfinite(features).all(axis=1)
    ).astype(np.int64)

    feature_prefix = np.concatenate(
        ([0], np.cumsum(bad_feature_row, dtype=np.int64))
    )

    context_bad_features = (
        feature_prefix[cutoffs + 1]
        - feature_prefix[context_starts]
    )

    valid_features = context_bad_features == 0

    # ------------------------------------------------------------
    # Temporal split — no boundary-crossing samples
    # ------------------------------------------------------------

    train_end_ns = int(train_end.value)
    validation_end_ns = int(validation_end.value)

    cutoff_ns = timestamp_ns[cutoffs]
    target_end_ns = timestamp_ns[target_ends]

    train_mask = target_end_ns <= train_end_ns

    validation_mask = (
        (cutoff_ns > train_end_ns)
        & (target_end_ns <= validation_end_ns)
    )

    test_mask = cutoff_ns > validation_end_ns

    valid_split = train_mask | validation_mask | test_mask

    valid = contiguous & valid_features & valid_split

    cutoffs = cutoffs[valid]
    context_starts = context_starts[valid]
    target_ends = target_ends[valid]

    cutoff_ns = timestamp_ns[cutoffs]
    target_end_ns = timestamp_ns[target_ends]

    train_mask = target_end_ns <= train_end_ns

    validation_mask = (
        (cutoff_ns > train_end_ns)
        & (target_end_ns <= validation_end_ns)
    )

    split = np.full(len(cutoffs), "test", dtype=object)
    split[validation_mask] = "validation"
    split[train_mask] = "train"

    print("valid continuous samples:", f"{len(cutoffs):,}")

    # ------------------------------------------------------------
    # Future geometry
    # ------------------------------------------------------------

    offsets = np.arange(
        1,
        HORIZON_BARS + 1,
        dtype=np.int64,
    )

    future_indices = cutoffs[:, None] + offsets[None, :]

    close = bars["close"].to_numpy(dtype=np.float64)
    high = bars["high"].to_numpy(dtype=np.float64)
    low = bars["low"].to_numpy(dtype=np.float64)

    reference_price = close[cutoffs]

    future_close = close[future_indices]
    future_high = high[future_indices]
    future_low = low[future_indices]

    future_close_log_returns = np.log(
        future_close / reference_price[:, None]
    ).astype(np.float32)

    future_high_log_return = np.log(
        future_high.max(axis=1) / reference_price
    ).astype(np.float32)

    future_low_log_return = np.log(
        future_low.min(axis=1) / reference_price
    ).astype(np.float32)

    future_close_log_return = future_close_log_returns[:, -1]

    time_to_high_bars = (
        np.argmax(future_high, axis=1) + 1
    ).astype(np.int16)

    time_to_low_bars = (
        np.argmin(future_low, axis=1) + 1
    ).astype(np.int16)

    # Step-by-step future returns:
    # reference -> t+5 -> ... -> t+30
    path_prices = np.concatenate(
        (
            reference_price[:, None],
            future_close,
        ),
        axis=1,
    )

    future_step_returns = np.diff(
        np.log(path_prices),
        axis=1,
    )

    future_realized_volatility = np.std(
        future_step_returns,
        axis=1,
        ddof=0,
    ).astype(np.float32)

    print("future labels calculated")

    samples = pd.DataFrame(
        {
            "split": split,
            "cutoff_index": cutoffs,
            "context_start_index": context_starts,
            "target_end_index": target_ends,
            "cutoff_time": pd.to_datetime(
                cutoff_ns,
                utc=True,
            ),
            "target_end_time": pd.to_datetime(
                target_end_ns,
                utc=True,
            ),
            "reference_price": reference_price.astype(
                np.float32
            ),
            "future_close_log_returns": (
                future_close_log_returns.tolist()
            ),
            "future_high_log_return": future_high_log_return,
            "future_low_log_return": future_low_log_return,
            "future_close_log_return": future_close_log_return,
            "time_to_high_bars": time_to_high_bars,
            "time_to_low_bars": time_to_low_bars,
            "future_realized_volatility": (
                future_realized_volatility
            ),
        }
    )

    return samples

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--validation-end", required=True)

    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_market_data(args.input)
    frame = ensure_m5(frame)
    frame = add_features(frame)

    train_end = parse_boundary(args.train_end)
    validation_end = parse_boundary(args.validation_end)

    samples = build_samples(
        frame,
        train_end=train_end,
        validation_end=validation_end,
    )

    if samples.empty:
        raise RuntimeError("dataset builder produced zero valid samples")

    bars_path = output_dir / "bars.parquet"
    samples_path = output_dir / "samples.parquet"
    metadata_path = output_dir / "metadata.json"

    frame.to_parquet(bars_path, index=False)
    samples.to_parquet(samples_path, index=False)

    split_counts = {
        key: int(value)
        for key, value in samples["split"].value_counts().items()
    }

    metadata = {
        "schema_version": "ten-world-model-dataset-v1",
        "bar_seconds": BAR_SECONDS,
        "context_bars": CONTEXT_BARS,
        "context_minutes": CONTEXT_BARS * 5,
        "horizon_bars": HORIZON_BARS,
        "horizon_minutes": HORIZON_BARS * 5,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "bar_count": len(frame),
        "sample_count": len(samples),
        "split_counts": split_counts,
        "train_end": train_end.isoformat(),
        "validation_end": validation_end.isoformat(),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))
    print()
    print("bars:", bars_path)
    print("samples:", samples_path)
    print("metadata:", metadata_path)


if __name__ == "__main__":
    main()
