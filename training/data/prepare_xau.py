from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("training/raw/xau")
OUT_DIR = Path("training/processed/xau")

START_YEAR = 2016
END_YEAR = 2026

# A completely identical OHLC price lasting this long is treated as
# a closed/stale market interval rather than useful market information.
STALE_RUN_MINUTES = 60


def read_year(year: int) -> pd.DataFrame:
    files = sorted(RAW_DIR.glob(f"xauusd_bid_m1_{year}_*.csv"))

    if not files:
        raise RuntimeError(f"no files for year {year}")

    pieces: list[pd.DataFrame] = []

    for path in files:
        frame = pd.read_csv(
            path,
            usecols=["timestamp", "open", "high", "low", "close"],
        )

        pieces.append(frame)

    frame = pd.concat(pieces, ignore_index=True)

    frame["timestamp"] = pd.to_datetime(
        pd.to_numeric(frame["timestamp"], errors="coerce"),
        unit="ms",
        utc=True,
        errors="coerce",
    )

    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(
        subset=["timestamp", "open", "high", "low", "close"]
    )

    frame = (
        frame.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    return frame


def valid_ohlc(frame: pd.DataFrame) -> pd.Series:
    prices = frame[["open", "high", "low", "close"]]

    positive = (prices > 0).all(axis=1)

    high_valid = (
        frame["high"]
        >= frame[["open", "close", "low"]].max(axis=1)
    )

    low_valid = (
        frame["low"]
        <= frame[["open", "close", "high"]].min(axis=1)
    )

    return positive & high_valid & low_valid


def stale_mask(frame: pd.DataFrame) -> tuple[pd.Series, list[int]]:
    ohlc = frame[["open", "high", "low", "close"]]

    same_as_previous = ohlc.eq(ohlc.shift()).all(axis=1)

    # Every price change begins a new run.
    groups = (~same_as_previous).cumsum()

    run_sizes = groups.groupby(groups).transform("size")

    stale = run_sizes >= STALE_RUN_MINUTES

    longest = (
        run_sizes.drop_duplicates()
        .sort_values(ascending=False)
        .head(10)
        .astype(int)
        .tolist()
    )

    return stale, longest


def resample_m5(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index("timestamp")

    m5 = indexed.resample(
        "5min",
        label="left",
        closed="left",
    ).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        source_m1_count=("close", "count"),
    )

    # Training integrity is more important than squeezing every bar
    # from the source. Require all five underlying M1 candles.
    m5 = m5.loc[m5["source_m1_count"] == 5]

    m5 = m5.drop(columns=["source_m1_count"])
    m5 = m5.dropna()

    return m5.reset_index()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    yearly: list[pd.DataFrame] = []

    audit: dict[str, object] = {
        "source": "Dukascopy mirror / dukascopy-node",
        "instrument": "XAUUSD",
        "side": "bid",
        "source_timeframe": "M1",
        "output_timeframe": "M5",
        "stale_run_minutes": STALE_RUN_MINUTES,
        "years": {},
    }

    total_raw = 0
    total_invalid = 0
    total_stale = 0

    for year in range(START_YEAR, END_YEAR + 1):
        print()
        print(f"=== {year} ===")

        frame = read_year(year)

        raw_rows = len(frame)
        total_raw += raw_rows

        validity = valid_ohlc(frame)
        invalid_rows = int((~validity).sum())
        total_invalid += invalid_rows

        frame = frame.loc[validity].reset_index(drop=True)

        stale, longest_runs = stale_mask(frame)

        stale_rows = int(stale.sum())
        total_stale += stale_rows

        active = frame.loc[~stale].reset_index(drop=True)

        m5 = resample_m5(active)

        yearly.append(m5)

        year_path = OUT_DIR / f"xauusd_bid_m5_{year}.parquet"
        m5.to_parquet(year_path, index=False)

        audit["years"][str(year)] = {
            "raw_rows": raw_rows,
            "invalid_rows": invalid_rows,
            "stale_rows_removed": stale_rows,
            "active_m1_rows": len(active),
            "m5_rows": len(m5),
            "longest_identical_runs_minutes": longest_runs,
            "first_timestamp": (
                frame["timestamp"].iloc[0].isoformat()
                if len(frame)
                else None
            ),
            "last_timestamp": (
                frame["timestamp"].iloc[-1].isoformat()
                if len(frame)
                else None
            ),
        }

        print("raw M1       :", f"{raw_rows:,}")
        print("invalid      :", f"{invalid_rows:,}")
        print("stale removed:", f"{stale_rows:,}")
        print("active M1    :", f"{len(active):,}")
        print("M5           :", f"{len(m5):,}")
        print("longest flat :", longest_runs[:5])

    combined = (
        pd.concat(yearly, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    deltas = combined["timestamp"].diff().dt.total_seconds()

    normal_delta = int((deltas == 300).sum())
    gaps = deltas.loc[deltas > 300]

    gap_minutes = gaps / 60.0

    final_path = (
        OUT_DIR
        / "xauusd_bid_m5_2016_2026-06.parquet"
    )

    combined.to_parquet(final_path, index=False)

    audit.update(
        {
            "total_raw_m1_rows": total_raw,
            "total_invalid_rows": total_invalid,
            "total_stale_rows_removed": total_stale,
            "final_m5_rows": len(combined),
            "first_timestamp": combined["timestamp"].iloc[0].isoformat(),
            "last_timestamp": combined["timestamp"].iloc[-1].isoformat(),
            "continuous_m5_transitions": normal_delta,
            "gap_count": int(len(gaps)),
            "largest_gap_minutes": (
                float(gap_minutes.max())
                if len(gap_minutes)
                else 0.0
            ),
            "median_gap_minutes": (
                float(gap_minutes.median())
                if len(gap_minutes)
                else 0.0
            ),
        }
    )

    audit_path = OUT_DIR / "xauusd_data_audit.json"

    audit_path.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print()
    print("================================")
    print("FINAL XAUUSD DATASET")
    print("================================")
    print("raw M1 rows :", f"{total_raw:,}")
    print("invalid     :", f"{total_invalid:,}")
    print("stale rm    :", f"{total_stale:,}")
    print("final M5    :", f"{len(combined):,}")
    print("gaps        :", f"{len(gaps):,}")
    print("from        :", combined["timestamp"].iloc[0])
    print("to          :", combined["timestamp"].iloc[-1])
    print()
    print("Parquet:", final_path)
    print("Audit  :", audit_path)


if __name__ == "__main__":
    main()
