from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = Path(
    "training/v6/data_lake/large_move_v60/"
    "large_move_targets_v60.parquet"
)

DEFAULT_M1_DIR = Path(
    "training/vendor/dukascopy_xau_m1/xauusd/bid/m1"
)


def parse_month(value: str) -> tuple[int, int]:
    try:
        y, m = value.split("-")
        year = int(y)
        month = int(m)
    except Exception as exc:
        raise argparse.ArgumentTypeError("month must be YYYY-MM") from exc
    if not 1 <= month <= 12:
        raise argparse.ArgumentTypeError("month must be YYYY-MM")
    return year, month


def month_bounds(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    if month == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
    else:
        end = pd.Timestamp(year=year, month=month + 1, day=1, tz="UTC")
    return start, end


def mode_stats(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.int64)
    if len(values) == 0:
        return {
            "n": 0,
            "mode": None,
            "mode_count": 0,
            "mode_share": 0.0,
            "unique": 0,
            "min": None,
            "max": None,
            "span": None,
        }

    counts = Counter(values.tolist())
    mode, mode_count = counts.most_common(1)[0]
    return {
        "n": int(len(values)),
        "mode": int(mode),
        "mode_count": int(mode_count),
        "mode_share": float(mode_count / len(values)),
        "unique": int(len(counts)),
        "min": int(values.min()),
        "max": int(values.max()),
        "span": int(values.max() - values.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=parse_month, default=parse_month("2023-01"))
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--shift-min", type=int, default=-10)
    parser.add_argument("--shift-max", type=int, default=10)
    args = parser.parse_args()

    year, month = args.month
    label = f"{year:04d}-{month:02d}"
    m1_file = args.m1_dir / f"xauusd_bid_m1_{year:04d}_{month:02d}.csv"

    print("TEN V6.9.3D RECOVERED M1 ALIGNMENT AUDIT")
    print("=" * 120)
    print("month:", label)
    print("target:", TARGET)
    print("m1:", m1_file)

    if not TARGET.exists():
        raise SystemExit(f"missing target: {TARGET}")
    if not m1_file.exists():
        raise SystemExit(f"missing M1 file: {m1_file}")

    target = pd.read_parquet(
        TARGET,
        columns=["available_at", "m1_end_index", "source_row", "year"],
    )

    available = pd.to_datetime(target["available_at"], utc=True, errors="coerce")
    start, end = month_bounds(year, month)
    mask = (available >= start) & (available < end)

    t = target.loc[mask].copy().reset_index(drop=True)
    t["available_at"] = available.loc[mask].reset_index(drop=True)

    m1 = pd.read_csv(
        m1_file,
        usecols=["timestamp", "open", "high", "low", "close"],
    )
    m1_ts = pd.to_datetime(
        pd.to_numeric(m1["timestamp"], errors="coerce"),
        unit="ms",
        utc=True,
        errors="coerce",
    )
    good = m1_ts.notna()
    m1 = m1.loc[good].reset_index(drop=True)
    m1_ts = m1_ts.loc[good].reset_index(drop=True)

    if not m1_ts.is_monotonic_increasing:
        raise RuntimeError("M1 timestamps are not sorted")
    if m1_ts.duplicated().any():
        raise RuntimeError("M1 timestamps contain duplicates")

    m1_ns = m1_ts.astype("int64").to_numpy()
    avail_ns = t["available_at"].astype("int64").to_numpy()
    old_idx = t["m1_end_index"].to_numpy(np.int64)

    print("target rows in month:", f"{len(t):,}")
    print("M1 rows in month:", f"{len(m1):,}")
    print("M1 first:", m1_ts.iloc[0])
    print("M1 last :", m1_ts.iloc[-1])
    print()

    results: list[dict[str, object]] = []
    one_min_ns = int(pd.Timedelta(minutes=1).value)

    for shift in range(args.shift_min, args.shift_max + 1):
        query = avail_ns + shift * one_min_ns
        local = np.searchsorted(m1_ns, query, side="right") - 1
        valid = (local >= 0) & (local < len(m1_ns))

        offsets = old_idx[valid] - local[valid]
        stats = mode_stats(offsets)
        row = {
            "shift_minutes": int(shift),
            "valid_rows": int(valid.sum()),
            **stats,
        }
        results.append(row)

    ranked = sorted(
        results,
        key=lambda x: (
            float(x["mode_share"]),
            -int(x["unique"]),
            -int(x["span"] or 0),
        ),
        reverse=True,
    )

    print("SHIFT SEARCH: old_m1_end_index - local_recovered_m1_position")
    print("-" * 120)
    print("shift   valid    mode_offset   mode_share   unique   span")
    for r in ranked[:10]:
        print(
            f"{r['shift_minutes']:>+5}m "
            f"{r['valid_rows']:>7,} "
            f"{str(r['mode']):>14} "
            f"{r['mode_share']:>10.2%} "
            f"{r['unique']:>8} "
            f"{str(r['span']):>8}"
        )

    best = ranked[0]
    best_shift = int(best["shift_minutes"])
    query = avail_ns + best_shift * one_min_ns
    local = np.searchsorted(m1_ns, query, side="right") - 1
    valid = (local >= 0) & (local < len(m1_ns))
    offsets = old_idx[valid] - local[valid]
    mode_offset = int(best["mode"])

    exact = valid & ((old_idx - local) == mode_offset)

    # A second, offset-independent test: differences between old indices
    # should equal differences between recovered local positions for ordered anchors.
    order = np.argsort(avail_ns)
    old_ordered = old_idx[order]
    local_ordered = local[order]
    good_pair = (
        (local_ordered[1:] >= 0)
        & (local_ordered[:-1] >= 0)
    )
    delta_equal = (
        np.diff(old_ordered)[good_pair]
        == np.diff(local_ordered)[good_pair]
    )
    delta_match = float(delta_equal.mean()) if len(delta_equal) else 0.0

    print()
    print("BEST ALIGNMENT")
    print("-" * 120)
    print("shift_minutes:", best_shift)
    print("global_index_offset:", mode_offset)
    print("exact_offset_share:", f"{exact.sum() / max(valid.sum(), 1):.4%}")
    print("delta_match_share:", f"{delta_match:.4%}")

    # Show first few reconstructed timestamps for human inspection.
    print()
    print("SAMPLE")
    print("-" * 120)
    shown = 0
    for i in np.flatnonzero(valid)[:8]:
        li = int(local[i])
        print(
            "available=",
            t.loc[i, "available_at"],
            " recovered_m1=",
            m1_ts.iloc[li],
            " old_idx=",
            int(old_idx[i]),
            " local_idx=",
            li,
            " offset=",
            int(old_idx[i] - li),
        )
        shown += 1

    if float(best["mode_share"]) >= 0.995 and delta_match >= 0.995:
        verdict = "EXACT_ALIGNMENT_RECOVERED"
    elif float(best["mode_share"]) >= 0.95 and delta_match >= 0.95:
        verdict = "NEAR_EXACT_ALIGNMENT_RECOVERED"
    else:
        verdict = "ALIGNMENT_NOT_REPRODUCED"

    summary = {
        "version": "v6.9.3d",
        "month": label,
        "target_rows": int(len(t)),
        "m1_rows": int(len(m1)),
        "best_shift_minutes": best_shift,
        "global_index_offset": mode_offset,
        "mode_share": float(best["mode_share"]),
        "delta_match_share": delta_match,
        "verdict": verdict,
    }

    print()
    print("=" * 120)
    print("VERDICT:", verdict)
    print(json.dumps(summary, indent=2))

    if verdict == "EXACT_ALIGNMENT_RECOVERED":
        print("GO: recovered Dukascopy M1 ordering matches the historical index semantics.")
    elif verdict == "NEAR_EXACT_ALIGNMENT_RECOVERED":
        print("PARTIAL: mostly consistent; inspect mismatches before a multi-year download.")
    else:
        print("STOP: do not trust historical m1_end_index against the recovered raw M1 yet.")


if __name__ == "__main__":
    main()
