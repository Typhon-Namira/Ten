from pathlib import Path

import numpy as np
import pandas as pd


M5_FILE = Path(
    "training/v2/data_lake/xau/"
    "xauusd_m5_bid_ask_2016_2026-06.parquet"
)

HIST_FILE = Path(
    "training/v5/data_lake/actionable_onset_v57/"
    "actionable_onset_targets_v57.parquet"
)

FUTURE_FILE = Path(
    "training/v5/data_lake/"
    "actionable_onset_2026h1_v57/"
    "actionable_onset_targets_v57.parquet"
)

OUT_DIR = Path(
    "training/v6/data_lake/"
    "large_move_v60"
)

OUT_FILE = (
    OUT_DIR
    / "large_move_targets_v60.parquet"
)

HORIZON_BARS = 6

PAIRS = (
    (20, 10),
    (30, 15),
    (40, 20),
)

FIVE_MIN_NS = int(
    pd.Timedelta(minutes=5).value
)

THIRTY_MIN_NS = int(
    pd.Timedelta(minutes=30).value
)


def first_hit(mask):
    any_hit = mask.any(axis=1)

    first = np.where(
        any_hit,
        mask.argmax(axis=1) + 1,
        99,
    )

    return first.astype(
        np.int16
    )


def race(tp_hit, sl_hit):
    out = np.full(
        len(tp_hit),
        -1,
        dtype=np.int8,
    )

    tp_first = (
        tp_hit < sl_hit
    )

    sl_first = (
        sl_hit < tp_hit
    )

    same = (
        (tp_hit == sl_hit)
        & (tp_hit < 99)
    )

    out[tp_first] = 1
    out[sl_first] = 0
    out[same] = -2

    return out


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.0 LARGE-MOVE "
        "FIRST-PASSAGE TARGET BUILDER"
    )
    print("=" * 100)

    m5 = pd.read_parquet(
        M5_FILE
    ).reset_index(
        drop=True
    )

    hist = pd.read_parquet(
        HIST_FILE
    )

    future = pd.read_parquet(
        FUTURE_FILE
    )

    anchors = pd.concat(
        [
            hist,
            future,
        ],
        ignore_index=True,
    )

    anchors = (
        anchors.sort_values(
            "source_row"
        )
        .reset_index(
            drop=True
        )
    )

    source = anchors[
        "source_row"
    ].to_numpy(
        np.int64
    )

    if source.max() >= len(m5):
        raise RuntimeError(
            "source_row exceeds M5 data"
        )

    ts = pd.to_datetime(
        m5["timestamp"],
        utc=True,
    ).astype(
        "int64"
    ).to_numpy()

    entry_bid = m5[
        "bid_close"
    ].to_numpy(
        np.float64
    )[source]

    entry_ask = m5[
        "ask_close"
    ].to_numpy(
        np.float64
    )[source]

    mid = (
        entry_bid + entry_ask
    ) / 2.0

    spread = (
        entry_ask - entry_bid
    )

    valid = (
        source + HORIZON_BARS
        < len(m5)
    )

    safe_source = np.minimum(
        source,
        len(m5)
        - HORIZON_BARS
        - 1,
    )

    future_idx = (
        safe_source[:, None]
        + np.arange(
            1,
            HORIZON_BARS + 1,
            dtype=np.int64,
        )[None, :]
    )

    future_ts = ts[
        future_idx
    ]

    current_ts = ts[
        safe_source
    ]

    # Exact contiguous 5-minute bars only.
    all_ts = np.concatenate(
        [
            current_ts[:, None],
            future_ts,
        ],
        axis=1,
    )

    contiguous = np.all(
        np.diff(
            all_ts,
            axis=1,
        )
        == FIVE_MIN_NS,
        axis=1,
    )

    horizon_ok = (
        future_ts[:, -1]
        - current_ts
        == THIRTY_MIN_NS
    )

    valid &= contiguous
    valid &= horizon_ok

    bid_high = m5[
        "bid_high"
    ].to_numpy(
        np.float64
    )[future_idx]

    bid_low = m5[
        "bid_low"
    ].to_numpy(
        np.float64
    )[future_idx]

    ask_high = m5[
        "ask_high"
    ].to_numpy(
        np.float64
    )[future_idx]

    ask_low = m5[
        "ask_low"
    ].to_numpy(
        np.float64
    )[future_idx]

    bid_close = m5[
        "bid_close"
    ].to_numpy(
        np.float64
    )[future_idx[:, -1]]

    ask_close = m5[
        "ask_close"
    ].to_numpy(
        np.float64
    )[future_idx[:, -1]]

    # Executable favorable/adverse excursions.
    long_mfe_bps = (
        (
            bid_high.max(axis=1)
            - entry_ask
        )
        / mid
        * 10000.0
    )

    long_mae_bps = (
        (
            entry_ask
            - bid_low.min(axis=1)
        )
        / mid
        * 10000.0
    )

    short_mfe_bps = (
        (
            entry_bid
            - ask_low.min(axis=1)
        )
        / mid
        * 10000.0
    )

    short_mae_bps = (
        (
            ask_high.max(axis=1)
            - entry_bid
        )
        / mid
        * 10000.0
    )

    long_terminal_bps = (
        (
            bid_close
            - entry_ask
        )
        / mid
        * 10000.0
    )

    short_terminal_bps = (
        (
            entry_bid
            - ask_close
        )
        / mid
        * 10000.0
    )

    out = anchors[
        [
            "source_row",
            "available_at",
            "m1_end_index",
            "year",
        ]
    ].copy()

    out["entry_bid"] = (
        entry_bid.astype(
            np.float32
        )
    )

    out["entry_ask"] = (
        entry_ask.astype(
            np.float32
        )
    )

    out["entry_mid"] = (
        mid.astype(
            np.float32
        )
    )

    out["spread_abs"] = (
        spread.astype(
            np.float32
        )
    )

    out["spread_bps_v6"] = (
        (
            spread
            / mid
            * 10000.0
        ).astype(
            np.float32
        )
    )

    out["horizon_valid"] = (
        valid.astype(
            np.int8
        )
    )

    out["long_mfe_bps"] = (
        long_mfe_bps.astype(
            np.float32
        )
    )

    out["long_mae_bps"] = (
        long_mae_bps.astype(
            np.float32
        )
    )

    out["short_mfe_bps"] = (
        short_mfe_bps.astype(
            np.float32
        )
    )

    out["short_mae_bps"] = (
        short_mae_bps.astype(
            np.float32
        )
    )

    out["long_terminal_bps"] = (
        long_terminal_bps.astype(
            np.float32
        )
    )

    out["short_terminal_bps"] = (
        short_terminal_bps.astype(
            np.float32
        )
    )

    for tp_bps, sl_bps in PAIRS:
        tp_abs = (
            mid
            * tp_bps
            / 10000.0
        )

        sl_abs = (
            mid
            * sl_bps
            / 10000.0
        )

        long_tp_level = (
            entry_ask + tp_abs
        )

        long_sl_level = (
            entry_ask - sl_abs
        )

        short_tp_level = (
            entry_bid - tp_abs
        )

        short_sl_level = (
            entry_bid + sl_abs
        )

        long_tp_hit = first_hit(
            bid_high
            >= long_tp_level[
                :, None
            ]
        )

        long_sl_hit = first_hit(
            bid_low
            <= long_sl_level[
                :, None
            ]
        )

        short_tp_hit = first_hit(
            ask_low
            <= short_tp_level[
                :, None
            ]
        )

        short_sl_hit = first_hit(
            ask_high
            >= short_sl_level[
                :, None
            ]
        )

        long_race = race(
            long_tp_hit,
            long_sl_hit,
        )

        short_race = race(
            short_tp_hit,
            short_sl_hit,
        )

        prefix = (
            f"tp{tp_bps}_sl{sl_bps}"
        )

        out[
            f"long_race_{prefix}"
        ] = long_race

        out[
            f"short_race_{prefix}"
        ] = short_race

        out[
            f"long_tp_min_{tp_bps}"
        ] = np.where(
            long_tp_hit < 99,
            long_tp_hit * 5,
            0,
        ).astype(
            np.int16
        )

        out[
            f"short_tp_min_{tp_bps}"
        ] = np.where(
            short_tp_hit < 99,
            short_tp_hit * 5,
            0,
        ).astype(
            np.int16
        )

    # Invalid horizons must never enter training.
    value_cols = [
        c
        for c in out.columns
        if (
            c.startswith(
                "long_race_"
            )
            or c.startswith(
                "short_race_"
            )
        )
    ]

    for c in value_cols:
        out.loc[
            ~valid,
            c,
        ] = -3

    out.to_parquet(
        OUT_FILE,
        index=False,
        compression="zstd",
    )

    print(
        "Anchors:",
        f"{len(out):,}",
    )

    print(
        "Valid 30m horizons:",
        f"{valid.sum():,}",
        f"({valid.mean():.2%})",
    )

    print()
    print(
        "EXECUTABLE EXCURSIONS"
    )
    print("-" * 100)

    v = valid

    print(
        "Median long MFE:",
        f"{np.median(long_mfe_bps[v]):.2f} bps",
    )

    print(
        "Median short MFE:",
        f"{np.median(short_mfe_bps[v]):.2f} bps",
    )

    print(
        "Median long MAE:",
        f"{np.median(long_mae_bps[v]):.2f} bps",
    )

    print(
        "Median short MAE:",
        f"{np.median(short_mae_bps[v]):.2f} bps",
    )

    for tp_bps, sl_bps in PAIRS:
        prefix = (
            f"tp{tp_bps}_sl{sl_bps}"
        )

        print()
        print(
            f"TP {tp_bps} / SL {sl_bps}"
        )

        for side in (
            "long",
            "short",
        ):
            x = out[
                f"{side}_race_{prefix}"
            ].to_numpy()

            mask = (
                valid
                & (x != -2)
            )

            print(
                side.upper(),
                "TP-first:",
                f"{(x[mask] == 1).mean():.2%}",
                "SL-first:",
                f"{(x[mask] == 0).mean():.2%}",
                "timeout:",
                f"{(x[mask] == -1).mean():.2%}",
                "ambiguous:",
                f"{(x[v] == -2).mean():.2%}",
            )

    print()
    print(
        "Saved:",
        OUT_FILE,
    )


if __name__ == "__main__":
    main()
