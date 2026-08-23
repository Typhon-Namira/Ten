from pathlib import Path
import json

import numpy as np
import pandas as pd


SRC = Path(
    "training/v2/data_lake/xau/"
    "xauusd_m5_bid_ask_2016_2026-06.parquet"
)

OUT = Path(
    "training/v6/data_lake/technical_state_v60"
)

WINDOWS = (
    3,    # 15m
    6,    # 30m
    12,   # 1h
    24,   # 2h
    48,   # 4h
    96,   # 8h
)


def safe_div(a, b):
    return (
        a
        / b.replace(0, np.nan)
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.0 TECHNICAL STATE ENGINE"
    )
    print("=" * 100)

    df = pd.read_parquet(
        SRC
    ).reset_index(drop=True)

    close = df["mid_close"].astype(
        np.float64
    )

    high = df["mid_high"].astype(
        np.float64
    )

    low = df["mid_low"].astype(
        np.float64
    )

    open_ = df["mid_open"].astype(
        np.float64
    )

    spread_bps = df[
        "spread_bps"
    ].astype(
        np.float64
    )

    f = pd.DataFrame(
        index=df.index
    )

    # --------------------------------------------------
    # BAR GEOMETRY
    # --------------------------------------------------

    f["spread_bps"] = spread_bps

    f["range_bps"] = (
        (high - low)
        / close
        * 10000.0
    )

    f["body_bps"] = (
        (close - open_)
        / close
        * 10000.0
    )

    body_high = pd.concat(
        [open_, close],
        axis=1,
    ).max(axis=1)

    body_low = pd.concat(
        [open_, close],
        axis=1,
    ).min(axis=1)

    f["upper_wick_bps"] = (
        (high - body_high)
        / close
        * 10000.0
    )

    f["lower_wick_bps"] = (
        (body_low - low)
        / close
        * 10000.0
    )

    f["close_position"] = safe_div(
        close - low,
        high - low,
    )

    # --------------------------------------------------
    # RETURNS / MOMENTUM
    # --------------------------------------------------

    for w in WINDOWS:
        f[f"return_{w}_bps"] = (
            close.pct_change(w)
            * 10000.0
        )

    ret1 = (
        close.pct_change()
        * 10000.0
    )

    f["return_1_bps"] = ret1

    # --------------------------------------------------
    # VOLATILITY / ATR-LIKE STATE
    # --------------------------------------------------

    for w in WINDOWS:
        f[f"atr_{w}_bps"] = (
            f["range_bps"]
            .rolling(
                w,
                min_periods=w,
            )
            .mean()
        )

        f[f"rv_{w}_bps"] = (
            ret1
            .rolling(
                w,
                min_periods=w,
            )
            .std()
        )

    f["compression_3_12"] = safe_div(
        f["atr_3_bps"],
        f["atr_12_bps"],
    )

    f["compression_6_24"] = safe_div(
        f["atr_6_bps"],
        f["atr_24_bps"],
    )

    f["compression_12_48"] = safe_div(
        f["atr_12_bps"],
        f["atr_48_bps"],
    )

    f["expansion_vs_atr12"] = safe_div(
        f["range_bps"],
        f["atr_12_bps"].shift(1),
    )

    f["spread_vs_atr12"] = safe_div(
        spread_bps,
        f["atr_12_bps"],
    )

    f["spread_ratio_24"] = safe_div(
        spread_bps,
        spread_bps
        .rolling(
            24,
            min_periods=24,
        )
        .median(),
    )

    # --------------------------------------------------
    # EMA / TREND
    # --------------------------------------------------

    emas = {}

    for w in WINDOWS:
        ema = close.ewm(
            span=w,
            adjust=False,
        ).mean()

        emas[w] = ema

        f[f"ema_gap_{w}_bps"] = (
            (close - ema)
            / close
            * 10000.0
        )

        f[f"ema_slope_{w}_bps"] = (
            ema.pct_change(3)
            * 10000.0
        )

    pairs = (
        (3, 12),
        (6, 24),
        (12, 48),
        (24, 96),
    )

    for fast, slow in pairs:
        f[
            f"ema_{fast}_{slow}_gap_bps"
        ] = (
            (
                emas[fast]
                - emas[slow]
            )
            / close
            * 10000.0
        )

    # --------------------------------------------------
    # PRIOR STRUCTURE / SUPPORT / RESISTANCE
    #
    # shift(1) is important:
    # levels contain ONLY bars known before current bar.
    # --------------------------------------------------

    prior_high = {}
    prior_low = {}

    for w in WINDOWS:
        ph = (
            high.shift(1)
            .rolling(
                w,
                min_periods=w,
            )
            .max()
        )

        pl = (
            low.shift(1)
            .rolling(
                w,
                min_periods=w,
            )
            .min()
        )

        prior_high[w] = ph
        prior_low[w] = pl

        f[
            f"distance_high_{w}_bps"
        ] = (
            (ph - close)
            / close
            * 10000.0
        )

        f[
            f"distance_low_{w}_bps"
        ] = (
            (close - pl)
            / close
            * 10000.0
        )

        f[
            f"range_position_{w}"
        ] = safe_div(
            close - pl,
            ph - pl,
        )

        f[
            f"breakout_up_{w}"
        ] = (
            close > ph
        ).astype(
            np.float32
        )

        f[
            f"breakout_down_{w}"
        ] = (
            close < pl
        ).astype(
            np.float32
        )

        # Liquidity sweep:
        # pierce old extreme but close back inside.
        f[
            f"sweep_up_{w}"
        ] = (
            (high > ph)
            & (close <= ph)
        ).astype(
            np.float32
        )

        f[
            f"sweep_down_{w}"
        ] = (
            (low < pl)
            & (close >= pl)
        ).astype(
            np.float32
        )

    # --------------------------------------------------
    # REJECTION
    # --------------------------------------------------

    range_safe = (
        high - low
    ).replace(
        0,
        np.nan,
    )

    f[
        "upper_rejection"
    ] = (
        (high - body_high)
        / range_safe
    )

    f[
        "lower_rejection"
    ] = (
        (body_low - low)
        / range_safe
    )

    # --------------------------------------------------
    # ACCELERATION
    # --------------------------------------------------

    f["momentum_accel_3_12"] = (
        f["return_3_bps"]
        - (
            f["return_12_bps"]
            / 4.0
        )
    )

    f["momentum_accel_6_24"] = (
        f["return_6_bps"]
        - (
            f["return_24_bps"]
            / 4.0
        )
    )

    # --------------------------------------------------
    # TREND ALIGNMENT
    # --------------------------------------------------

    trend_components = np.column_stack(
        [
            np.sign(
                f[
                    "ema_3_12_gap_bps"
                ].fillna(0)
            ),
            np.sign(
                f[
                    "ema_6_24_gap_bps"
                ].fillna(0)
            ),
            np.sign(
                f[
                    "ema_12_48_gap_bps"
                ].fillna(0)
            ),
            np.sign(
                f[
                    "ema_24_96_gap_bps"
                ].fillna(0)
            ),
        ]
    )

    f[
        "trend_alignment"
    ] = trend_components.mean(
        axis=1
    )

    # --------------------------------------------------
    # TIME / SESSION CONTEXT
    # --------------------------------------------------

    ts = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    hour = (
        ts.dt.hour
        + ts.dt.minute / 60.0
    )

    f["hour_sin"] = np.sin(
        2.0
        * np.pi
        * hour
        / 24.0
    )

    f["hour_cos"] = np.cos(
        2.0
        * np.pi
        * hour
        / 24.0
    )

    dow = ts.dt.dayofweek

    f["dow_sin"] = np.sin(
        2.0
        * np.pi
        * dow
        / 7.0
    )

    f["dow_cos"] = np.cos(
        2.0
        * np.pi
        * dow
        / 7.0
    )

    # Approximate session flags UTC.
    f["session_asia"] = (
        (ts.dt.hour >= 0)
        & (ts.dt.hour < 8)
    ).astype(
        np.float32
    )

    f["session_london"] = (
        (ts.dt.hour >= 7)
        & (ts.dt.hour < 16)
    ).astype(
        np.float32
    )

    f["session_newyork"] = (
        (ts.dt.hour >= 12)
        & (ts.dt.hour < 21)
    ).astype(
        np.float32
    )

    # --------------------------------------------------
    # QUALITY / FINITE MASK
    # --------------------------------------------------

    f = f.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid = (
        f.notna()
        .all(axis=1)
        .to_numpy()
    )

    names = f.columns.tolist()

    matrix = (
        f.fillna(0.0)
        .to_numpy(
            dtype=np.float32
        )
    )

    np.save(
        OUT
        / "technical_features.npy",
        matrix,
    )

    np.save(
        OUT
        / "technical_valid.npy",
        valid.astype(
            np.uint8
        ),
    )

    np.save(
        OUT
        / "timestamps_ns.npy",
        ts.astype(
            "int64"
        ).to_numpy(),
    )

    with open(
        OUT
        / "feature_names.json",
        "w",
    ) as fp:
        json.dump(
            names,
            fp,
            indent=2,
        )

    print(
        "Rows:",
        f"{len(matrix):,}",
    )

    print(
        "Features:",
        matrix.shape[1],
    )

    print(
        "Fully valid rows:",
        f"{valid.sum():,}",
        f"({valid.mean():.2%})",
    )

    print()
    print(
        "FEATURE FAMILIES"
    )
    print("-" * 100)

    for key in (
        "return",
        "atr",
        "rv_",
        "ema",
        "distance",
        "breakout",
        "sweep",
        "compression",
        "rejection",
        "session",
    ):
        count = sum(
            key in name
            for name in names
        )

        print(
            f"{key:<15}",
            count,
        )

    print()
    print(
        "Saved:",
        OUT,
    )


if __name__ == "__main__":
    main()
