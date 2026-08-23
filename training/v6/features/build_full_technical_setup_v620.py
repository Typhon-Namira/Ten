from pathlib import Path
import json

import numpy as np
import pandas as pd


M5_FILE = Path(
    "training/v2/data_lake/xau/"
    "xauusd_m5_bid_ask_2016_2026-06.parquet"
)

BASE_DIR = Path(
    "training/v6/data_lake/technical_state_v60"
)

OUT = Path(
    "training/v6/data_lake/technical_setup_v620"
)


def clip01(x):
    return np.clip(
        np.asarray(x, dtype=np.float64),
        0.0,
        1.0,
    )


def pos(x):
    return clip01(x)


def neg(x):
    return clip01(-np.asarray(x))


def confluence(*items):
    z = np.column_stack(
        [
            clip01(x)
            for x in items
        ]
    )

    # Requires every component to contribute,
    # without destroying gradients between 0 and 1.
    return (
        0.5 * z.min(axis=1)
        + 0.5 * z.mean(axis=1)
    )


def rsi(close, period=14):
    diff = close.diff()

    gain = diff.clip(lower=0.0)
    loss = -diff.clip(upper=0.0)

    avg_gain = gain.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(0.0, np.nan)
    )

    return (
        100.0
        - (
            100.0
            / (1.0 + rs)
        )
    )


def build_timeframe_features(
    frame,
    prefix,
    minutes,
):
    x = frame.copy()

    close = x["mid_close"].astype(
        np.float64
    )

    open_ = x["mid_open"].astype(
        np.float64
    )

    high = x["mid_high"].astype(
        np.float64
    )

    low = x["mid_low"].astype(
        np.float64
    )

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr14_abs = tr.rolling(
        14,
        min_periods=14,
    ).mean()

    atr56_abs = tr.rolling(
        56,
        min_periods=56,
    ).mean()

    atr14_bps = (
        atr14_abs
        / close
        * 10000.0
    )

    atr56_bps = (
        atr56_abs
        / close
        * 10000.0
    )

    out = pd.DataFrame(
        index=x.index
    )

    out[
        f"{prefix}_range_bps"
    ] = (
        (high - low)
        / close
        * 10000.0
    )

    out[
        f"{prefix}_body_bps"
    ] = (
        (close - open_)
        / close
        * 10000.0
    )

    denom = (
        high - low
    ).replace(
        0.0,
        np.nan,
    )

    out[
        f"{prefix}_close_position"
    ] = (
        (close - low)
        / denom
    )

    body_high = pd.concat(
        [open_, close],
        axis=1,
    ).max(axis=1)

    body_low = pd.concat(
        [open_, close],
        axis=1,
    ).min(axis=1)

    out[
        f"{prefix}_upper_rejection"
    ] = (
        (high - body_high)
        / denom
    )

    out[
        f"{prefix}_lower_rejection"
    ] = (
        (body_low - low)
        / denom
    )

    for n in (
        1,
        3,
        6,
        12,
    ):
        out[
            f"{prefix}_return_{n}_bps"
        ] = (
            close.pct_change(n)
            * 10000.0
        )

    out[
        f"{prefix}_atr14_bps"
    ] = atr14_bps

    out[
        f"{prefix}_atr56_bps"
    ] = atr56_bps

    out[
        f"{prefix}_volatility_ratio"
    ] = (
        atr14_bps
        / atr56_bps.replace(
            0.0,
            np.nan,
        )
    )

    ema8 = close.ewm(
        span=8,
        adjust=False,
    ).mean()

    ema21 = close.ewm(
        span=21,
        adjust=False,
    ).mean()

    out[
        f"{prefix}_ema8_gap_bps"
    ] = (
        (close - ema8)
        / close
        * 10000.0
    )

    out[
        f"{prefix}_ema21_gap_bps"
    ] = (
        (close - ema21)
        / close
        * 10000.0
    )

    out[
        f"{prefix}_ema8_21_gap_bps"
    ] = (
        (ema8 - ema21)
        / close
        * 10000.0
    )

    ema21_slope = (
        ema21 - ema21.shift(3)
    )

    out[
        f"{prefix}_ema21_slope_bps"
    ] = (
        ema21_slope
        / close
        * 10000.0
    )

    trend_raw = (
        (
            ema8 - ema21
        )
        / (
            atr14_abs
            + 1e-9
        )
        + 0.50
        * (
            ema21_slope
            / (
                atr14_abs
                + 1e-9
            )
        )
    )

    out[
        f"{prefix}_trend_score"
    ] = np.tanh(
        trend_raw
    )

    current_high5 = high.rolling(
        5,
        min_periods=5,
    ).max()

    previous_high5 = (
        high.shift(5)
        .rolling(
            5,
            min_periods=5,
        )
        .max()
    )

    current_low5 = low.rolling(
        5,
        min_periods=5,
    ).min()

    previous_low5 = (
        low.shift(5)
        .rolling(
            5,
            min_periods=5,
        )
        .min()
    )

    structure_raw = (
        (
            current_high5
            - previous_high5
        )
        + (
            current_low5
            - previous_low5
        )
    ) / (
        2.0
        * (
            atr14_abs
            + 1e-9
        )
    )

    out[
        f"{prefix}_structure_score"
    ] = np.tanh(
        structure_raw
    )

    rsi14 = rsi(
        close,
        14,
    )

    out[
        f"{prefix}_rsi14"
    ] = rsi14

    out[
        f"{prefix}_rsi_centered"
    ] = (
        (rsi14 - 50.0)
        / 50.0
    )

    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    macd = (
        ema12 - ema26
    )

    signal = macd.ewm(
        span=9,
        adjust=False,
    ).mean()

    macd_hist = (
        macd - signal
    )

    out[
        f"{prefix}_macd_hist_bps"
    ] = (
        macd_hist
        / close
        * 10000.0
    )

    out[
        f"{prefix}_macd_strength"
    ] = np.tanh(
        macd_hist
        / (
            atr14_abs
            + 1e-9
        )
    )

    ma20 = close.rolling(
        20,
        min_periods=20,
    ).mean()

    sd20 = close.rolling(
        20,
        min_periods=20,
    ).std()

    out[
        f"{prefix}_bb_z"
    ] = (
        (close - ma20)
        / sd20.replace(
            0.0,
            np.nan,
        )
    )

    prior_high20 = (
        high.shift(1)
        .rolling(
            20,
            min_periods=20,
        )
        .max()
    )

    prior_low20 = (
        low.shift(1)
        .rolling(
            20,
            min_periods=20,
        )
        .min()
    )

    out[
        f"{prefix}_distance_high20_bps"
    ] = (
        (prior_high20 - close)
        / close
        * 10000.0
    )

    out[
        f"{prefix}_distance_low20_bps"
    ] = (
        (close - prior_low20)
        / close
        * 10000.0
    )

    breakout_up = (
        (close - prior_high20)
        / (
            atr14_abs
            + 1e-9
        )
    )

    breakout_down = (
        (prior_low20 - close)
        / (
            atr14_abs
            + 1e-9
        )
    )

    out[
        f"{prefix}_breakout_up_strength"
    ] = clip01(
        breakout_up / 2.0
    )

    out[
        f"{prefix}_breakout_down_strength"
    ] = clip01(
        breakout_down / 2.0
    )

    sweep_up = np.where(
        (
            (high > prior_high20)
            & (close <= prior_high20)
        ),
        (
            high - prior_high20
        )
        / (
            atr14_abs
            + 1e-9
        ),
        0.0,
    )

    sweep_down = np.where(
        (
            (low < prior_low20)
            & (close >= prior_low20)
        ),
        (
            prior_low20 - low
        )
        / (
            atr14_abs
            + 1e-9
        ),
        0.0,
    )

    out[
        f"{prefix}_sweep_up_strength"
    ] = clip01(
        sweep_up
    )

    out[
        f"{prefix}_sweep_down_strength"
    ] = clip01(
        sweep_down
    )

    compression_ratio = (
        atr14_abs
        / atr56_abs.replace(
            0.0,
            np.nan,
        )
    )

    out[
        f"{prefix}_compression_score"
    ] = clip01(
        (
            1.0
            - compression_ratio
        )
        / 0.50
    )

    expansion_ratio = (
        tr
        / atr14_abs.shift(1)
        .replace(
            0.0,
            np.nan,
        )
    )

    out[
        f"{prefix}_expansion_score"
    ] = clip01(
        (
            expansion_ratio
            - 1.0
        )
        / 2.0
    )

    momentum_raw = (
        out[
            f"{prefix}_return_3_bps"
        ]
        / (
            atr14_bps
            * np.sqrt(3.0)
            + 1e-9
        )
    )

    out[
        f"{prefix}_momentum_score"
    ] = np.tanh(
        momentum_raw
    )

    out[
        f"{prefix}_available_at"
    ] = (
        pd.to_datetime(
            x["timestamp"],
            utc=True,
        )
        + pd.Timedelta(
            minutes=minutes
        )
    )

    internal = {
        "high": high,
        "low": low,
        "close": close,
        "atr14_abs":
            atr14_abs,
        "prior_high20":
            prior_high20,
        "prior_low20":
            prior_low20,
    }

    return out, internal


def aggregate_tf(
    df,
    rule,
):
    x = (
        df.set_index(
            "timestamp"
        )
        .resample(
            rule,
            label="left",
            closed="left",
        )
        .agg(
            {
                "mid_open": "first",
                "mid_high": "max",
                "mid_low": "min",
                "mid_close": "last",
            }
        )
        .dropna(
            subset=[
                "mid_open",
                "mid_high",
                "mid_low",
                "mid_close",
            ]
        )
        .reset_index()
    )

    return x


def recent_event_value(
    values,
    event,
    max_lag=3,
):
    result = np.full(
        len(values),
        np.nan,
        dtype=np.float64,
    )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    event = np.asarray(
        event,
        dtype=bool,
    )

    for lag in range(
        1,
        max_lag + 1,
    ):
        candidate = np.roll(
            values,
            lag,
        )

        candidate_event = np.roll(
            event,
            lag,
        )

        candidate_event[
            :lag
        ] = False

        fill = (
            np.isnan(result)
            & candidate_event
        )

        result[
            fill
        ] = candidate[
            fill
        ]

    return result


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.2 FULL TECHNICAL SETUP ENGINE"
    )
    print("=" * 110)

    df = pd.read_parquet(
        M5_FILE
    ).reset_index(
        drop=True
    )

    df[
        "timestamp"
    ] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    print(
        "M5 rows:",
        f"{len(df):,}",
    )

    base = np.load(
        BASE_DIR
        / "technical_features.npy",
        mmap_mode="r",
    )

    base_valid = np.load(
        BASE_DIR
        / "technical_valid.npy",
        mmap_mode="r",
    )

    with open(
        BASE_DIR
        / "feature_names.json"
    ) as fp:
        base_names = json.load(
            fp
        )

    if len(base) != len(df):
        raise RuntimeError(
            "V6.0 technical rows != M5 rows"
        )

    anchor = pd.DataFrame(
        {
            "row":
                np.arange(
                    len(df),
                    dtype=np.int64,
                ),

            "available_at":
                df["timestamp"]
                + pd.Timedelta(
                    minutes=5
                ),
        }
    )

    feature_frames = []

    # --------------------------------------------------
    # TRUE M5
    # --------------------------------------------------

    m5_features, m5_internal = (
        build_timeframe_features(
            df[
                [
                    "timestamp",
                    "mid_open",
                    "mid_high",
                    "mid_low",
                    "mid_close",
                ]
            ],
            "m5",
            5,
        )
    )

    m5_feature_cols = [
        c
        for c in m5_features.columns
        if not c.endswith(
            "_available_at"
        )
    ]

    feature_frames.append(
        m5_features[
            m5_feature_cols
        ].reset_index(
            drop=True
        )
    )

    # --------------------------------------------------
    # TRUE COMPLETED M15 / H1 / H4
    # --------------------------------------------------

    for (
        rule,
        prefix,
        minutes,
    ) in (
        ("15min", "m15", 15),
        ("1h", "h1", 60),
        ("4h", "h4", 240),
    ):
        print(
            "Building",
            prefix.upper(),
            "..."
        )

        tf = aggregate_tf(
            df[
                [
                    "timestamp",
                    "mid_open",
                    "mid_high",
                    "mid_low",
                    "mid_close",
                ]
            ],
            rule,
        )

        tf_features, _ = (
            build_timeframe_features(
                tf,
                prefix,
                minutes,
            )
        )

        time_col = (
            f"{prefix}_available_at"
        )

        tf_features = (
            tf_features.rename(
                columns={
                    time_col:
                        "available_at"
                }
            )
            .sort_values(
                "available_at"
            )
        )

        merged = pd.merge_asof(
            anchor.sort_values(
                "available_at"
            ),
            tf_features,
            on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )

        merged = (
            merged.sort_values(
                "row"
            )
            .reset_index(
                drop=True
            )
        )

        cols = [
            c
            for c in merged.columns
            if c not in (
                "row",
                "available_at",
            )
        ]

        feature_frames.append(
            merged[cols]
        )

    f = pd.concat(
        feature_frames,
        axis=1,
    )

    # --------------------------------------------------
    # REAL BREAKOUT -> RETEST DETECTION ON M5
    # --------------------------------------------------

    close = np.asarray(
        m5_internal["close"],
        dtype=np.float64,
    )

    high = np.asarray(
        m5_internal["high"],
        dtype=np.float64,
    )

    low = np.asarray(
        m5_internal["low"],
        dtype=np.float64,
    )

    atr = np.asarray(
        m5_internal["atr14_abs"],
        dtype=np.float64,
    )

    ph = np.asarray(
        m5_internal["prior_high20"],
        dtype=np.float64,
    )

    pl = np.asarray(
        m5_internal["prior_low20"],
        dtype=np.float64,
    )

    bo_up_event = (
        close > ph
    )

    bo_down_event = (
        close < pl
    )

    up_level = recent_event_value(
        ph,
        bo_up_event,
        3,
    )

    down_level = recent_event_value(
        pl,
        bo_down_event,
        3,
    )

    up_distance = np.abs(
        low - up_level
    ) / (
        0.50 * atr
        + 1e-9
    )

    down_distance = np.abs(
        high - down_level
    ) / (
        0.50 * atr
        + 1e-9
    )

    retest_long = (
        np.isfinite(
            up_level
        )
        & (
            close
            >= up_level
        )
    )

    retest_short = (
        np.isfinite(
            down_level
        )
        & (
            close
            <= down_level
        )
    )

    # Sparse setup feature:
    # no valid recent breakout/retest => strength 0, not NaN.
    f[
        "m5_retest_long_strength"
    ] = np.where(
        retest_long,
        clip01(
            1.0 - up_distance
        ),
        0.0,
    )

    f[
        "m5_retest_short_strength"
    ] = np.where(
        retest_short,
        clip01(
            1.0 - down_distance
        ),
        0.0,
    )

    # --------------------------------------------------
    # NORMALIZED ANALYSIS COMPONENTS
    # --------------------------------------------------

    m5_bull = pos(
        f["m5_trend_score"]
    )

    m5_bear = neg(
        f["m5_trend_score"]
    )

    m15_bull = pos(
        f["m15_trend_score"]
    )

    m15_bear = neg(
        f["m15_trend_score"]
    )

    h1_bull = pos(
        f["h1_trend_score"]
    )

    h1_bear = neg(
        f["h1_trend_score"]
    )

    h4_bull = pos(
        f["h4_trend_score"]
    )

    h4_bear = neg(
        f["h4_trend_score"]
    )

    mtf_bull = confluence(
        m5_bull,
        m15_bull,
        h1_bull,
    )

    mtf_bear = confluence(
        m5_bear,
        m15_bear,
        h1_bear,
    )

    higher_bull = confluence(
        m15_bull,
        h1_bull,
        h4_bull,
    )

    higher_bear = confluence(
        m15_bear,
        h1_bear,
        h4_bear,
    )

    mom_long = pos(
        f["m5_momentum_score"]
    )

    mom_short = neg(
        f["m5_momentum_score"]
    )

    macd_long = pos(
        f["m5_macd_strength"]
    )

    macd_short = neg(
        f["m5_macd_strength"]
    )

    breakout_long = clip01(
        f[
            "m5_breakout_up_strength"
        ]
    )

    breakout_short = clip01(
        f[
            "m5_breakout_down_strength"
        ]
    )

    sweep_up = clip01(
        f[
            "m5_sweep_up_strength"
        ]
    )

    sweep_down = clip01(
        f[
            "m5_sweep_down_strength"
        ]
    )

    lower_rej = clip01(
        f[
            "m5_lower_rejection"
        ]
    )

    upper_rej = clip01(
        f[
            "m5_upper_rejection"
        ]
    )

    expansion = clip01(
        f[
            "m5_expansion_score"
        ]
    )

    compression = clip01(
        f[
            "m5_compression_score"
        ]
    )

    prior_compression = (
        pd.Series(
            compression
        )
        .shift(1)
        .rolling(
            6,
            min_periods=1,
        )
        .max()
        .fillna(0.0)
        .to_numpy()
    )

    atr_bps = np.asarray(
        f[
            "m5_atr14_bps"
        ],
        dtype=np.float64,
    )

    dist_low = np.asarray(
        f[
            "m5_distance_low20_bps"
        ],
        dtype=np.float64,
    )

    dist_high = np.asarray(
        f[
            "m5_distance_high20_bps"
        ],
        dtype=np.float64,
    )

    support_proximity = np.exp(
        -np.abs(
            dist_low
        )
        / (
            atr_bps
            + 1e-6
        )
    )

    resistance_proximity = np.exp(
        -np.abs(
            dist_high
        )
        / (
            atr_bps
            + 1e-6
        )
    )

    rsi5 = np.asarray(
        f["m5_rsi14"],
        dtype=np.float64,
    )

    rsi_oversold = clip01(
        (
            45.0 - rsi5
        )
        / 25.0
    )

    rsi_overbought = clip01(
        (
            rsi5 - 55.0
        )
        / 25.0
    )

    bb = np.asarray(
        f["m5_bb_z"],
        dtype=np.float64,
    )

    bb_low = clip01(
        (
            -bb - 0.75
        )
        / 1.50
    )

    bb_high = clip01(
        (
            bb - 0.75
        )
        / 1.50
    )

    pullback_long = clip01(
        (
            -np.asarray(
                f["m5_return_1_bps"],
                dtype=np.float64,
            )
        )
        / (
            atr_bps
            + 1e-6
        )
    )

    pullback_short = clip01(
        (
            np.asarray(
                f["m5_return_1_bps"],
                dtype=np.float64,
            )
        )
        / (
            atr_bps
            + 1e-6
        )
    )

    # --------------------------------------------------
    # NAMED TECHNICAL SETUPS
    # --------------------------------------------------

    setups = pd.DataFrame(
        index=f.index
    )

    setups[
        "setup_multitf_trend_long"
    ] = mtf_bull

    setups[
        "setup_multitf_trend_short"
    ] = mtf_bear

    setups[
        "setup_higher_tf_long"
    ] = higher_bull

    setups[
        "setup_higher_tf_short"
    ] = higher_bear

    setups[
        "setup_trend_continuation_long"
    ] = confluence(
        mtf_bull,
        mom_long,
        macd_long,
    )

    setups[
        "setup_trend_continuation_short"
    ] = confluence(
        mtf_bear,
        mom_short,
        macd_short,
    )

    setups[
        "setup_breakout_long"
    ] = confluence(
        breakout_long,
        expansion,
        mtf_bull,
    )

    setups[
        "setup_breakout_short"
    ] = confluence(
        breakout_short,
        expansion,
        mtf_bear,
    )

    setups[
        "setup_breakout_retest_long"
    ] = confluence(
        f[
            "m5_retest_long_strength"
        ],
        lower_rej,
        mtf_bull,
    )

    setups[
        "setup_breakout_retest_short"
    ] = confluence(
        f[
            "m5_retest_short_strength"
        ],
        upper_rej,
        mtf_bear,
    )

    setups[
        "setup_sweep_reversal_long"
    ] = confluence(
        sweep_down,
        lower_rej,
        rsi_oversold,
    )

    setups[
        "setup_sweep_reversal_short"
    ] = confluence(
        sweep_up,
        upper_rej,
        rsi_overbought,
    )

    setups[
        "setup_compression_breakout_long"
    ] = confluence(
        prior_compression,
        breakout_long,
        expansion,
    )

    setups[
        "setup_compression_breakout_short"
    ] = confluence(
        prior_compression,
        breakout_short,
        expansion,
    )

    setups[
        "setup_support_rejection_long"
    ] = confluence(
        support_proximity,
        lower_rej,
        pos(
            f[
                "m15_trend_score"
            ]
        ),
    )

    setups[
        "setup_resistance_rejection_short"
    ] = confluence(
        resistance_proximity,
        upper_rej,
        neg(
            f[
                "m15_trend_score"
            ]
        ),
    )

    setups[
        "setup_trend_pullback_long"
    ] = confluence(
        higher_bull,
        pullback_long,
        lower_rej,
    )

    setups[
        "setup_trend_pullback_short"
    ] = confluence(
        higher_bear,
        pullback_short,
        upper_rej,
    )

    setups[
        "setup_momentum_long"
    ] = confluence(
        mom_long,
        macd_long,
        expansion,
    )

    setups[
        "setup_momentum_short"
    ] = confluence(
        mom_short,
        macd_short,
        expansion,
    )

    setups[
        "setup_rsi_reversal_long"
    ] = confluence(
        rsi_oversold,
        lower_rej,
        support_proximity,
    )

    setups[
        "setup_rsi_reversal_short"
    ] = confluence(
        rsi_overbought,
        upper_rej,
        resistance_proximity,
    )

    setups[
        "setup_bb_reversion_long"
    ] = confluence(
        bb_low,
        lower_rej,
        support_proximity,
    )

    setups[
        "setup_bb_reversion_short"
    ] = confluence(
        bb_high,
        upper_rej,
        resistance_proximity,
    )

    setups[
        "setup_false_breakout_long"
    ] = confluence(
        sweep_down,
        lower_rej,
        expansion,
    )

    setups[
        "setup_false_breakout_short"
    ] = confluence(
        sweep_up,
        upper_rej,
        expansion,
    )

    # --------------------------------------------------
    # COMBINATION-READY CONTEXT
    # --------------------------------------------------

    setups[
        "context_bull_confluence"
    ] = confluence(
        mtf_bull,
        macd_long,
        mom_long,
        lower_rej,
    )

    setups[
        "context_bear_confluence"
    ] = confluence(
        mtf_bear,
        macd_short,
        mom_short,
        upper_rej,
    )

    # --------------------------------------------------
    # FINAL MATRIX
    # --------------------------------------------------

    f = f.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    setups = setups.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    new_names = (
        f.columns.tolist()
        + setups.columns.tolist()
    )

    new_matrix = np.concatenate(
        [
            f.fillna(0.0)
            .to_numpy(
                dtype=np.float32
            ),

            setups.fillna(0.0)
            .to_numpy(
                dtype=np.float32
            ),
        ],
        axis=1,
    )

    new_valid = (
        f.notna()
        .all(axis=1)
        .to_numpy()
        & setups.notna()
        .all(axis=1)
        .to_numpy()
    )

    final = np.concatenate(
        [
            np.asarray(
                base,
                dtype=np.float32,
            ),
            new_matrix,
        ],
        axis=1,
    )

    valid = (
        base_valid.astype(
            bool
        )
        & new_valid
    )

    final_names = (
        base_names
        + new_names
    )


    setup_names = [
        x
        for x in setups.columns
        if x.startswith("setup_")
    ]

    np.save(
        OUT / "technical_setup_features.npy",
        final,
    )

    np.save(
        OUT / "technical_setup_valid.npy",
        valid.astype(np.uint8),
    )

    np.save(
        OUT / "timestamps_ns.npy",
        anchor["available_at"]
        .astype("int64")
        .to_numpy(),
    )

    with open(
        OUT / "feature_names.json",
        "w",
    ) as fp:
        json.dump(
            final_names,
            fp,
            indent=2,
        )

    with open(
        OUT / "setup_names.json",
        "w",
    ) as fp:
        json.dump(
            setup_names,
            fp,
            indent=2,
        )

    metadata = {
        "rows": int(final.shape[0]),
        "features": int(final.shape[1]),
        "base_v60_features": int(
            len(base_names)
        ),
        "new_analysis_features": int(
            len(new_names)
        ),
        "named_setups": int(
            len(setup_names)
        ),
        "timeframes": [
            "M5",
            "M15",
            "H1",
            "H4",
        ],
        "causality": (
            "Higher timeframe features use only "
            "completed bars available at each "
            "M5 anchor."
        ),
    }

    with open(
        OUT / "metadata.json",
        "w",
    ) as fp:
        json.dump(
            metadata,
            fp,
            indent=2,
        )

    print()
    print("=" * 110)

    print(
        "FINAL ROWS:",
        f"{final.shape[0]:,}",
    )

    print(
        "FINAL FEATURES:",
        final.shape[1],
    )

    print(
        "BASE V6.0:",
        len(base_names),
    )

    print(
        "NEW FEATURES:",
        len(new_names),
    )

    print(
        "NAMED SETUPS:",
        len(setup_names),
    )

    print(
        "VALID:",
        f"{valid.sum():,}",
        f"({valid.mean():.2%})",
    )

    print()
    print("SETUPS")
    print("-" * 110)

    for name in setup_names:
        values = (
            setups[name]
            .fillna(0.0)
            .to_numpy()
        )

        print(
            f"{name:<42}"
            f" mean={values.mean():.4f}"
            f" p95={np.quantile(values, .95):.4f}"
            f" p99={np.quantile(values, .99):.4f}"
        )

    print()
    print(
        "Saved:",
        OUT,
    )


if __name__ == "__main__":
    main()
