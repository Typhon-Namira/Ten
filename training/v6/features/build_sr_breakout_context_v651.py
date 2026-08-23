from pathlib import Path
import json

import numpy as np
import pandas as pd


M5_FILE = Path(
    "training/v2/data_lake/xau/"
    "xauusd_m5_bid_ask_2016_2026-06.parquet"
)

BASE_DIR = Path(
    "training/v6/data_lake/"
    "technical_state_v650"
)

OUT = Path(
    "training/v6/data_lake/"
    "technical_state_v651"
)

STEP = pd.Timedelta(
    minutes=5
)

TOUCH_LOOKBACK = 48
RETEST_WINDOW = 6


def clip01(x):
    return np.clip(
        np.asarray(
            x,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )


def aggregate_tf(
    df,
    rule,
):
    return (
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
                "mid_open":
                    "first",
                "mid_high":
                    "max",
                "mid_low":
                    "min",
                "mid_close":
                    "last",
            }
        )
        .dropna()
        .reset_index()
    )


def true_range(
    high,
    low,
    close,
):
    prev = close.shift(
        1
    )

    return pd.concat(
        [
            high - low,
            (
                high - prev
            ).abs(),
            (
                low - prev
            ).abs(),
        ],
        axis=1,
    ).max(
        axis=1
    )


def bars_since(
    event,
):
    event = np.asarray(
        event,
        dtype=bool,
    )

    idx = np.arange(
        len(event),
        dtype=np.int64,
    )

    last = np.where(
        event,
        idx,
        -1,
    )

    last = np.maximum.accumulate(
        last
    )

    age = idx - last

    age[
        last < 0
    ] = -1

    return age.astype(
        np.float32
    )


def confirmed_levels(
    high,
    low,
    left=3,
    right=3,
):
    window = (
        left
        + right
        + 1
    )

    max_ = high.rolling(
        window,
        center=True,
        min_periods=window,
    ).max()

    min_ = low.rolling(
        window,
        center=True,
        min_periods=window,
    ).min()

    raw_high = high.eq(
        max_
    )

    raw_low = low.eq(
        min_
    )

    high_event = raw_high.shift(
        right,
        fill_value=False,
    ).astype(
        bool
    )

    low_event = raw_low.shift(
        right,
        fill_value=False,
    ).astype(
        bool
    )

    high_value = (
        high.where(
            raw_high
        )
        .shift(
            right
        )
    )

    low_value = (
        low.where(
            raw_low
        )
        .shift(
            right
        )
    )

    last_high = (
        high_value.where(
            high_event
        )
        .ffill()
    )

    last_low = (
        low_value.where(
            low_event
        )
        .ffill()
    )

    return (
        high_event,
        low_event,
        last_high,
        last_low,
    )


def touch_counts(
    high,
    low,
    level_high,
    level_low,
    atr,
    age_high,
    age_low,
):
    tolerance = (
        atr * 0.15
    )

    high_count = np.where(
        np.isfinite(
            level_high
        ),
        1.0,
        0.0,
    )

    low_count = np.where(
        np.isfinite(
            level_low
        ),
        1.0,
        0.0,
    )

    for lag in range(
        1,
        TOUCH_LOOKBACK + 1,
    ):
        past_high = high.shift(
            lag
        )

        past_low = low.shift(
            lag
        )

        high_touch = (
            (
                past_high
                - level_high
            ).abs()
            <= tolerance
        )

        low_touch = (
            (
                past_low
                - level_low
            ).abs()
            <= tolerance
        )

        high_touch &= (
            age_high >= lag
        )

        low_touch &= (
            age_low >= lag
        )

        high_count += (
            high_touch.fillna(
                False
            )
            .to_numpy(
                np.float64
            )
        )

        low_count += (
            low_touch.fillna(
                False
            )
            .to_numpy(
                np.float64
            )
        )

    return (
        high_count,
        low_count,
    )


def sr_context(
    frame,
    prefix,
    minutes,
):
    x = frame.copy()

    o = x[
        "mid_open"
    ].astype(
        np.float64
    )

    h = x[
        "mid_high"
    ].astype(
        np.float64
    )

    l = x[
        "mid_low"
    ].astype(
        np.float64
    )

    c = x[
        "mid_close"
    ].astype(
        np.float64
    )

    tr = true_range(
        h,
        l,
        c,
    )

    atr = tr.rolling(
        14,
        min_periods=14,
    ).mean()

    (
        resistance_event,
        support_event,
        resistance,
        support,
    ) = confirmed_levels(
        h,
        l,
    )

    resistance_age = bars_since(
        resistance_event
    )

    support_age = bars_since(
        support_event
    )

    (
        resistance_touches,
        support_touches,
    ) = touch_counts(
        h,
        l,
        resistance,
        support,
        atr,
        resistance_age,
        support_age,
    )

    resistance_strength = (
        clip01(
            resistance_touches
            / 5.0
        )
        * np.exp(
            -np.maximum(
                resistance_age,
                0.0,
            )
            / 288.0
        )
    )

    support_strength = (
        clip01(
            support_touches
            / 5.0
        )
        * np.exp(
            -np.maximum(
                support_age,
                0.0,
            )
            / 288.0
        )
    )

    resistance_distance = (
        resistance - c
    ) / (
        atr + 1e-9
    )

    support_distance = (
        c - support
    ) / (
        atr + 1e-9
    )

    resistance_proximity = clip01(
        1.0
        - np.abs(
            resistance_distance
        )
        / 2.0
    )

    support_proximity = clip01(
        1.0
        - np.abs(
            support_distance
        )
        / 2.0
    )

    # Use levels already known on
    # the previous bar.
    resistance_ref = (
        resistance.shift(
            1
        )
    )

    support_ref = (
        support.shift(
            1
        )
    )

    prev_close = c.shift(
        1
    )

    breakout_up = (
        resistance_ref.notna()
        & (
            c > resistance_ref
        )
        & (
            prev_close
            <= resistance_ref
        )
    )

    breakout_down = (
        support_ref.notna()
        & (
            c < support_ref
        )
        & (
            prev_close
            >= support_ref
        )
    )

    breakout_up_disp = np.where(
        breakout_up,
        clip01(
            (
                c - resistance_ref
            )
            / (
                atr * 0.75
                + 1e-9
            )
        ),
        0.0,
    )

    breakout_down_disp = np.where(
        breakout_down,
        clip01(
            (
                support_ref - c
            )
            / (
                atr * 0.75
                + 1e-9
            )
        ),
        0.0,
    )

    breakout_up_body = np.where(
        breakout_up,
        clip01(
            (
                c
                - np.maximum(
                    o,
                    resistance_ref,
                )
            )
            / (
                atr * 0.50
                + 1e-9
            )
        ),
        0.0,
    )

    breakout_down_body = np.where(
        breakout_down,
        clip01(
            (
                np.minimum(
                    o,
                    support_ref,
                )
                - c
            )
            / (
                atr * 0.50
                + 1e-9
            )
        ),
        0.0,
    )

    up_break_level = (
        resistance_ref.where(
            breakout_up
        )
        .ffill()
    )

    down_break_level = (
        support_ref.where(
            breakout_down
        )
        .ffill()
    )

    up_break_age = bars_since(
        breakout_up
    )

    down_break_age = bars_since(
        breakout_down
    )

    recent_up = (
        (up_break_age >= 0)
        & (
            up_break_age
            <= RETEST_WINDOW
        )
    )

    recent_down = (
        (down_break_age >= 0)
        & (
            down_break_age
            <= RETEST_WINDOW
        )
    )

    up_event_strength = (
        pd.Series(
            np.where(
                breakout_up,
                breakout_up_disp,
                np.nan,
            ),
            index=x.index,
        )
        .ffill()
        .to_numpy(
            np.float64
        )
    )

    down_event_strength = (
        pd.Series(
            np.where(
                breakout_down,
                breakout_down_disp,
                np.nan,
            ),
            index=x.index,
        )
        .ffill()
        .to_numpy(
            np.float64
        )
    )

    recent_up_strength = np.where(
        recent_up,
        up_event_strength
        * np.exp(
            -np.maximum(
                up_break_age,
                0.0,
            )
            / 6.0
        ),
        0.0,
    )

    recent_down_strength = np.where(
        recent_down,
        down_event_strength
        * np.exp(
            -np.maximum(
                down_break_age,
                0.0,
            )
            / 6.0
        ),
        0.0,
    )

    long_touch = (
        recent_up
        & (
            l
            <= (
                up_break_level
                + atr * 0.20
            )
        )
        & (
            h
            >= (
                up_break_level
                - atr * 0.10
            )
        )
    )

    short_touch = (
        recent_down
        & (
            h
            >= (
                down_break_level
                - atr * 0.20
            )
        )
        & (
            l
            <= (
                down_break_level
                + atr * 0.10
            )
        )
    )

    long_hold = (
        c > up_break_level
    )

    short_hold = (
        c < down_break_level
    )

    long_retest = (
        long_touch
        & long_hold
        & (
            up_break_age >= 1
        )
    )

    short_retest = (
        short_touch
        & short_hold
        & (
            down_break_age >= 1
        )
    )

    long_precision = clip01(
        1.0
        - (
            l - up_break_level
        ).abs()
        / (
            atr * 0.40
            + 1e-9
        )
    )

    short_precision = clip01(
        1.0
        - (
            h - down_break_level
        ).abs()
        / (
            atr * 0.40
            + 1e-9
        )
    )

    long_rejection = clip01(
        (
            c - up_break_level
        )
        / (
            atr * 0.35
            + 1e-9
        )
    )

    short_rejection = clip01(
        (
            down_break_level - c
        )
        / (
            atr * 0.35
            + 1e-9
        )
    )

    retest_long_quality = np.where(
        long_retest,
        long_precision
        * (
            0.5
            + 0.5
            * long_rejection
        ),
        0.0,
    )

    retest_short_quality = np.where(
        short_retest,
        short_precision
        * (
            0.5
            + 0.5
            * short_rejection
        ),
        0.0,
    )

    retest_long_depth = np.where(
        long_retest,
        clip01(
            (
                up_break_level
                - l
            )
            / (
                atr * 0.50
                + 1e-9
            )
        ),
        0.0,
    )

    retest_short_depth = np.where(
        short_retest,
        clip01(
            (
                h
                - down_break_level
            )
            / (
                atr * 0.50
                + 1e-9
            )
        ),
        0.0,
    )

    retest_long_delay = np.where(
        long_retest,
        up_break_age
        / RETEST_WINDOW,
        0.0,
    )

    retest_short_delay = np.where(
        short_retest,
        down_break_age
        / RETEST_WINDOW,
        0.0,
    )

    long_acceptance = np.zeros(
        len(x),
        dtype=np.float64,
    )

    short_acceptance = np.zeros(
        len(x),
        dtype=np.float64,
    )

    for lag in range(
        3
    ):
        long_acceptance += (
            c.shift(
                lag
            )
            > up_break_level
        ).fillna(
            False
        ).to_numpy(
            np.float64
        )

        short_acceptance += (
            c.shift(
                lag
            )
            < down_break_level
        ).fillna(
            False
        ).to_numpy(
            np.float64
        )

    long_acceptance /= 3.0
    short_acceptance /= 3.0

    long_acceptance = np.where(
        recent_up,
        long_acceptance,
        0.0,
    )

    short_acceptance = np.where(
        recent_down,
        short_acceptance,
        0.0,
    )

    long_failure = np.where(
        recent_up
        & (
            c < up_break_level
        ),
        clip01(
            (
                up_break_level - c
            )
            / (
                atr * 0.50
                + 1e-9
            )
        ),
        0.0,
    )

    short_failure = np.where(
        recent_down
        & (
            c > down_break_level
        ),
        clip01(
            (
                c - down_break_level
            )
            / (
                atr * 0.50
                + 1e-9
            )
        ),
        0.0,
    )

    out = pd.DataFrame(
        index=x.index
    )

    out[
        f"{prefix}_resistance_distance_atr"
    ] = resistance_distance

    out[
        f"{prefix}_support_distance_atr"
    ] = support_distance

    out[
        f"{prefix}_resistance_proximity"
    ] = resistance_proximity

    out[
        f"{prefix}_support_proximity"
    ] = support_proximity

    out[
        f"{prefix}_resistance_age"
    ] = resistance_age

    out[
        f"{prefix}_support_age"
    ] = support_age

    out[
        f"{prefix}_resistance_touch_count"
    ] = resistance_touches

    out[
        f"{prefix}_support_touch_count"
    ] = support_touches

    out[
        f"{prefix}_resistance_strength"
    ] = resistance_strength

    out[
        f"{prefix}_support_strength"
    ] = support_strength

    out[
        f"{prefix}_breakout_up_displacement"
    ] = breakout_up_disp

    out[
        f"{prefix}_breakout_down_displacement"
    ] = breakout_down_disp

    out[
        f"{prefix}_breakout_up_body_through"
    ] = breakout_up_body

    out[
        f"{prefix}_breakout_down_body_through"
    ] = breakout_down_body

    out[
        f"{prefix}_recent_breakout_up_strength"
    ] = recent_up_strength

    out[
        f"{prefix}_recent_breakout_down_strength"
    ] = recent_down_strength

    out[
        f"{prefix}_retest_long_quality"
    ] = retest_long_quality

    out[
        f"{prefix}_retest_short_quality"
    ] = retest_short_quality

    out[
        f"{prefix}_retest_long_depth"
    ] = retest_long_depth

    out[
        f"{prefix}_retest_short_depth"
    ] = retest_short_depth

    out[
        f"{prefix}_retest_long_delay"
    ] = retest_long_delay

    out[
        f"{prefix}_retest_short_delay"
    ] = retest_short_delay

    out[
        f"{prefix}_breakout_long_acceptance"
    ] = long_acceptance

    out[
        f"{prefix}_breakout_short_acceptance"
    ] = short_acceptance

    out[
        f"{prefix}_breakout_long_failure"
    ] = long_failure

    out[
        f"{prefix}_breakout_short_failure"
    ] = short_failure

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

    return out


def session_dynamics(
    df,
):
    ts = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    available = (
        ts + STEP
    )

    h = df[
        "mid_high"
    ].astype(
        np.float64
    )

    l = df[
        "mid_low"
    ].astype(
        np.float64
    )

    c = df[
        "mid_close"
    ].astype(
        np.float64
    )

    utc_day = ts.dt.floor(
        "D"
    )

    utc_hour = (
        ts.dt.hour
        + ts.dt.minute
        / 60.0
    )

    available_hour = (
        available.dt.hour
        + available.dt.minute
        / 60.0
    )

    asia_mask = (
        (utc_hour >= 0.0)
        & (
            utc_hour < 8.0
        )
    )

    asia_daily = (
        pd.DataFrame(
            {
                "day":
                    utc_day[
                        asia_mask
                    ],
                "high":
                    h[
                        asia_mask
                    ],
                "low":
                    l[
                        asia_mask
                    ],
            }
        )
        .groupby(
            "day"
        )
        .agg(
            high=(
                "high",
                "max",
            ),
            low=(
                "low",
                "min",
            ),
        )
    )

    asia_high = utc_day.map(
        asia_daily[
            "high"
        ]
    )

    asia_low = utc_day.map(
        asia_daily[
            "low"
        ]
    )

    asia_ready = (
        available_hour
        >= 8.0
    )

    london_time = (
        ts.dt.tz_convert(
            "Europe/London"
        )
    )

    london_day = (
        london_time.dt.date
    )

    london_hour = (
        london_time.dt.hour
        + london_time.dt.minute
        / 60.0
    )

    london_mask = (
        (london_hour >= 8.0)
        & (
            london_hour < 16.0
        )
    )

    london_run_high = (
        h.where(
            london_mask
        )
        .groupby(
            london_day
        )
        .cummax()
        .groupby(
            london_day
        )
        .ffill()
    )

    london_run_low = (
        l.where(
            london_mask
        )
        .groupby(
            london_day
        )
        .cummin()
        .groupby(
            london_day
        )
        .ffill()
    )

    london_daily = (
        pd.DataFrame(
            {
                "day":
                    pd.Series(
                        london_day
                    )[
                        london_mask
                    ],
                "high":
                    h[
                        london_mask
                    ],
                "low":
                    l[
                        london_mask
                    ],
            }
        )
        .groupby(
            "day"
        )
        .agg(
            high=(
                "high",
                "max",
            ),
            low=(
                "low",
                "min",
            ),
        )
    )

    prev_london_high = pd.Series(
        london_day
    ).map(
        london_daily[
            "high"
        ].shift(
            1
        )
    )

    prev_london_low = pd.Series(
        london_day
    ).map(
        london_daily[
            "low"
        ].shift(
            1
        )
    )

    ny_time = ts.dt.tz_convert(
        "America/New_York"
    )

    ny_day = ny_time.dt.date

    ny_hour = (
        ny_time.dt.hour
        + ny_time.dt.minute
        / 60.0
    )

    ny_mask = (
        (ny_hour >= 8.0)
        & (
            ny_hour < 17.0
        )
    )

    ny_run_high = (
        h.where(
            ny_mask
        )
        .groupby(
            ny_day
        )
        .cummax()
        .groupby(
            ny_day
        )
        .ffill()
    )

    ny_run_low = (
        l.where(
            ny_mask
        )
        .groupby(
            ny_day
        )
        .cummin()
        .groupby(
            ny_day
        )
        .ffill()
    )

    asia_range = (
        asia_high - asia_low
    )

    london_range = (
        london_run_high
        - london_run_low
    )

    ny_range = (
        ny_run_high
        - ny_run_low
    )

    prior_london_high = (
        london_run_high.shift(
            1
        )
    )

    prior_london_low = (
        london_run_low.shift(
            1
        )
    )

    prev_c = c.shift(
        1
    )

    out = pd.DataFrame(
        index=df.index
    )

    out[
        "session_v651_asia_range_bps"
    ] = np.where(
        asia_ready,
        asia_range
        / c
        * 10000.0,
        0.0,
    )

    out[
        "session_v651_london_range_bps"
    ] = np.where(
        london_run_high.notna(),
        london_range
        / c
        * 10000.0,
        0.0,
    )

    out[
        "session_v651_ny_range_bps"
    ] = np.where(
        ny_run_high.notna(),
        ny_range
        / c
        * 10000.0,
        0.0,
    )

    out[
        "session_v651_london_vs_asia_range"
    ] = np.where(
        asia_ready
        & london_run_high.notna(),
        london_range
        / (
            asia_range
            + 1e-9
        ),
        0.0,
    )

    out[
        "session_v651_ny_vs_london_range"
    ] = np.where(
        ny_run_high.notna()
        & london_run_high.notna(),
        ny_range
        / (
            london_range
            + 1e-9
        ),
        0.0,
    )

    out[
        "session_v651_london_position"
    ] = np.where(
        london_run_high.notna(),
        (
            c - london_run_low
        )
        / (
            london_range
            + 1e-9
        ),
        0.0,
    )

    out[
        "session_v651_ny_position"
    ] = np.where(
        ny_run_high.notna(),
        (
            c - ny_run_low
        )
        / (
            ny_range
            + 1e-9
        ),
        0.0,
    )

    asia_mid = (
        asia_high + asia_low
    ) / 2.0

    london_mid = (
        london_run_high
        + london_run_low
    ) / 2.0

    out[
        "session_v651_from_asia_mid_bps"
    ] = np.where(
        asia_ready,
        (
            c - asia_mid
        )
        / c
        * 10000.0,
        0.0,
    )

    out[
        "session_v651_from_london_mid_bps"
    ] = np.where(
        london_run_high.notna(),
        (
            c - london_mid
        )
        / c
        * 10000.0,
        0.0,
    )

    out[
        "session_v651_prev_london_high_bps"
    ] = (
        prev_london_high
        - c
    ) / c * 10000.0

    out[
        "session_v651_prev_london_low_bps"
    ] = (
        c
        - prev_london_low
    ) / c * 10000.0

    out[
        "session_v651_break_asia_high"
    ] = np.where(
        asia_ready,
        (
            (c > asia_high)
            & (
                prev_c <= asia_high
            )
        ).astype(
            np.float64
        ),
        0.0,
    )

    out[
        "session_v651_break_asia_low"
    ] = np.where(
        asia_ready,
        (
            (c < asia_low)
            & (
                prev_c >= asia_low
            )
        ).astype(
            np.float64
        ),
        0.0,
    )

    out[
        "session_v651_sweep_asia_high"
    ] = np.where(
        asia_ready,
        (
            (h > asia_high)
            & (
                c < asia_high
            )
        ).astype(
            np.float64
        ),
        0.0,
    )

    out[
        "session_v651_sweep_asia_low"
    ] = np.where(
        asia_ready,
        (
            (l < asia_low)
            & (
                c > asia_low
            )
        ).astype(
            np.float64
        ),
        0.0,
    )

    out[
        "session_v651_ny_break_london_high"
    ] = np.where(
        ny_mask
        & prior_london_high.notna(),
        (
            (c > prior_london_high)
            & (
                prev_c
                <= prior_london_high
            )
        ).astype(
            np.float64
        ),
        0.0,
    )

    out[
        "session_v651_ny_break_london_low"
    ] = np.where(
        ny_mask
        & prior_london_low.notna(),
        (
            (c < prior_london_low)
            & (
                prev_c
                >= prior_london_low
            )
        ).astype(
            np.float64
        ),
        0.0,
    )

    out[
        "session_v651_ny_sweep_london_high"
    ] = np.where(
        ny_mask
        & prior_london_high.notna(),
        (
            (h > prior_london_high)
            & (
                c < prior_london_high
            )
        ).astype(
            np.float64
        ),
        0.0,
    )

    out[
        "session_v651_ny_sweep_london_low"
    ] = np.where(
        ny_mask
        & prior_london_low.notna(),
        (
            (l < prior_london_low)
            & (
                c > prior_london_low
            )
        ).astype(
            np.float64
        ),
        0.0,
    )

    return out


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.5B "
        "S/R + BREAKOUT + RETEST "
        "+ SESSION DYNAMICS ENGINE"
    )

    print(
        "=" * 120
    )

    df = pd.read_parquet(
        M5_FILE
    ).reset_index(
        drop=True
    )

    df[
        "timestamp"
    ] = pd.to_datetime(
        df[
            "timestamp"
        ],
        utc=True,
    )

    base = np.load(
        BASE_DIR
        / "technical_features_v650.npy",
        mmap_mode="r",
    )

    base_valid = np.load(
        BASE_DIR
        / "technical_valid_v650.npy",
        mmap_mode="r",
    ).astype(
        bool
    )

    timestamps = np.load(
        BASE_DIR
        / "timestamps_ns.npy",
        mmap_mode="r",
    ).astype(
        np.int64
    )

    with open(
        BASE_DIR
        / "feature_names.json"
    ) as f:
        base_names = json.load(
            f
        )

    if len(df) != len(base):
        raise RuntimeError(
            "M5/base v650 row mismatch"
        )

    anchor = pd.DataFrame(
        {
            "row":
                np.arange(
                    len(df),
                    dtype=np.int64,
                ),

            "available_at":
                df[
                    "timestamp"
                ]
                + STEP,
        }
    )

    frames = []

    print(
        "Building M5 S/R context ..."
    )

    m5 = sr_context(
        df[
            [
                "timestamp",
                "mid_open",
                "mid_high",
                "mid_low",
                "mid_close",
            ]
        ],
        "m5v651",
        5,
    )

    frames.append(
        m5.drop(
            columns=[
                "m5v651_available_at"
            ]
        ).reset_index(
            drop=True
        )
    )

    for (
        rule,
        prefix,
        minutes,
    ) in (
        (
            "15min",
            "m15v651",
            15,
        ),
        (
            "1h",
            "h1v651",
            60,
        ),
        (
            "4h",
            "h4v651",
            240,
        ),
    ):
        print(
            "Building",
            prefix.upper(),
            "S/R context ..."
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

        feat = sr_context(
            tf,
            prefix,
            minutes,
        ).rename(
            columns={
                f"{prefix}_available_at":
                    "available_at"
            }
        )

        merged = pd.merge_asof(
            anchor.sort_values(
                "available_at"
            ),
            feat.sort_values(
                "available_at"
            ),
            on="available_at",
            direction="backward",
            allow_exact_matches=True,
        ).sort_values(
            "row"
        )

        cols = [
            col
            for col in merged.columns
            if col not in (
                "row",
                "available_at",
            )
        ]

        frames.append(
            merged[
                cols
            ].reset_index(
                drop=True
            )
        )

    print(
        "Building session dynamics ..."
    )

    frames.append(
        session_dynamics(
            df
        ).reset_index(
            drop=True
        )
    )

    new = pd.concat(
        frames,
        axis=1,
    ).replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    new_names = (
        new.columns.tolist()
    )

    # Sparse event features have
    # already been explicitly set to 0.
    # Remaining NaN means genuine
    # warm-up / unavailable context.
    new_valid = (
        new.notna()
        .all(
            axis=1
        )
        .to_numpy()
    )

    new_matrix = (
        new.fillna(
            0.0
        )
        .to_numpy(
            np.float32
        )
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

    final_names = (
        base_names
        + new_names
    )

    final_valid = (
        base_valid
        & new_valid
    )

    np.save(
        OUT
        / "technical_features_v651.npy",
        final,
    )

    np.save(
        OUT
        / "technical_valid_v651.npy",
        final_valid.astype(
            np.uint8
        ),
    )

    np.save(
        OUT
        / "timestamps_ns.npy",
        timestamps,
    )

    with open(
        OUT
        / "feature_names.json",
        "w",
    ) as f:
        json.dump(
            final_names,
            f,
            indent=2,
        )

    with open(
        OUT
        / "new_feature_names.json",
        "w",
    ) as f:
        json.dump(
            new_names,
            f,
            indent=2,
        )

    metadata = {
        "rows":
            int(
                final.shape[0]
            ),

        "base_features":
            int(
                len(
                    base_names
                )
            ),

        "new_features":
            int(
                len(
                    new_names
                )
            ),

        "total_features":
            int(
                final.shape[1]
            ),

        "valid_rows":
            int(
                final_valid.sum()
            ),

        "valid_fraction":
            float(
                final_valid.mean()
            ),

        "touch_lookback":
            TOUCH_LOOKBACK,

        "retest_window":
            RETEST_WINDOW,

        "timeframes": [
            "M5",
            "M15",
            "H1",
            "H4",
        ],
    }

    with open(
        OUT
        / "metadata.json",
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(
        "=" * 120
    )

    print(
        "ROWS:",
        f"{final.shape[0]:,}",
    )

    print(
        "BASE V6.5A:",
        len(
            base_names
        ),
    )

    print(
        "NEW V6.5B:",
        len(
            new_names
        ),
    )

    print(
        "TOTAL FEATURES:",
        final.shape[1],
    )

    print(
        "VALID:",
        f"{final_valid.sum():,}",
        f"({final_valid.mean():.2%})",
    )

    print()
    print(
        "FEATURE FAMILIES"
    )

    print(
        "-" * 120
    )

    families = {
        "support_resistance":
            (
                "support_",
                "resistance_",
            ),

        "breakout":
            (
                "breakout_",
            ),

        "retest":
            (
                "retest_",
            ),

        "acceptance_failure":
            (
                "acceptance",
                "failure",
            ),

        "session":
            (
                "session_v651",
            ),
    }

    for family, keys in families.items():
        count = sum(
            any(
                key in name
                for key in keys
            )
            for name
            in new_names
        )

        print(
            f"{family:<24}"
            f"{count:>5}"
        )

    print()
    print(
        "Saved:",
        OUT,
    )


if __name__ == "__main__":
    main()
