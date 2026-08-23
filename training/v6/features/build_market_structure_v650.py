from pathlib import Path
import json
import numpy as np
import pandas as pd

M5_FILE = Path(
    'training/v2/data_lake/xau/'
    'xauusd_m5_bid_ask_2016_2026-06.parquet'
)

BASE_DIR = Path(
    'training/v6/data_lake/'
    'technical_setup_v620'
)

OUT = Path(
    'training/v6/data_lake/'
    'technical_state_v650'
)

STEP = pd.Timedelta(
    minutes=5
)


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
            'timestamp'
        )
        .resample(
            rule,
            label='left',
            closed='left',
        )
        .agg(
            {
                'mid_open':
                    'first',

                'mid_high':
                    'max',

                'mid_low':
                    'min',

                'mid_close':
                    'last',
            }
        )
        .dropna()
        .reset_index()
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

    age = (
        idx
        - last
    )

    age[
        last < 0
    ] = -1

    return age.astype(
        np.float32
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
                high
                - prev
            ).abs(),

            (
                low
                - prev
            ).abs(),
        ],
        axis=1,
    ).max(
        axis=1
    )


def confirmed_pivots(
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

    roll_high = high.rolling(
        window,
        center=True,
        min_periods=window,
    ).max()

    roll_low = low.rolling(
        window,
        center=True,
        min_periods=window,
    ).min()

    raw_high = high.eq(
        roll_high
    )

    raw_low = low.eq(
        roll_low
    )

    # Pivot i is only exposed after
    # `right` later bars have closed.
    ph_event = raw_high.shift(
        right,
        fill_value=False,
    ).astype(
        bool
    )

    pl_event = raw_low.shift(
        right,
        fill_value=False,
    ).astype(
        bool
    )

    ph_value = high.where(
        raw_high
    ).shift(
        right
    )

    pl_value = low.where(
        raw_low
    ).shift(
        right
    )

    return (
        ph_event,
        pl_event,
        ph_value,
        pl_value,
    )


def last_two_confirmed(
    value,
    event,
):
    event_values = (
        value.where(
            event
        )
    )

    last = (
        event_values.ffill()
    )

    previous_at_event = (
        last.shift(
            1
        )
        .where(
            event
        )
    )

    previous = (
        previous_at_event.ffill()
    )

    return (
        last,
        previous,
    )


def efficiency_ratio(
    close,
    n,
):
    direction = (
        close
        - close.shift(
            n
        )
    ).abs()

    path = (
        close.diff()
        .abs()
        .rolling(
            n,
            min_periods=n,
        )
        .sum()
    )

    return (
        direction
        / path.replace(
            0.0,
            np.nan,
        )
    )


def structure_features(
    frame,
    prefix,
    minutes,
):
    x = frame.copy()

    o = x[
        'mid_open'
    ].astype(
        np.float64
    )

    h = x[
        'mid_high'
    ].astype(
        np.float64
    )

    l = x[
        'mid_low'
    ].astype(
        np.float64
    )

    c = x[
        'mid_close'
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

    atr_slow = tr.rolling(
        56,
        min_periods=56,
    ).mean()

    (
        ph_event,
        pl_event,
        ph_value,
        pl_value,
    ) = confirmed_pivots(
        h,
        l,
        3,
        3,
    )

    (
        last_ph,
        prev_ph,
    ) = last_two_confirmed(
        ph_value,
        ph_event,
    )

    (
        last_pl,
        prev_pl,
    ) = last_two_confirmed(
        pl_value,
        pl_event,
    )

    hh = (
        last_ph
        > prev_ph
    ).astype(
        np.float64
    )

    lh = (
        last_ph
        < prev_ph
    ).astype(
        np.float64
    )

    hl = (
        last_pl
        > prev_pl
    ).astype(
        np.float64
    )

    ll = (
        last_pl
        < prev_pl
    ).astype(
        np.float64
    )

    bull_struct = (
        (
            hh + hl
        )
        / 2.0
    ).where(
        last_ph.notna()
        & last_pl.notna()
    )

    bear_struct = (
        (
            lh + ll
        )
        / 2.0
    ).where(
        last_ph.notna()
        & last_pl.notna()
    )

    struct_score = (
        bull_struct.fillna(
            0.0
        )
        - bear_struct.fillna(
            0.0
        )
    )

    prev_c = c.shift(
        1
    )

    bos_up = (
        (c > last_ph)
        & (
            prev_c
            <= last_ph.shift(
                1
            )
        )
    )

    bos_down = (
        (c < last_pl)
        & (
            prev_c
            >= last_pl.shift(
                1
            )
        )
    )

    bos_up_strength = clip01(
        (
            c
            - last_ph
        )
        / (
            atr
            + 1e-9
        )
    )

    bos_down_strength = clip01(
        (
            last_pl
            - c
        )
        / (
            atr
            + 1e-9
        )
    )

    bos_up_strength = np.where(
        bos_up,
        bos_up_strength,
        0.0,
    )

    bos_down_strength = np.where(
        bos_down,
        bos_down_strength,
        0.0,
    )

    prior_struct = (
        struct_score
        .shift(
            1
        )
        .fillna(
            0.0
        )
    )

    choch_up = (
        bos_up
        & (
            prior_struct
            < 0
        )
    )

    choch_down = (
        bos_down
        & (
            prior_struct
            > 0
        )
    )

    sweep_high = (
        (h > last_ph)
        & (
            c < last_ph
        )
    )

    sweep_low = (
        (l < last_pl)
        & (
            c > last_pl
        )
    )

    sweep_high_strength = np.where(
        sweep_high,
        clip01(
            (
                h
                - last_ph
            )
            / (
                atr
                + 1e-9
            )
        ),
        0.0,
    )

    sweep_low_strength = np.where(
        sweep_low,
        clip01(
            (
                last_pl
                - l
            )
            / (
                atr
                + 1e-9
            )
        ),
        0.0,
    )

    eq_high_dist = (
        (
            last_ph
            - prev_ph
        ).abs()
        / (
            atr
            + 1e-9
        )
    )

    eq_low_dist = (
        (
            last_pl
            - prev_pl
        ).abs()
        / (
            atr
            + 1e-9
        )
    )

    equal_high_strength = clip01(
        1.0
        - eq_high_dist
        / 0.20
    )

    equal_low_strength = clip01(
        1.0
        - eq_low_dist
        / 0.20
    )

    body = (
        c - o
    )

    range_ = (
        h - l
    ).replace(
        0.0,
        np.nan,
    )

    body_abs = body.abs()

    upper_wick = (
        h
        - pd.concat(
            [
                o,
                c,
            ],
            axis=1,
        ).max(
            axis=1
        )
    )

    lower_wick = (
        pd.concat(
            [
                o,
                c,
            ],
            axis=1,
        ).min(
            axis=1
        )
        - l
    )

    vol_ratio = (
        atr
        / atr_slow.replace(
            0.0,
            np.nan,
        )
    )

    atr_change = atr.pct_change()

    vol_of_vol = (
        atr_change
        .rolling(
            24,
            min_periods=12,
        )
        .std()
    )

    er12 = efficiency_ratio(
        c,
        12,
    )

    er24 = efficiency_ratio(
        c,
        24,
    )

    delta = c.diff()

    up_path = (
        delta.clip(
            lower=0.0
        )
        .rolling(
            12,
            min_periods=12,
        )
        .sum()
    )

    down_path = (
        (
            -delta.clip(
                upper=0.0
            )
        )
        .rolling(
            12,
            min_periods=12,
        )
        .sum()
    )

    impulse_balance = (
        up_path
        - down_path
    ) / (
        up_path
        + down_path
        + 1e-9
    )

    up_candle = (
        c > o
    ).astype(
        np.int8
    )

    down_candle = (
        c < o
    ).astype(
        np.int8
    )

    up_run = np.zeros(
        len(c),
        dtype=np.float32,
    )

    down_run = np.zeros(
        len(c),
        dtype=np.float32,
    )

    for i in range(
        1,
        len(c),
    ):
        up_run[i] = (
            up_run[i - 1] + 1
            if up_candle.iat[i]
            else 0
        )

        down_run[i] = (
            down_run[i - 1] + 1
            if down_candle.iat[i]
            else 0
        )

    inside = (
        (h < h.shift(1))
        & (
            l > l.shift(1)
        )
    )

    outside = (
        (h > h.shift(1))
        & (
            l < l.shift(1)
        )
    )

    body_expand = (
        body_abs
        / body_abs.rolling(
            20,
            min_periods=10,
        )
        .median()
        .replace(
            0.0,
            np.nan,
        )
    )

    wick_imbalance = (
        lower_wick
        - upper_wick
    ) / range_

    out = pd.DataFrame(
        index=x.index
    )

    out[
        f'{prefix}_swing_high_bps'
    ] = (
        (
            c - last_ph
        )
        / c
        * 10000.0
    )

    out[
        f'{prefix}_swing_low_bps'
    ] = (
        (
            c - last_pl
        )
        / c
        * 10000.0
    )

    out[
        f'{prefix}_swing_high_age'
    ] = bars_since(
        ph_event
    )

    out[
        f'{prefix}_swing_low_age'
    ] = bars_since(
        pl_event
    )

    out[
        f'{prefix}_hh'
    ] = hh

    out[
        f'{prefix}_hl'
    ] = hl

    out[
        f'{prefix}_lh'
    ] = lh

    out[
        f'{prefix}_ll'
    ] = ll

    out[
        f'{prefix}_structure_score_v65'
    ] = struct_score

    out[
        f'{prefix}_bos_up'
    ] = bos_up.astype(
        np.float64
    )

    out[
        f'{prefix}_bos_down'
    ] = bos_down.astype(
        np.float64
    )

    out[
        f'{prefix}_bos_up_strength'
    ] = bos_up_strength

    out[
        f'{prefix}_bos_down_strength'
    ] = bos_down_strength

    out[
        f'{prefix}_choch_up'
    ] = choch_up.astype(
        np.float64
    )

    out[
        f'{prefix}_choch_down'
    ] = choch_down.astype(
        np.float64
    )

    out[
        f'{prefix}_sweep_swing_high'
    ] = sweep_high_strength

    out[
        f'{prefix}_sweep_swing_low'
    ] = sweep_low_strength

    out[
        f'{prefix}_equal_high_strength'
    ] = equal_high_strength

    out[
        f'{prefix}_equal_low_strength'
    ] = equal_low_strength

    out[
        f'{prefix}_trend_efficiency_12'
    ] = er12

    out[
        f'{prefix}_trend_efficiency_24'
    ] = er24

    out[
        f'{prefix}_impulse_balance_12'
    ] = impulse_balance

    out[
        f'{prefix}_volatility_ratio_v65'
    ] = vol_ratio

    out[
        f'{prefix}_vol_of_vol_24'
    ] = vol_of_vol

    out[
        f'{prefix}_up_candle_run'
    ] = up_run

    out[
        f'{prefix}_down_candle_run'
    ] = down_run

    out[
        f'{prefix}_inside_bar'
    ] = inside.astype(
        np.float64
    )

    out[
        f'{prefix}_outside_bar'
    ] = outside.astype(
        np.float64
    )

    out[
        f'{prefix}_body_expansion'
    ] = body_expand

    out[
        f'{prefix}_wick_imbalance'
    ] = wick_imbalance

    out[
        f'{prefix}_available_at'
    ] = (
        pd.to_datetime(
            x['timestamp'],
            utc=True,
        )
        + pd.Timedelta(
            minutes=minutes
        )
    )

    return out


def session_features(
    df,
):
    bar_ts = pd.to_datetime(
        df['timestamp'],
        utc=True,
    )

    available = (
        bar_ts
        + STEP
    )

    h = df[
        'mid_high'
    ].astype(
        np.float64
    )

    l = df[
        'mid_low'
    ].astype(
        np.float64
    )

    c = df[
        'mid_close'
    ].astype(
        np.float64
    )

    utc_date = (
        bar_ts.dt.floor(
            'D'
        )
    )

    day_high = h.groupby(
        utc_date
    ).cummax()

    day_low = l.groupby(
        utc_date
    ).cummin()

    daily = (
        pd.DataFrame(
            {
                'date':
                    utc_date,

                'high':
                    h,

                'low':
                    l,
            }
        )
        .groupby(
            'date'
        )
        .agg(
            {
                'high':
                    'max',

                'low':
                    'min',
            }
        )
    )

    prev_day_high_map = (
        daily[
            'high'
        ].shift(
            1
        )
    )

    prev_day_low_map = (
        daily[
            'low'
        ].shift(
            1
        )
    )

    prev_day_high = utc_date.map(
        prev_day_high_map
    )

    prev_day_low = utc_date.map(
        prev_day_low_map
    )

    utc_hour = (
        bar_ts.dt.hour
        + bar_ts.dt.minute
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

    asia_date = utc_date

    asia_full = (
        pd.DataFrame(
            {
                'date':
                    asia_date[
                        asia_mask
                    ],

                'high':
                    h[
                        asia_mask
                    ],

                'low':
                    l[
                        asia_mask
                    ],
            }
        )
        .groupby(
            'date'
        )
        .agg(
            {
                'high':
                    'max',

                'low':
                    'min',
            }
        )
    )

    asia_high = asia_date.map(
        asia_full[
            'high'
        ]
    )

    asia_low = asia_date.map(
        asia_full[
            'low'
        ]
    )

    asia_ready = (
        available_hour
        >= 8.0
    )

    london_local = (
        bar_ts.dt.tz_convert(
            'Europe/London'
        )
    )

    london_hour = (
        london_local.dt.hour
        + london_local.dt.minute
        / 60.0
    )

    london_open = 8.0
    london_close = 16.0

    london_session = (
        (
            london_hour
            >= london_open
        )
        & (
            london_hour
            < london_close
        )
    )

    ny_local = (
        bar_ts.dt.tz_convert(
            'America/New_York'
        )
    )

    ny_hour = (
        ny_local.dt.hour
        + ny_local.dt.minute
        / 60.0
    )

    ny_open = 8.0
    ny_close = 17.0

    ny_session = (
        (
            ny_hour
            >= ny_open
        )
        & (
            ny_hour
            < ny_close
        )
    )

    day_range = (
        day_high
        - day_low
    ).replace(
        0.0,
        np.nan,
    )

    asia_range = (
        asia_high
        - asia_low
    ).replace(
        0.0,
        np.nan,
    )

    london_sweep_asia_high = (
        asia_ready
        & (
            h > asia_high
        )
        & (
            c < asia_high
        )
    )

    london_sweep_asia_low = (
        asia_ready
        & (
            l < asia_low
        )
        & (
            c > asia_low
        )
    )

    out = pd.DataFrame(
        index=df.index
    )

    out[
        'session_day_position'
    ] = (
        c - day_low
    ) / day_range

    out[
        'session_prev_day_high_bps'
    ] = (
        (
            prev_day_high
            - c
        )
        / c
        * 10000.0
    )

    out[
        'session_prev_day_low_bps'
    ] = (
        (
            c
            - prev_day_low
        )
        / c
        * 10000.0
    )

    out[
        'session_break_prev_day_high'
    ] = (
        c > prev_day_high
    ).astype(
        np.float64
    )

    out[
        'session_break_prev_day_low'
    ] = (
        c < prev_day_low
    ).astype(
        np.float64
    )

    out[
        'session_asia_ready'
    ] = asia_ready.astype(
        np.float64
    )

    out[
        'session_asia_high_bps'
    ] = np.where(
        asia_ready,
        (
            asia_high - c
        )
        / c
        * 10000.0,
        0.0,
    )

    out[
        'session_asia_low_bps'
    ] = np.where(
        asia_ready,
        (
            c - asia_low
        )
        / c
        * 10000.0,
        0.0,
    )

    out[
        'session_asia_range_position'
    ] = np.where(
        asia_ready,
        (
            c - asia_low
        )
        / asia_range,
        0.0,
    )

    out[
        'session_london_sweep_asia_high'
    ] = (
        london_sweep_asia_high.astype(
            np.float64
        )
    )

    out[
        'session_london_sweep_asia_low'
    ] = (
        london_sweep_asia_low.astype(
            np.float64
        )
    )

    out[
        'session_is_london'
    ] = london_session.astype(
        np.float64
    )

    out[
        'session_is_new_york'
    ] = ny_session.astype(
        np.float64
    )

    out[
        'session_is_overlap'
    ] = (
        london_session
        & ny_session
    ).astype(
        np.float64
    )

    out[
        'session_minutes_from_london_open'
    ] = np.clip(
        (
            london_hour
            - london_open
        )
        * 60.0,
        -480.0,
        480.0,
    )

    out[
        'session_minutes_from_ny_open'
    ] = np.clip(
        (
            ny_hour
            - ny_open
        )
        * 60.0,
        -480.0,
        480.0,
    )

    return out


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        'TEN V6.5A '
        'INSTITUTIONAL MARKET STRUCTURE '
        '& LIQUIDITY ENGINE'
    )

    print(
        '=' * 120
    )

    df = pd.read_parquet(
        M5_FILE
    ).reset_index(
        drop=True
    )

    df[
        'timestamp'
    ] = pd.to_datetime(
        df['timestamp'],
        utc=True,
    )

    base = np.load(
        BASE_DIR
        / 'technical_setup_features.npy',
        mmap_mode='r',
    )

    base_valid = np.load(
        BASE_DIR
        / 'technical_setup_valid.npy',
        mmap_mode='r',
    ).astype(
        bool
    )

    base_ts = np.load(
        BASE_DIR
        / 'timestamps_ns.npy',
        mmap_mode='r',
    ).astype(
        np.int64
    )

    with open(
        BASE_DIR
        / 'feature_names.json'
    ) as f:
        base_names = json.load(
            f
        )

    if len(df) != len(base):
        raise RuntimeError(
            'M5/base row mismatch'
        )

    anchor = pd.DataFrame(
        {
            'row':
                np.arange(
                    len(df),
                    dtype=np.int64,
                ),

            'available_at':
                df[
                    'timestamp'
                ]
                + STEP,
        }
    )

    frames = []

    print(
        'Building M5 structure ...'
    )

    m5 = structure_features(
        df[
            [
                'timestamp',
                'mid_open',
                'mid_high',
                'mid_low',
                'mid_close',
            ]
        ],
        'm5v65',
        5,
    )

    frames.append(
        m5.drop(
            columns=[
                'm5v65_available_at'
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
            '15min',
            'm15v65',
            15,
        ),
        (
            '1h',
            'h1v65',
            60,
        ),
        (
            '4h',
            'h4v65',
            240,
        ),
    ):
        print(
            'Building',
            prefix.upper(),
            'structure ...'
        )

        tf = aggregate_tf(
            df[
                [
                    'timestamp',
                    'mid_open',
                    'mid_high',
                    'mid_low',
                    'mid_close',
                ]
            ],
            rule,
        )

        feat = structure_features(
            tf,
            prefix,
            minutes,
        ).rename(
            columns={
                f'{prefix}_available_at':
                    'available_at'
            }
        )

        merged = pd.merge_asof(
            anchor.sort_values(
                'available_at'
            ),
            feat.sort_values(
                'available_at'
            ),
            on='available_at',
            direction='backward',
            allow_exact_matches=True,
        ).sort_values(
            'row'
        )

        cols = [
            c
            for c in merged.columns
            if c not in (
                'row',
                'available_at',
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
        'Building daily/session liquidity ...'
    )

    frames.append(
        session_features(
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

    # Multi-timeframe structure
    # agreement / conflict.
    for direction in (
        'up',
        'down',
    ):
        vals = []

        for prefix in (
            'm5v65',
            'm15v65',
            'h1v65',
            'h4v65',
        ):
            col = (
                f'{prefix}_'
                'structure_score_v65'
            )

            s = new[
                col
            ].to_numpy(
                np.float64
            )

            vals.append(
                np.clip(
                    s
                    if direction == 'up'
                    else -s,
                    0.0,
                    1.0,
                )
            )

        mat = np.column_stack(
            vals
        )

        new[
            f'mtf_structure_'
            f'{direction}_agreement'
        ] = mat.mean(
            axis=1
        )

        new[
            f'mtf_structure_'
            f'{direction}_minimum'
        ] = mat.min(
            axis=1
        )

    signs = np.column_stack(
        [
            np.sign(
                new[
                    f'{p}_'
                    'structure_score_v65'
                ].to_numpy(
                    np.float64
                )
            )
            for p in (
                'm5v65',
                'm15v65',
                'h1v65',
                'h4v65',
            )
        ]
    )

    new[
        'mtf_structure_conflict'
    ] = (
        1.0
        - np.abs(
            signs.mean(
                axis=1
            )
        )
    )

    new_names = (
        new.columns.tolist()
    )

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

    final_valid = (
        base_valid
        & new_valid
    )

    final_names = (
        base_names
        + new_names
    )

    np.save(
        OUT
        / 'technical_features_v650.npy',
        final,
    )

    np.save(
        OUT
        / 'technical_valid_v650.npy',
        final_valid.astype(
            np.uint8
        ),
    )

    np.save(
        OUT
        / 'timestamps_ns.npy',
        base_ts,
    )

    with open(
        OUT
        / 'feature_names.json',
        'w',
    ) as f:
        json.dump(
            final_names,
            f,
            indent=2,
        )

    with open(
        OUT
        / 'new_feature_names.json',
        'w',
    ) as f:
        json.dump(
            new_names,
            f,
            indent=2,
        )

    metadata = {
        'rows':
            int(
                final.shape[0]
            ),

        'features':
            int(
                final.shape[1]
            ),

        'base_v620_features':
            int(
                len(
                    base_names
                )
            ),

        'new_v650_features':
            int(
                len(
                    new_names
                )
            ),

        'valid_rows':
            int(
                final_valid.sum()
            ),

        'valid_fraction':
            float(
                final_valid.mean()
            ),

        'timeframes': [
            'M5',
            'M15',
            'H1',
            'H4',
        ],

        'pivot_rule': (
            '3-left/3-right confirmed; '
            'exposed only after '
            'right-side confirmation'
        ),

        'sessions': (
            'London/NY use '
            'timezone-aware local clocks; '
            'Asia uses 00:00-08:00 UTC'
        ),
    }

    with open(
        OUT
        / 'metadata.json',
        'w',
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(
        '=' * 120
    )

    print(
        'ROWS:',
        f'{final.shape[0]:,}',
    )

    print(
        'BASE FEATURES:',
        len(
            base_names
        ),
    )

    print(
        'NEW V6.5A FEATURES:',
        len(
            new_names
        ),
    )

    print(
        'TOTAL FEATURES:',
        final.shape[1],
    )

    print(
        'VALID:',
        f'{final_valid.sum():,}',
        f'({final_valid.mean():.2%})',
    )

    print()
    print(
        'NEW FEATURE FAMILIES'
    )

    print(
        '-' * 120
    )

    for key in (
        'swing',
        'bos',
        'choch',
        'sweep',
        'equal_',
        'efficiency',
        'volatility',
        'session_',
        'mtf_',
    ):
        n = sum(
            key in name
            for name
            in new_names
        )

        print(
            f'{key:<18} '
            f'{n:>4}'
        )

    print()
    print(
        'Saved:',
        OUT,
    )


if __name__ == '__main__':
    main()
