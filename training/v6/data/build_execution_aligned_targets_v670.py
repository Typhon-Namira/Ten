from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


INPUT = Path(
    "training/v2/data_lake/xau/"
    "xauusd_m5_bid_ask_2016_2026-06.parquet"
)

OUT = Path(
    "training/v6/data_lake/"
    "execution_aligned_targets_v670"
)

OUT_FILE = (
    OUT
    / "execution_aligned_targets_v670.parquet"
)

STEP_NS = 300_000_000_000

HORIZONS = {
    30: 6,
    60: 12,
    120: 24,
}

BARRIERS = (
    (30, 15),
    (40, 20),
    (60, 30),
)

EXTRA_COSTS = (
    0.5,
    1.0,
)


def horizon_valid(
    timestamps_ns,
    steps,
):
    n = len(
        timestamps_ns
    )

    good_edge = (
        np.diff(
            timestamps_ns
        )
        == STEP_NS
    )

    bad = (
        ~good_edge
    ).astype(
        np.int32
    )

    prefix = np.zeros(
        n,
        dtype=np.int64,
    )

    prefix[
        1:
    ] = np.cumsum(
        bad,
        dtype=np.int64,
    )

    out = np.zeros(
        n,
        dtype=bool,
    )

    end = (
        n - steps
    )

    if end <= 0:
        return out

    anchors = np.arange(
        end,
        dtype=np.int64,
    )

    bad_count = (
        prefix[
            anchors + steps
        ]
        - prefix[
            anchors
        ]
    )

    out[
        :end
    ] = (
        bad_count == 0
    )

    return out


def make_float(
    n,
    fill=np.nan,
):
    return np.full(
        n,
        fill,
        dtype=np.float32,
    )


def make_int8(
    n,
    fill=-3,
):
    return np.full(
        n,
        fill,
        dtype=np.int8,
    )


def make_int16(
    n,
    fill=0,
):
    return np.full(
        n,
        fill,
        dtype=np.int16,
    )


def make_uint8(
    n,
    fill=0,
):
    return np.full(
        n,
        fill,
        dtype=np.uint8,
    )


def bps_long(
    exit_price,
    entry_price,
):
    return (
        (
            exit_price
            - entry_price
        )
        / entry_price
        * 10000.0
    )


def bps_short(
    exit_price,
    entry_price,
):
    return (
        (
            entry_price
            - exit_price
        )
        / entry_price
        * 10000.0
    )


def snapshot_task(
    side,
    state,
    valid,
    horizon,
    steps,
    tp,
    sl,
    terminal_price,
    entry_price,
):
    n = len(
        entry_price
    )

    key = (
        f"h{horizon}_"
        f"tp{tp}_"
        f"sl{sl}"
    )

    status = state[
        "status"
    ]

    gross_resolved = state[
        "gross"
    ]

    first_exit = state[
        "first_exit"
    ]

    ambiguous = state[
        "ambiguous"
    ]

    outcome = make_int8(
        n
    )

    gross = make_float(
        n
    )

    exit_bar = make_int16(
        n
    )

    amb = make_uint8(
        n
    )

    valid_idx = np.flatnonzero(
        valid
    )

    st = status[
        valid_idx
    ]

    tp_mask = (
        st == 1
    )

    sl_mask = (
        st == 2
    )

    timeout_mask = (
        st == 0
    )

    v_tp = valid_idx[
        tp_mask
    ]

    v_sl = valid_idx[
        sl_mask
    ]

    v_to = valid_idx[
        timeout_mask
    ]

    # 1 = TP
    # 0 = SL
    # -1 = TIMEOUT
    # -3 = invalid horizon

    outcome[
        v_tp
    ] = 1

    outcome[
        v_sl
    ] = 0

    outcome[
        v_to
    ] = -1

    gross[
        v_tp
    ] = gross_resolved[
        v_tp
    ]

    gross[
        v_sl
    ] = gross_resolved[
        v_sl
    ]

    if len(
        v_to
    ):
        if side == "long":
            gross[
                v_to
            ] = bps_long(
                terminal_price[
                    v_to
                ],
                entry_price[
                    v_to
                ],
            ).astype(
                np.float32
            )

        else:
            gross[
                v_to
            ] = bps_short(
                terminal_price[
                    v_to
                ],
                entry_price[
                    v_to
                ],
            ).astype(
                np.float32
            )

    exit_bar[
        v_tp
    ] = first_exit[
        v_tp
    ]

    exit_bar[
        v_sl
    ] = first_exit[
        v_sl
    ]

    exit_bar[
        v_to
    ] = steps

    amb[
        valid_idx
    ] = ambiguous[
        valid_idx
    ]

    columns = {
        f"{side}_outcome_{key}":
            outcome,

        f"{side}_gross_bps_{key}":
            gross,

        f"{side}_exit_bar_{key}":
            exit_bar,

        f"{side}_ambiguous_{key}":
            amb,
    }

    for cost in EXTRA_COSTS:
        suffix = (
            str(cost)
            .replace(
                ".",
                ""
            )
        )

        win = make_uint8(
            n
        )

        net_positive = (
            gross[
                valid_idx
            ]
            - cost
            > 0.0
        )

        win[
            valid_idx
        ] = net_positive.astype(
            np.uint8
        )

        columns[
            f"{side}_win_c{suffix}_{key}"
        ] = win

    return columns


def main():
    started = time.time()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.7.0 "
        "EXECUTION-ALIGNED TARGET SURFACE"
    )

    print(
        "=" * 120
    )

    df = pd.read_parquet(
        INPUT,
        columns=[
            "timestamp",
            "bid_open",
            "bid_high",
            "bid_low",
            "bid_close",
            "ask_open",
            "ask_high",
            "ask_low",
            "ask_close",
        ],
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

    n = len(
        df
    )

    timestamps_ns = (
        df[
            "timestamp"
        ]
        .astype(
            "int64"
        )
        .to_numpy(
            np.int64
        )
    )

    bid_open = df[
        "bid_open"
    ].to_numpy(
        np.float64
    )

    bid_high = df[
        "bid_high"
    ].to_numpy(
        np.float64
    )

    bid_low = df[
        "bid_low"
    ].to_numpy(
        np.float64
    )

    bid_close = df[
        "bid_close"
    ].to_numpy(
        np.float64
    )

    ask_open = df[
        "ask_open"
    ].to_numpy(
        np.float64
    )

    ask_high = df[
        "ask_high"
    ].to_numpy(
        np.float64
    )

    ask_low = df[
        "ask_low"
    ].to_numpy(
        np.float64
    )

    ask_close = df[
        "ask_close"
    ].to_numpy(
        np.float64
    )

    print(
        "Rows:",
        f"{n:,}",
    )

    print(
        "Signal timing: M5 close"
    )

    print(
        "Entry timing: NEXT M5 OPEN"
    )

    print(
        "LONG entry: ASK open"
    )

    print(
        "SHORT entry: BID open"
    )

    print(
        "Same-bar TP+SL: conservative SL"
    )

    # Entry belongs to next M5 bar.
    entry_long = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    entry_short = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    entry_long[
        :-1
    ] = ask_open[
        1:
    ]

    entry_short[
        :-1
    ] = bid_open[
        1:
    ]

    valid_by_horizon = {}

    for horizon, steps in (
        HORIZONS.items()
    ):
        valid_by_horizon[
            horizon
        ] = horizon_valid(
            timestamps_ns,
            steps,
        )

        print(
            f"H{horizon}: "
            f"{valid_by_horizon[horizon].sum():,} "
            f"valid"
        )

    # MFE / MAE cumulative states.
    #
    # Zero is included as starting
    # excursion, so MFE/MAE can never
    # become negative.

    long_mfe = np.zeros(
        n,
        dtype=np.float64,
    )

    long_mae = np.zeros(
        n,
        dtype=np.float64,
    )

    short_mfe = np.zeros(
        n,
        dtype=np.float64,
    )

    short_mae = np.zeros(
        n,
        dtype=np.float64,
    )

    # One independent state per barrier.
    states = {}

    for tp, sl in BARRIERS:
        states[
            (
                "long",
                tp,
                sl,
            )
        ] = {
            "status":
                np.zeros(
                    n,
                    dtype=np.int8,
                ),

            "gross":
                make_float(
                    n
                ),

            "first_exit":
                make_int16(
                    n
                ),

            "ambiguous":
                make_uint8(
                    n
                ),
        }

        states[
            (
                "short",
                tp,
                sl,
            )
        ] = {
            "status":
                np.zeros(
                    n,
                    dtype=np.int8,
                ),

            "gross":
                make_float(
                    n
                ),

            "first_exit":
                make_int16(
                    n
                ),

            "ambiguous":
                make_uint8(
                    n
                ),
        }

    output = {
        "source_row":
            np.arange(
                n,
                dtype=np.int64,
            ),

        "timestamp":
            df[
                "timestamp"
            ],

        "year":
            df[
                "timestamp"
            ].dt.year.to_numpy(
                np.int16
            ),
    }

    for horizon in HORIZONS:
        output[
            f"horizon_valid_h{horizon}"
        ] = valid_by_horizon[
            horizon
        ].astype(
            np.uint8
        )

    max_steps = max(
        HORIZONS.values()
    )

    for step in range(
        1,
        max_steps + 1,
    ):
        valid_n = (
            n - step
        )

        if valid_n <= 0:
            break

        anchors = slice(
            0,
            valid_n,
        )

        bar = slice(
            step,
            step + valid_n,
        )

        e_long = entry_long[
            anchors
        ]

        e_short = entry_short[
            anchors
        ]

        # --------------------------------
        # Executable excursions
        # --------------------------------

        long_fav = bps_long(
            bid_high[
                bar
            ],
            e_long,
        )

        long_adv = -bps_long(
            bid_low[
                bar
            ],
            e_long,
        )

        short_fav = bps_short(
            ask_low[
                bar
            ],
            e_short,
        )

        short_adv = -bps_short(
            ask_high[
                bar
            ],
            e_short,
        )

        long_mfe[
            anchors
        ] = np.maximum(
            long_mfe[
                anchors
            ],
            np.maximum(
                long_fav,
                0.0,
            ),
        )

        long_mae[
            anchors
        ] = np.maximum(
            long_mae[
                anchors
            ],
            np.maximum(
                long_adv,
                0.0,
            ),
        )

        short_mfe[
            anchors
        ] = np.maximum(
            short_mfe[
                anchors
            ],
            np.maximum(
                short_fav,
                0.0,
            ),
        )

        short_mae[
            anchors
        ] = np.maximum(
            short_mae[
                anchors
            ],
            np.maximum(
                short_adv,
                0.0,
            ),
        )

        # --------------------------------
        # Barrier execution
        # --------------------------------

        for tp, sl in BARRIERS:

            # ============================
            # LONG
            # ============================

            state = states[
                (
                    "long",
                    tp,
                    sl,
                )
            ]

            status = state[
                "status"
            ][
                anchors
            ]

            unresolved = (
                status == 0
            )

            tp_price = (
                e_long
                * (
                    1.0
                    + tp
                    / 10000.0
                )
            )

            sl_price = (
                e_long
                * (
                    1.0
                    - sl
                    / 10000.0
                )
            )

            open_px = bid_open[
                bar
            ]

            high_px = bid_high[
                bar
            ]

            low_px = bid_low[
                bar
            ]

            # Gap-through SL has priority.
            hit_sl_gap = (
                unresolved
                & (
                    open_px
                    <= sl_price
                )
            )

            # Favorable gap fills at target,
            # never above target.
            hit_tp_gap = (
                unresolved
                & ~hit_sl_gap
                & (
                    open_px
                    >= tp_price
                )
            )

            remaining = (
                unresolved
                & ~hit_sl_gap
                & ~hit_tp_gap
            )

            hit_tp = (
                remaining
                & (
                    high_px
                    >= tp_price
                )
            )

            hit_sl = (
                remaining
                & (
                    low_px
                    <= sl_price
                )
            )

            both = (
                hit_tp
                & hit_sl
            )

            sl_normal = (
                hit_sl
                & ~both
            )

            tp_normal = (
                hit_tp
                & ~both
            )

            idx = np.arange(
                valid_n
            )

            global_idx = idx

            for mask in (
                hit_sl_gap,
                hit_tp_gap,
                both,
                sl_normal,
                tp_normal,
            ):
                pass

            g = global_idx[
                hit_sl_gap
            ]

            if len(g):
                state[
                    "status"
                ][g] = 2

                state[
                    "gross"
                ][g] = bps_long(
                    open_px[
                        hit_sl_gap
                    ],
                    e_long[
                        hit_sl_gap
                    ],
                ).astype(
                    np.float32
                )

                state[
                    "first_exit"
                ][g] = step

            g = global_idx[
                hit_tp_gap
            ]

            if len(g):
                state[
                    "status"
                ][g] = 1

                state[
                    "gross"
                ][g] = float(
                    tp
                )

                state[
                    "first_exit"
                ][g] = step

            g = global_idx[
                both
            ]

            if len(g):
                state[
                    "status"
                ][g] = 2

                state[
                    "gross"
                ][g] = -float(
                    sl
                )

                state[
                    "first_exit"
                ][g] = step

                state[
                    "ambiguous"
                ][g] = 1

            g = global_idx[
                sl_normal
            ]

            if len(g):
                state[
                    "status"
                ][g] = 2

                state[
                    "gross"
                ][g] = -float(
                    sl
                )

                state[
                    "first_exit"
                ][g] = step

            g = global_idx[
                tp_normal
            ]

            if len(g):
                state[
                    "status"
                ][g] = 1

                state[
                    "gross"
                ][g] = float(
                    tp
                )

                state[
                    "first_exit"
                ][g] = step

            # ============================
            # SHORT
            # ============================

            state = states[
                (
                    "short",
                    tp,
                    sl,
                )
            ]

            status = state[
                "status"
            ][
                anchors
            ]

            unresolved = (
                status == 0
            )

            tp_price = (
                e_short
                * (
                    1.0
                    - tp
                    / 10000.0
                )
            )

            sl_price = (
                e_short
                * (
                    1.0
                    + sl
                    / 10000.0
                )
            )

            open_px = ask_open[
                bar
            ]

            low_px = ask_low[
                bar
            ]

            high_px = ask_high[
                bar
            ]

            hit_sl_gap = (
                unresolved
                & (
                    open_px
                    >= sl_price
                )
            )

            hit_tp_gap = (
                unresolved
                & ~hit_sl_gap
                & (
                    open_px
                    <= tp_price
                )
            )

            remaining = (
                unresolved
                & ~hit_sl_gap
                & ~hit_tp_gap
            )

            hit_tp = (
                remaining
                & (
                    low_px
                    <= tp_price
                )
            )

            hit_sl = (
                remaining
                & (
                    high_px
                    >= sl_price
                )
            )

            both = (
                hit_tp
                & hit_sl
            )

            sl_normal = (
                hit_sl
                & ~both
            )

            tp_normal = (
                hit_tp
                & ~both
            )

            g = global_idx[
                hit_sl_gap
            ]

            if len(g):
                state[
                    "status"
                ][g] = 2

                state[
                    "gross"
                ][g] = bps_short(
                    open_px[
                        hit_sl_gap
                    ],
                    e_short[
                        hit_sl_gap
                    ],
                ).astype(
                    np.float32
                )

                state[
                    "first_exit"
                ][g] = step

            g = global_idx[
                hit_tp_gap
            ]

            if len(g):
                state[
                    "status"
                ][g] = 1

                state[
                    "gross"
                ][g] = float(
                    tp
                )

                state[
                    "first_exit"
                ][g] = step

            g = global_idx[
                both
            ]

            if len(g):
                state[
                    "status"
                ][g] = 2

                state[
                    "gross"
                ][g] = -float(
                    sl
                )

                state[
                    "first_exit"
                ][g] = step

                state[
                    "ambiguous"
                ][g] = 1

            g = global_idx[
                sl_normal
            ]

            if len(g):
                state[
                    "status"
                ][g] = 2

                state[
                    "gross"
                ][g] = -float(
                    sl
                )

                state[
                    "first_exit"
                ][g] = step

            g = global_idx[
                tp_normal
            ]

            if len(g):
                state[
                    "status"
                ][g] = 1

                state[
                    "gross"
                ][g] = float(
                    tp
                )

                state[
                    "first_exit"
                ][g] = step

        # --------------------------------
        # Horizon snapshot
        # --------------------------------

        horizon = next(
            (
                h
                for h, s
                in HORIZONS.items()
                if s == step
            ),
            None,
        )

        if horizon is None:
            continue

        valid = valid_by_horizon[
            horizon
        ]

        output[
            f"long_mfe_bps_h{horizon}"
        ] = np.where(
            valid,
            long_mfe,
            np.nan,
        ).astype(
            np.float32
        )

        output[
            f"long_mae_bps_h{horizon}"
        ] = np.where(
            valid,
            long_mae,
            np.nan,
        ).astype(
            np.float32
        )

        output[
            f"short_mfe_bps_h{horizon}"
        ] = np.where(
            valid,
            short_mfe,
            np.nan,
        ).astype(
            np.float32
        )

        output[
            f"short_mae_bps_h{horizon}"
        ] = np.where(
            valid,
            short_mae,
            np.nan,
        ).astype(
            np.float32
        )

        terminal_row = np.minimum(
            np.arange(
                n,
                dtype=np.int64,
            )
            + step,
            n - 1,
        )

        terminal_long = bid_close[
            terminal_row
        ]

        terminal_short = ask_close[
            terminal_row
        ]

        for tp, sl in BARRIERS:

            output.update(
                snapshot_task(
                    "long",
                    states[
                        (
                            "long",
                            tp,
                            sl,
                        )
                    ],
                    valid,
                    horizon,
                    step,
                    tp,
                    sl,
                    terminal_long,
                    entry_long,
                )
            )

            output.update(
                snapshot_task(
                    "short",
                    states[
                        (
                            "short",
                            tp,
                            sl,
                        )
                    ],
                    valid,
                    horizon,
                    step,
                    tp,
                    sl,
                    terminal_short,
                    entry_short,
                )
            )

        print(
            f"Snapshot H{horizon} complete"
        )

    result = pd.DataFrame(
        output
    )

    result.to_parquet(
        OUT_FILE,
        index=False,
    )

    metadata = {
        "version":
            "v6.7.0",

        "rows":
            int(
                len(
                    result
                )
            ),

        "signal_time":
            "M5 close",

        "entry_time":
            "next M5 open",

        "long_entry":
            "ask_open[next_bar]",

        "short_entry":
            "bid_open[next_bar]",

        "long_exit":
            "bid",

        "short_exit":
            "ask",

        "same_bar_tp_sl":
            "SL conservative",

        "stop_gap":
            "actual executable open",

        "tp_gap":
            "fill at target, no improvement",

        "horizons":
            HORIZONS,

        "barriers":
            [
                list(x)
                for x in BARRIERS
            ],

        "extra_cost_labels_bps":
            list(
                EXTRA_COSTS
            ),

        "spread":
            "embedded via bid/ask",
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
        "TEN V6.7.0 "
        "EXECUTION TARGET SUMMARY"
    )

    print(
        "=" * 120
    )

    for horizon in HORIZONS:

        valid = (
            result[
                f"horizon_valid_h{horizon}"
            ].to_numpy(
                bool
            )
        )

        print()
        print(
            f"HORIZON {horizon}m"
        )

        print(
            "-" * 100
        )

        print(
            "Valid:",
            f"{valid.sum():,}",
            f"({valid.mean():.2%})",
        )

        print(
            "MFE/MAE | "
            f"LONG "
            f"{np.nanmedian(result[f'long_mfe_bps_h{horizon}']):.2f}/"
            f"{np.nanmedian(result[f'long_mae_bps_h{horizon}']):.2f} "
            f"| SHORT "
            f"{np.nanmedian(result[f'short_mfe_bps_h{horizon}']):.2f}/"
            f"{np.nanmedian(result[f'short_mae_bps_h{horizon}']):.2f}"
        )

        for side in (
            "long",
            "short",
        ):
            for tp, sl in BARRIERS:

                key = (
                    f"h{horizon}_"
                    f"tp{tp}_"
                    f"sl{sl}"
                )

                outcome = result[
                    f"{side}_outcome_{key}"
                ].to_numpy(
                    np.int8
                )

                z = outcome[
                    valid
                ]

                tp_rate = (
                    z == 1
                ).mean()

                sl_rate = (
                    z == 0
                ).mean()

                timeout_rate = (
                    z == -1
                ).mean()

                amb_rate = result[
                    f"{side}_ambiguous_{key}"
                ].to_numpy(
                    np.uint8
                )[
                    valid
                ].mean()

                gross = result[
                    f"{side}_gross_bps_{key}"
                ].to_numpy(
                    np.float32
                )[
                    valid
                ]

                win05 = result[
                    f"{side}_win_c05_{key}"
                ].to_numpy(
                    np.uint8
                )[
                    valid
                ].mean()

                win10 = result[
                    f"{side}_win_c10_{key}"
                ].to_numpy(
                    np.uint8
                )[
                    valid
                ].mean()

                resolved = (
                    tp_rate
                    + sl_rate
                )

                tp_res = (
                    tp_rate
                    / resolved
                    if resolved > 0
                    else np.nan
                )

                print(
                    f"{side.upper():<5} "
                    f"TP{tp}/SL{sl} "
                    f"TP={tp_rate:>6.2%} "
                    f"SL={sl_rate:>6.2%} "
                    f"TO={timeout_rate:>6.2%} "
                    f"AMB={amb_rate:>6.3%} "
                    f"TP|RES={tp_res:>6.2%} "
                    f"WIN@.5={win05:>6.2%} "
                    f"WIN@1={win10:>6.2%} "
                    f"MEAN_GROSS={np.nanmean(gross):>+7.3f}"
                )

    print()
    print(
        "Columns:",
        len(
            result.columns
        ),
    )

    print(
        "Saved:",
        OUT_FILE,
    )

    print(
        "Elapsed:",
        f"{time.time() - started:.2f}s",
    )


if __name__ == "__main__":
    main()
