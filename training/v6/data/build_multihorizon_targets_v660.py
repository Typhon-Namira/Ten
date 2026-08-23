from pathlib import Path
import json

import numpy as np
import pandas as pd


M5_FILE = Path(
    "training/v2/data_lake/xau/"
    "xauusd_m5_bid_ask_2016_2026-06.parquet"
)

OUT = Path(
    "training/v6/data_lake/"
    "multihorizon_targets_v660"
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


def horizon_valid(
    timestamps_ns,
    steps,
):
    n = len(
        timestamps_ns
    )

    bad = np.zeros(
        n,
        dtype=np.int32,
    )

    bad[1:] = (
        np.diff(
            timestamps_ns
        )
        != STEP_NS
    ).astype(
        np.int32
    )

    pref = np.concatenate(
        [
            np.array(
                [0],
                dtype=np.int64,
            ),
            np.cumsum(
                bad,
                dtype=np.int64,
            ),
        ]
    )

    source = np.arange(
        n,
        dtype=np.int64,
    )

    end = (
        source
        + steps
    )

    in_range = (
        end < n
    )

    safe_end = np.minimum(
        end,
        n - 1,
    )

    gap_count = (
        pref[
            safe_end + 1
        ]
        - pref[
            source + 1
        ]
    )

    return (
        in_range
        & (
            gap_count == 0
        )
    )


def race_from_hits(
    first_tp,
    first_sl,
    valid,
):
    race = np.full(
        len(first_tp),
        -3,
        dtype=np.int8,
    )

    active = valid.copy()

    no_tp = (
        first_tp == 0
    )

    no_sl = (
        first_sl == 0
    )

    timeout = (
        active
        & no_tp
        & no_sl
    )

    race[
        timeout
    ] = -1

    tp_only = (
        active
        & (
            first_tp > 0
        )
        & no_sl
    )

    race[
        tp_only
    ] = 1

    sl_only = (
        active
        & no_tp
        & (
            first_sl > 0
        )
    )

    race[
        sl_only
    ] = 0

    both = (
        active
        & (
            first_tp > 0
        )
        & (
            first_sl > 0
        )
    )

    tp_first = (
        both
        & (
            first_tp
            < first_sl
        )
    )

    sl_first = (
        both
        & (
            first_sl
            < first_tp
        )
    )

    ambiguous = (
        both
        & (
            first_tp
            == first_sl
        )
    )

    race[
        tp_first
    ] = 1

    race[
        sl_first
    ] = 0

    race[
        ambiguous
    ] = -2

    return race


def masked_float(
    values,
    valid,
):
    out = np.full(
        len(values),
        np.nan,
        dtype=np.float32,
    )

    out[
        valid
    ] = np.asarray(
        values[
            valid
        ],
        dtype=np.float32,
    )

    return out


def masked_int(
    values,
    valid,
):
    out = np.full(
        len(values),
        -1,
        dtype=np.int16,
    )

    out[
        valid
    ] = values[
        valid
    ]

    return out


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.6 "
        "MULTI-HORIZON / "
        "MULTI-BARRIER TARGET SURFACE"
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

    n = len(
        df
    )

    ts_ns = (
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

    # Executable entries.
    #
    # LONG enters at ASK.
    # SHORT enters at BID.
    long_entry = (
        ask_close.copy()
    )

    short_entry = (
        bid_close.copy()
    )

    source = np.arange(
        n,
        dtype=np.int64,
    )

    output = {
        "source_row":
            source,

        "timestamp":
            df[
                "timestamp"
            ],

        "year":
            df[
                "timestamp"
            ]
            .dt.year
            .to_numpy(
                np.int16
            ),
    }

    valid_by_horizon = {}

    for minutes, steps in (
        HORIZONS.items()
    ):
        valid = horizon_valid(
            ts_ns,
            steps,
        )

        valid_by_horizon[
            minutes
        ] = valid

        output[
            f"horizon_valid_h{minutes}"
        ] = valid.astype(
            np.uint8
        )

    # First-hit state is reusable
    # across all horizons.
    states = {}

    for tp_bps, sl_bps in BARRIERS:
        states[
            (
                tp_bps,
                sl_bps,
            )
        ] = {
            "long_tp":
                np.zeros(
                    n,
                    dtype=np.int16,
                ),

            "long_sl":
                np.zeros(
                    n,
                    dtype=np.int16,
                ),

            "short_tp":
                np.zeros(
                    n,
                    dtype=np.int16,
                ),

            "short_sl":
                np.zeros(
                    n,
                    dtype=np.int16,
                ),
        }

    # Cumulative executable
    # excursion state.
    max_bid_high = np.full(
        n,
        -np.inf,
        dtype=np.float64,
    )

    min_bid_low = np.full(
        n,
        np.inf,
        dtype=np.float64,
    )

    max_ask_high = np.full(
        n,
        -np.inf,
        dtype=np.float64,
    )

    min_ask_low = np.full(
        n,
        np.inf,
        dtype=np.float64,
    )

    max_steps = max(
        HORIZONS.values()
    )

    steps_to_minutes = {
        steps: minutes
        for minutes, steps
        in HORIZONS.items()
    }

    print(
        "Rows:",
        f"{n:,}",
    )

    print(
        "Maximum horizon:",
        max_steps,
        "M5 bars",
    )

    print(
        "Barrier pairs:",
        BARRIERS,
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

        future = slice(
            step,
            n,
        )

        max_bid_high[
            anchors
        ] = np.maximum(
            max_bid_high[
                anchors
            ],
            bid_high[
                future
            ],
        )

        min_bid_low[
            anchors
        ] = np.minimum(
            min_bid_low[
                anchors
            ],
            bid_low[
                future
            ],
        )

        max_ask_high[
            anchors
        ] = np.maximum(
            max_ask_high[
                anchors
            ],
            ask_high[
                future
            ],
        )

        min_ask_low[
            anchors
        ] = np.minimum(
            min_ask_low[
                anchors
            ],
            ask_low[
                future
            ],
        )

        for (
            tp_bps,
            sl_bps,
        ) in BARRIERS:

            state = states[
                (
                    tp_bps,
                    sl_bps,
                )
            ]

            long_tp_price = (
                long_entry[
                    anchors
                ]
                * (
                    1.0
                    + tp_bps
                    / 10000.0
                )
            )

            long_sl_price = (
                long_entry[
                    anchors
                ]
                * (
                    1.0
                    - sl_bps
                    / 10000.0
                )
            )

            short_tp_price = (
                short_entry[
                    anchors
                ]
                * (
                    1.0
                    - tp_bps
                    / 10000.0
                )
            )

            short_sl_price = (
                short_entry[
                    anchors
                ]
                * (
                    1.0
                    + sl_bps
                    / 10000.0
                )
            )

            long_tp_hit = (
                bid_high[
                    future
                ]
                >= long_tp_price
            )

            long_sl_hit = (
                bid_low[
                    future
                ]
                <= long_sl_price
            )

            short_tp_hit = (
                ask_low[
                    future
                ]
                <= short_tp_price
            )

            short_sl_hit = (
                ask_high[
                    future
                ]
                >= short_sl_price
            )

            lt = state[
                "long_tp"
            ][
                anchors
            ]

            ls = state[
                "long_sl"
            ][
                anchors
            ]

            st = state[
                "short_tp"
            ][
                anchors
            ]

            ss = state[
                "short_sl"
            ][
                anchors
            ]

            lt[
                (
                    lt == 0
                )
                & long_tp_hit
            ] = step

            ls[
                (
                    ls == 0
                )
                & long_sl_hit
            ] = step

            st[
                (
                    st == 0
                )
                & short_tp_hit
            ] = step

            ss[
                (
                    ss == 0
                )
                & short_sl_hit
            ] = step

        if step not in (
            steps_to_minutes
        ):
            continue

        minutes = (
            steps_to_minutes[
                step
            ]
        )

        valid = (
            valid_by_horizon[
                minutes
            ]
        )

        print(
            "Snapshot horizon:",
            f"{minutes}m",
            "| valid:",
            f"{valid.sum():,}",
        )

        # Executable excursion metrics.
        long_mfe = (
            (
                max_bid_high
                - long_entry
            )
            / long_entry
            * 10000.0
        )

        long_mae = (
            (
                long_entry
                - min_bid_low
            )
            / long_entry
            * 10000.0
        )

        short_mfe = (
            (
                short_entry
                - min_ask_low
            )
            / short_entry
            * 10000.0
        )

        short_mae = (
            (
                max_ask_high
                - short_entry
            )
            / short_entry
            * 10000.0
        )

        terminal_long = np.full(
            n,
            np.nan,
            dtype=np.float64,
        )

        terminal_short = np.full(
            n,
            np.nan,
            dtype=np.float64,
        )

        terminal_long[
            :valid_n
        ] = (
            (
                bid_close[
                    step:
                ]
                - long_entry[
                    :valid_n
                ]
            )
            / long_entry[
                :valid_n
            ]
            * 10000.0
        )

        terminal_short[
            :valid_n
        ] = (
            (
                short_entry[
                    :valid_n
                ]
                - ask_close[
                    step:
                ]
            )
            / short_entry[
                :valid_n
            ]
            * 10000.0
        )

        output[
            f"long_mfe_bps_h{minutes}"
        ] = masked_float(
            long_mfe,
            valid,
        )

        output[
            f"long_mae_bps_h{minutes}"
        ] = masked_float(
            long_mae,
            valid,
        )

        output[
            f"short_mfe_bps_h{minutes}"
        ] = masked_float(
            short_mfe,
            valid,
        )

        output[
            f"short_mae_bps_h{minutes}"
        ] = masked_float(
            short_mae,
            valid,
        )

        output[
            f"long_terminal_bps_h{minutes}"
        ] = masked_float(
            terminal_long,
            valid,
        )

        output[
            f"short_terminal_bps_h{minutes}"
        ] = masked_float(
            terminal_short,
            valid,
        )

        for (
            tp_bps,
            sl_bps,
        ) in BARRIERS:

            state = states[
                (
                    tp_bps,
                    sl_bps,
                )
            ]

            key = (
                f"h{minutes}_"
                f"tp{tp_bps}_"
                f"sl{sl_bps}"
            )

            long_race = race_from_hits(
                state[
                    "long_tp"
                ],
                state[
                    "long_sl"
                ],
                valid,
            )

            short_race = race_from_hits(
                state[
                    "short_tp"
                ],
                state[
                    "short_sl"
                ],
                valid,
            )

            output[
                f"long_race_{key}"
            ] = long_race

            output[
                f"short_race_{key}"
            ] = short_race

            output[
                f"long_first_tp_bar_{key}"
            ] = masked_int(
                state[
                    "long_tp"
                ],
                valid,
            )

            output[
                f"long_first_sl_bar_{key}"
            ] = masked_int(
                state[
                    "long_sl"
                ],
                valid,
            )

            output[
                f"short_first_tp_bar_{key}"
            ] = masked_int(
                state[
                    "short_tp"
                ],
                valid,
            )

            output[
                f"short_first_sl_bar_{key}"
            ] = masked_int(
                state[
                    "short_sl"
                ],
                valid,
            )

    result = pd.DataFrame(
        output
    )

    path = (
        OUT
        / "multihorizon_targets_v660.parquet"
    )

    result.to_parquet(
        path,
        index=False,
    )

    metadata = {
        "rows":
            int(
                len(
                    result
                )
            ),

        "source":
            str(
                M5_FILE
            ),

        "horizons_minutes":
            list(
                HORIZONS.keys()
            ),

        "horizon_steps":
            HORIZONS,

        "barriers_bps": [
            {
                "tp":
                    tp,

                "sl":
                    sl,
            }
            for tp, sl
            in BARRIERS
        ],

        "race_labels": {
            "1":
                "TP first",

            "0":
                "SL first",

            "-1":
                "timeout",

            "-2":
                "TP and SL first hit "
                "inside same M5 bar",

            "-3":
                "invalid horizon",
        },

        "execution": {
            "long_entry":
                "ask_close",

            "long_tp":
                "future bid_high",

            "long_sl":
                "future bid_low",

            "long_exit":
                "future bid_close",

            "short_entry":
                "bid_close",

            "short_tp":
                "future ask_low",

            "short_sl":
                "future ask_high",

            "short_exit":
                "future ask_close",
        },

        "continuity":
            "Each horizon requires "
            "all future bars to be "
            "exactly contiguous M5 bars.",
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
        "TEN V6.6 TARGET SUMMARY"
    )

    print(
        "=" * 120
    )

    for minutes in HORIZONS:
        print()
        print(
            f"HORIZON {minutes} MINUTES"
        )

        print(
            "-" * 120
        )

        valid = (
            result[
                f"horizon_valid_h{minutes}"
            ].to_numpy()
            == 1
        )

        print(
            "Valid:",
            f"{valid.sum():,}",
            f"({valid.mean():.2%})",
        )

        print(
            "Median LONG MFE:",
            f"{np.nanmedian(result[f'long_mfe_bps_h{minutes}']):.2f}",
            "bps",
        )

        print(
            "Median LONG MAE:",
            f"{np.nanmedian(result[f'long_mae_bps_h{minutes}']):.2f}",
            "bps",
        )

        print(
            "Median SHORT MFE:",
            f"{np.nanmedian(result[f'short_mfe_bps_h{minutes}']):.2f}",
            "bps",
        )

        print(
            "Median SHORT MAE:",
            f"{np.nanmedian(result[f'short_mae_bps_h{minutes}']):.2f}",
            "bps",
        )

        for (
            tp_bps,
            sl_bps,
        ) in BARRIERS:

            key = (
                f"h{minutes}_"
                f"tp{tp_bps}_"
                f"sl{sl_bps}"
            )

            for side in (
                "long",
                "short",
            ):
                race = result[
                    f"{side}_race_{key}"
                ].to_numpy(
                    np.int8
                )

                x = race[
                    valid
                ]

                tp_rate = (
                    x == 1
                ).mean()

                sl_rate = (
                    x == 0
                ).mean()

                timeout_rate = (
                    x == -1
                ).mean()

                ambiguous_rate = (
                    x == -2
                ).mean()

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
                    f"TP{tp_bps}/SL{sl_bps} "
                    f"TP={tp_rate:>6.2%} "
                    f"SL={sl_rate:>6.2%} "
                    f"TO={timeout_rate:>6.2%} "
                    f"AMB={ambiguous_rate:>6.3%} "
                    f"TP|RES={tp_res:>6.2%}"
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
        path,
    )


if __name__ == "__main__":
    main()
