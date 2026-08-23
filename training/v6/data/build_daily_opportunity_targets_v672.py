from pathlib import Path
import time

import numpy as np
import pandas as pd

import training.v6.models.train_multisurface_technical_brain_v661 as brain


SOURCE = Path(
    "training/v6/data_lake/"
    "execution_aligned_targets_v670/"
    "execution_aligned_targets_v670.parquet"
)

OUT = Path(
    "training/v6/data_lake/"
    "daily_opportunity_targets_v672"
)

OUT_FILE = (
    OUT
    / "daily_opportunity_targets_v672.parquet"
)

COSTS = (
    0.5,
    1.0,
)

TOP_QUANTILES = (
    0.80,
    0.90,
    0.95,
    0.98,
    0.99,
)


def trading_day_ny(
    timestamp,
):
    ny = (
        timestamp
        .dt.tz_convert(
            "America/New_York"
        )
    )

    # Gold / FX trading-day roll:
    # 17:00 New York.
    return (
        (
            ny
            + pd.Timedelta(
                hours=7
            )
        )
        .dt.floor(
            "D"
        )
        .dt.tz_localize(
            None
        )
    )


def task_meta():
    rows = []

    for j, meta in enumerate(
        brain.TASKS
    ):
        rows.append(
            {
                "task":
                    j,

                "side":
                    str(
                        meta["side"]
                    ).lower(),

                "side_id":
                    int(
                        meta["side_id"]
                    ),

                "horizon":
                    int(
                        meta["horizon"]
                    ),

                "tp":
                    int(
                        meta["tp"]
                    ),

                "sl":
                    int(
                        meta["sl"]
                    ),
            }
        )

    return rows


TASKS = task_meta()


def rank_within_day(
    values,
    day,
):
    out = np.full(
        len(values),
        np.nan,
        dtype=np.float32,
    )

    valid = np.isfinite(
        values
    )

    tmp = pd.DataFrame(
        {
            "idx":
                np.flatnonzero(
                    valid
                ),

            "day":
                day[
                    valid
                ].to_numpy(),

            "value":
                values[
                    valid
                ],
        }
    )

    tmp[
        "rank"
    ] = (
        tmp
        .groupby(
            "day",
            sort=False,
        )[
            "value"
        ]
        .rank(
            method="average",
            pct=True,
        )
    )

    out[
        tmp[
            "idx"
        ].to_numpy(
            np.int64
        )
    ] = tmp[
        "rank"
    ].to_numpy(
        np.float32
    )

    return out


def main():
    started = time.time()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.7.2 "
        "DAILY OPPORTUNITY TARGET BUILDER"
    )

    print(
        "=" * 120
    )

    columns = [
        "source_row",
        "timestamp",
        "year",
        "horizon_valid_h30",
        "horizon_valid_h60",
        "horizon_valid_h120",
    ]

    for meta in TASKS:
        side = meta[
            "side"
        ]

        h = meta[
            "horizon"
        ]

        tp = meta[
            "tp"
        ]

        sl = meta[
            "sl"
        ]

        key = (
            f"h{h}_"
            f"tp{tp}_"
            f"sl{sl}"
        )

        columns.append(
            f"{side}_gross_bps_{key}"
        )

    df = pd.read_parquet(
        SOURCE,
        columns=columns,
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

    print(
        "Rows:",
        f"{n:,}",
    )

    source_row = df[
        "source_row"
    ].to_numpy(
        np.int64
    )

    if not np.array_equal(
        source_row,
        np.arange(
            n,
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "source_row alignment failure"
        )

    day = trading_day_ny(
        df[
            "timestamp"
        ]
    )

    gross = np.full(
        (
            n,
            18,
        ),
        np.nan,
        dtype=np.float32,
    )

    valid = np.zeros(
        (
            n,
            18,
        ),
        dtype=bool,
    )

    for meta in TASKS:

        j = meta[
            "task"
        ]

        side = meta[
            "side"
        ]

        h = meta[
            "horizon"
        ]

        tp = meta[
            "tp"
        ]

        sl = meta[
            "sl"
        ]

        key = (
            f"h{h}_"
            f"tp{tp}_"
            f"sl{sl}"
        )

        valid[
            :,
            j
        ] = (
            df[
                f"horizon_valid_h{h}"
            ].to_numpy(
                np.uint8
            )
            == 1
        )

        gross[
            :,
            j
        ] = df[
            f"{side}_gross_bps_{key}"
        ].to_numpy(
            np.float32
        )

    gross[
        ~valid
    ] = np.nan

    long_idx = np.arange(
        0,
        9,
        dtype=np.int64,
    )

    short_idx = np.arange(
        9,
        18,
        dtype=np.int64,
    )

    output = {
        "source_row":
            source_row,

        "timestamp":
            df[
                "timestamp"
            ],

        "year":
            df[
                "year"
            ].to_numpy(
                np.int16
            ),

        "trading_day":
            day,
    }

    for cost in COSTS:

        suffix = (
            "05"
            if cost == 0.5
            else "10"
        )

        net = (
            gross
            - cost
        )

        safe = np.where(
            np.isfinite(
                net
            ),
            net,
            -np.inf,
        )

        best_task = np.argmax(
            safe,
            axis=1,
        ).astype(
            np.int8
        )

        row_idx = np.arange(
            n,
            dtype=np.int64,
        )

        best_net = safe[
            row_idx,
            best_task,
        ].astype(
            np.float32
        )

        no_valid = (
            ~np.isfinite(
                best_net
            )
        )

        best_net[
            no_valid
        ] = np.nan

        best_task[
            no_valid
        ] = -1

        # Second-best action.
        partitioned = np.partition(
            safe,
            kth=-2,
            axis=1,
        )

        second_net = partitioned[
            :,
            -2
        ].astype(
            np.float32
        )

        second_net[
            ~np.isfinite(
                second_net
            )
        ] = np.nan

        action_gap = (
            best_net
            - second_net
        ).astype(
            np.float32
        )

        positive_count = (
            (
                net > 0
            )
            & valid
        ).sum(
            axis=1
        ).astype(
            np.int8
        )

        # Best LONG and SHORT separately.
        safe_long = safe[
            :,
            long_idx
        ]

        safe_short = safe[
            :,
            short_idx
        ]

        best_long = np.max(
            safe_long,
            axis=1,
        ).astype(
            np.float32
        )

        best_short = np.max(
            safe_short,
            axis=1,
        ).astype(
            np.float32
        )

        best_long[
            ~np.isfinite(
                best_long
            )
        ] = np.nan

        best_short[
            ~np.isfinite(
                best_short
            )
        ] = np.nan

        side_id = np.full(
            n,
            -1,
            dtype=np.int8,
        )

        both = (
            np.isfinite(
                best_long
            )
            & np.isfinite(
                best_short
            )
        )

        side_id[
            both
        ] = (
            best_short[
                both
            ]
            > best_long[
                both
            ]
        ).astype(
            np.int8
        )

        only_long = (
            np.isfinite(
                best_long
            )
            & ~np.isfinite(
                best_short
            )
        )

        only_short = (
            np.isfinite(
                best_short
            )
            & ~np.isfinite(
                best_long
            )
        )

        side_id[
            only_long
        ] = 0

        side_id[
            only_short
        ] = 1

        direction_gap = (
            best_short
            - best_long
        ).astype(
            np.float32
        )

        daily_rank = rank_within_day(
            best_net,
            day,
        )

        # Exact maximum of the trading day.
        temp = pd.DataFrame(
            {
                "day":
                    day,

                "best_net":
                    best_net,
            }
        )

        day_max = (
            temp
            .groupby(
                "day",
                sort=False,
            )[
                "best_net"
            ]
            .transform(
                "max"
            )
            .to_numpy(
                np.float32
            )
        )

        daily_best = (
            np.isfinite(
                best_net
            )
            & np.isclose(
                best_net,
                day_max,
                rtol=0.0,
                atol=1e-6,
            )
        ).astype(
            np.uint8
        )

        output[
            f"best_task_c{suffix}"
        ] = best_task

        output[
            f"best_side_c{suffix}"
        ] = side_id

        output[
            f"best_net_c{suffix}"
        ] = best_net

        output[
            f"second_net_c{suffix}"
        ] = second_net

        output[
            f"action_gap_c{suffix}"
        ] = action_gap

        output[
            f"best_long_net_c{suffix}"
        ] = best_long

        output[
            f"best_short_net_c{suffix}"
        ] = best_short

        output[
            f"direction_gap_c{suffix}"
        ] = direction_gap

        output[
            f"positive_task_count_c{suffix}"
        ] = positive_count

        output[
            f"daily_rank_c{suffix}"
        ] = daily_rank

        output[
            f"daily_best_c{suffix}"
        ] = daily_best

        for q in TOP_QUANTILES:

            qname = int(
                round(
                    q * 100
                )
            )

            label = (
                np.isfinite(
                    daily_rank
                )
                & (
                    daily_rank
                    >= q
                )
                & (
                    best_net
                    > 0
                )
            ).astype(
                np.uint8
            )

            output[
                f"daily_top{qname}_c{suffix}"
            ] = label

    result = pd.DataFrame(
        output
    )

    result.to_parquet(
        OUT_FILE,
        index=False,
    )

    print()
    print(
        "=" * 120
    )

    print(
        "V6.7.2 DAILY TARGET SUMMARY "
        "(DESIGN YEARS 2016-2024)"
    )

    print(
        "=" * 120
    )

    design = result[
        (
            result[
                "year"
            ]
            >= 2016
        )
        & (
            result[
                "year"
            ]
            <= 2024
        )
    ].copy()

    for suffix, cost in (
        (
            "05",
            0.5,
        ),
        (
            "10",
            1.0,
        ),
    ):

        print()
        print(
            f"EXTRA COST {cost:.1f}bps"
        )

        print(
            "-" * 90
        )

        print(
            "Positive best-bar rate:",
            f"{(design[f'best_net_c{suffix}'] > 0).mean():.2%}",
        )

        print(
            "Median best net/bar:",
            f"{design[f'best_net_c{suffix}'].median():+.2f} bps",
        )

        print(
            "Median action gap:",
            f"{design[f'action_gap_c{suffix}'].median():+.2f} bps",
        )

        print(
            "Median positive tasks/bar:",
            f"{design[f'positive_task_count_c{suffix}'].median():.0f}",
        )

        print(
            "LONG better:",
            f"{(design[f'best_side_c{suffix}'] == 0).mean():.2%}",
        )

        print(
            "SHORT better:",
            f"{(design[f'best_side_c{suffix}'] == 1).mean():.2%}",
        )

        for q in TOP_QUANTILES:

            qname = int(
                round(
                    q * 100
                )
            )

            col = (
                f"daily_top{qname}_"
                f"c{suffix}"
            )

            print(
                f"Top {100-qname:>2}% "
                f"positive opportunity bars:",
                f"{design[col].mean():.3%}",
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
        "2025/2026 labels were generated "
        "but NOT summarized or used "
        "for model selection."
    )

    print(
        "Elapsed:",
        f"{time.time() - started:.2f}s",
    )


if __name__ == "__main__":
    main()
