from pathlib import Path

import numpy as np
import pandas as pd


TARGET = Path(
    "training/v6/data_lake/"
    "execution_aligned_targets_v670/"
    "execution_aligned_targets_v670.parquet"
)

OUT = Path(
    "training/artifacts/v6/"
    "daily_opportunity_audit_v670a"
)

START_YEAR = 2016
DESIGN_END_YEAR = 2024

HORIZONS = (
    30,
    60,
    120,
)

BARRIERS = (
    (30, 15),
    (40, 20),
    (60, 30),
)

SIDES = (
    "long",
    "short",
)

COSTS = (
    0.5,
    1.0,
)

# Avoid judging a severely shortened
# holiday/session as a normal trading day.
MIN_BARS_PER_DAY = 100


def make_tasks():
    out = []

    for side in SIDES:
        for horizon in HORIZONS:
            for tp, sl in BARRIERS:
                out.append(
                    {
                        "side": side,
                        "horizon": horizon,
                        "tp": tp,
                        "sl": sl,
                        "key": (
                            f"h{horizon}_"
                            f"tp{tp}_"
                            f"sl{sl}"
                        ),
                    }
                )

    return out


TASKS = make_tasks()


def trading_day_ny(
    timestamps,
):
    # Gold/FX trading day approximately
    # rolls at 17:00 New York.
    #
    # Adding 7h makes 17:00 NY become
    # midnight of the next trading date.
    ny = timestamps.dt.tz_convert(
        "America/New_York"
    )

    return (
        (
            ny
            + pd.Timedelta(
                hours=7
            )
        )
        .dt.floor("D")
        .dt.tz_localize(None)
    )


def profit_factor(
    pnl,
):
    pnl = np.asarray(
        pnl,
        np.float64,
    )

    gain = pnl[
        pnl > 0
    ].sum()

    loss = -pnl[
        pnl < 0
    ].sum()

    if loss <= 0:
        return np.inf

    return float(
        gain / loss
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.7.0A "
        "DAILY OPPORTUNITY CEILING AUDIT"
    )

    print(
        "=" * 126
    )

    columns = [
        "timestamp",
        "year",
        "horizon_valid_h30",
        "horizon_valid_h60",
        "horizon_valid_h120",
    ]

    for task in TASKS:
        side = task[
            "side"
        ]

        key = task[
            "key"
        ]

        columns += [
            f"{side}_outcome_{key}",
            f"{side}_gross_bps_{key}",
            f"{side}_win_c05_{key}",
            f"{side}_win_c10_{key}",
        ]

    df = pd.read_parquet(
        TARGET,
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

    year = df[
        "year"
    ].to_numpy(
        np.int16
    )

    day = trading_day_ny(
        df[
            "timestamp"
        ]
    )

    print(
        "Rows:",
        f"{n:,}",
    )

    print(
        "Design years:",
        START_YEAR,
        "through",
        DESIGN_END_YEAR,
    )

    print(
        "2025 excluded from design audit."
    )

    print(
        "2026 excluded."
    )

    # -----------------------------------
    # Per-row opportunity surface
    # -----------------------------------

    best05 = np.full(
        n,
        -np.inf,
        dtype=np.float32,
    )

    best10 = np.full(
        n,
        -np.inf,
        dtype=np.float32,
    )

    best05_long = np.full(
        n,
        -np.inf,
        dtype=np.float32,
    )

    best05_short = np.full(
        n,
        -np.inf,
        dtype=np.float32,
    )

    positive05 = np.zeros(
        n,
        dtype=np.int16,
    )

    positive10 = np.zeros(
        n,
        dtype=np.int16,
    )

    any_tp = np.zeros(
        n,
        dtype=bool,
    )

    task_year_records = []
    task_day_frames = []

    label_mismatches = 0

    for task in TASKS:

        side = task[
            "side"
        ]

        horizon = task[
            "horizon"
        ]

        tp = task[
            "tp"
        ]

        sl = task[
            "sl"
        ]

        key = task[
            "key"
        ]

        valid = (
            df[
                f"horizon_valid_h{horizon}"
            ].to_numpy(
                np.uint8
            )
            == 1
        )

        gross = df[
            f"{side}_gross_bps_{key}"
        ].to_numpy(
            np.float32
        )

        outcome = df[
            f"{side}_outcome_{key}"
        ].to_numpy(
            np.int8
        )

        stored05 = df[
            f"{side}_win_c05_{key}"
        ].to_numpy(
            np.uint8
        )

        stored10 = df[
            f"{side}_win_c10_{key}"
        ].to_numpy(
            np.uint8
        )

        calc05 = (
            gross
            - 0.5
            > 0
        )

        calc10 = (
            gross
            - 1.0
            > 0
        )

        mismatch05 = np.sum(
            valid
            & (
                stored05
                != calc05.astype(
                    np.uint8
                )
            )
        )

        mismatch10 = np.sum(
            valid
            & (
                stored10
                != calc10.astype(
                    np.uint8
                )
            )
        )

        label_mismatches += int(
            mismatch05
            + mismatch10
        )

        if (
            not np.all(
                np.isfinite(
                    gross[
                        valid
                    ]
                )
            )
        ):
            raise RuntimeError(
                f"Non-finite gross in "
                f"{side} {key}"
            )

        net05 = (
            gross
            - 0.5
        )

        net10 = (
            gross
            - 1.0
        )

        best05[
            valid
        ] = np.maximum(
            best05[
                valid
            ],
            net05[
                valid
            ],
        )

        best10[
            valid
        ] = np.maximum(
            best10[
                valid
            ],
            net10[
                valid
            ],
        )

        if side == "long":
            best05_long[
                valid
            ] = np.maximum(
                best05_long[
                    valid
                ],
                net05[
                    valid
                ],
            )

        else:
            best05_short[
                valid
            ] = np.maximum(
                best05_short[
                    valid
                ],
                net05[
                    valid
                ],
            )

        positive05 += (
            valid
            & (
                net05 > 0
            )
        ).astype(
            np.int16
        )

        positive10 += (
            valid
            & (
                net10 > 0
            )
        ).astype(
            np.int16
        )

        any_tp |= (
            valid
            & (
                outcome == 1
            )
        )

        # -------------------------------
        # Year-level task statistics
        # -------------------------------

        task_tmp = pd.DataFrame(
            {
                "year":
                    year[
                        valid
                    ],

                "gross":
                    gross[
                        valid
                    ],

                "win05":
                    calc05[
                        valid
                    ].astype(
                        np.float32
                    ),

                "win10":
                    calc10[
                        valid
                    ].astype(
                        np.float32
                    ),
            }
        )

        grouped = task_tmp.groupby(
            "year",
            sort=True,
        )

        for yr, g in grouped:
            if (
                yr < START_YEAR
                or yr > DESIGN_END_YEAR
            ):
                continue

            task_year_records.append(
                {
                    "year":
                        int(yr),

                    "side":
                        side.upper(),

                    "horizon":
                        horizon,

                    "tp":
                        tp,

                    "sl":
                        sl,

                    "n":
                        len(g),

                    "win05":
                        float(
                            g[
                                "win05"
                            ].mean()
                        ),

                    "win10":
                        float(
                            g[
                                "win10"
                            ].mean()
                        ),

                    "mean_gross":
                        float(
                            g[
                                "gross"
                            ].mean()
                        ),
                }
            )

        # -------------------------------
        # Daily task opportunity ceiling
        # -------------------------------

        task_daily = pd.DataFrame(
            {
                "trading_day":
                    day[
                        valid
                    ].to_numpy(),

                "net05":
                    net05[
                        valid
                    ],

                "net10":
                    net10[
                        valid
                    ],
            }
        )

        td = task_daily.groupby(
            "trading_day",
            sort=True,
        ).agg(
            best_net05=(
                "net05",
                "max",
            ),

            best_net10=(
                "net10",
                "max",
            ),

            positive05=(
                "net05",
                lambda x:
                    int(
                        (
                            x > 0
                        ).sum()
                    ),
            ),
        ).reset_index()

        td[
            "side"
        ] = side.upper()

        td[
            "horizon"
        ] = horizon

        td[
            "tp"
        ] = tp

        td[
            "sl"
        ] = sl

        task_day_frames.append(
            td
        )

    print(
        "Stored win-label mismatches:",
        label_mismatches,
    )

    if label_mismatches:
        raise RuntimeError(
            "V6.7.0 WIN LABEL "
            "CONSISTENCY FAILURE"
        )

    # Replace impossible row state
    # with NaN before daily aggregation.
    for x in (
        best05,
        best10,
        best05_long,
        best05_short,
    ):
        x[
            ~np.isfinite(
                x
            )
        ] = np.nan

    row_frame = pd.DataFrame(
        {
            "trading_day":
                day,

            "year":
                day.dt.year.to_numpy(
                    np.int16
                ),

            "best05":
                best05,

            "best10":
                best10,

            "best05_long":
                best05_long,

            "best05_short":
                best05_short,

            "positive05":
                positive05,

            "positive10":
                positive10,

            "any_tp":
                any_tp.astype(
                    np.uint8
                ),
        }
    )

    daily = row_frame.groupby(
        "trading_day",
        sort=True,
    ).agg(
        year=(
            "year",
            "first",
        ),

        bars=(
            "year",
            "size",
        ),

        best_net05=(
            "best05",
            "max",
        ),

        best_net10=(
            "best10",
            "max",
        ),

        best_long_net05=(
            "best05_long",
            "max",
        ),

        best_short_net05=(
            "best05_short",
            "max",
        ),

        positive_candidates05=(
            "positive05",
            "sum",
        ),

        positive_candidates10=(
            "positive10",
            "sum",
        ),

        any_tp=(
            "any_tp",
            "max",
        ),
    ).reset_index()

    daily = daily[
        (
            daily[
                "year"
            ]
            >= START_YEAR
        )
        & (
            daily[
                "year"
            ]
            <= DESIGN_END_YEAR
        )
    ].copy()

    daily[
        "full_day"
    ] = (
        daily[
            "bars"
        ]
        >= MIN_BARS_PER_DAY
    )

    daily.to_csv(
        OUT
        / "daily_opportunity_v670a.csv",
        index=False,
    )

    task_year = pd.DataFrame(
        task_year_records
    )

    task_year.to_csv(
        OUT
        / "task_year_stats_v670a.csv",
        index=False,
    )

    task_daily = pd.concat(
        task_day_frames,
        ignore_index=True,
    )

    task_daily[
        "year"
    ] = pd.to_datetime(
        task_daily[
            "trading_day"
        ]
    ).dt.year

    task_daily = task_daily[
        (
            task_daily[
                "year"
            ]
            >= START_YEAR
        )
        & (
            task_daily[
                "year"
            ]
            <= DESIGN_END_YEAR
        )
    ]

    task_daily.to_csv(
        OUT
        / "task_daily_opportunity_v670a.csv",
        index=False,
    )

    # -----------------------------------
    # Daily oracle ceiling
    # -----------------------------------

    full = daily[
        daily[
            "full_day"
        ]
    ].copy()

    print()
    print(
        "=" * 126
    )

    print(
        "YEAR-BY-YEAR DAILY "
        "OPPORTUNITY CEILING"
    )

    print(
        "=" * 126
    )

    print(
        "YEAR DAYS  COVER+.5 COVER+1 "
        "TP-DAY  LONG+.5 SHORT+.5 "
        "MED_POS05  MED_BEST05"
    )

    yearly_rows = []

    for yr, g in full.groupby(
        "year",
        sort=True,
    ):

        cover05 = float(
            (
                g[
                    "best_net05"
                ]
                > 0
            ).mean()
        )

        cover10 = float(
            (
                g[
                    "best_net10"
                ]
                > 0
            ).mean()
        )

        tp_day = float(
            g[
                "any_tp"
            ].mean()
        )

        long_cover = float(
            (
                g[
                    "best_long_net05"
                ]
                > 0
            ).mean()
        )

        short_cover = float(
            (
                g[
                    "best_short_net05"
                ]
                > 0
            ).mean()
        )

        med_positive = float(
            g[
                "positive_candidates05"
            ].median()
        )

        med_best = float(
            g[
                "best_net05"
            ].median()
        )

        yearly_rows.append(
            {
                "year":
                    int(yr),

                "days":
                    len(g),

                "coverage05":
                    cover05,

                "coverage10":
                    cover10,

                "tp_day":
                    tp_day,

                "long_coverage05":
                    long_cover,

                "short_coverage05":
                    short_cover,

                "median_positive_candidates05":
                    med_positive,

                "median_best_net05":
                    med_best,
            }
        )

        print(
            f"{int(yr):<4} "
            f"{len(g):>4} "
            f"{cover05:>8.2%} "
            f"{cover10:>7.2%} "
            f"{tp_day:>7.2%} "
            f"{long_cover:>8.2%} "
            f"{short_cover:>9.2%} "
            f"{med_positive:>9.0f} "
            f"{med_best:>11.2f}"
        )

    pd.DataFrame(
        yearly_rows
    ).to_csv(
        OUT
        / "yearly_daily_ceiling_v670a.csv",
        index=False,
    )

    print()
    print(
        "=" * 126
    )

    print(
        "2016-2024 COMBINED "
        "DAILY CEILING"
    )

    print(
        "=" * 126
    )

    for cost, col in (
        (
            0.5,
            "best_net05",
        ),
        (
            1.0,
            "best_net10",
        ),
    ):
        x = full[
            col
        ].to_numpy(
            np.float64
        )

        print(
            f"EXTRA COST {cost:.1f}bps"
        )

        print(
            "  Trading days:",
            f"{len(x):,}",
        )

        print(
            "  Days with >=1 "
            "net-positive opportunity:",
            f"{(x > 0).mean():.2%}",
        )

        print(
            "  Oracle one-trade/day "
            "win rate:",
            f"{(x > 0).mean():.2%}",
        )

        print(
            "  Oracle mean best "
            "daily net:",
            f"{np.nanmean(x):+.2f} bps",
        )

        print(
            "  Oracle median best "
            "daily net:",
            f"{np.nanmedian(x):+.2f} bps",
        )

        print(
            "  Oracle P10 best "
            "daily net:",
            f"{np.nanquantile(x, .10):+.2f} bps",
        )

        print(
            "  Oracle daily PF:",
            f"{profit_factor(x):.3f}",
        )

    print()
    print(
        "LONG-only daily coverage @0.5:",
        f"{(full['best_long_net05'] > 0).mean():.2%}",
    )

    print(
        "SHORT-only daily coverage @0.5:",
        f"{(full['best_short_net05'] > 0).mean():.2%}",
    )

    print(
        "Days with at least one "
        "full TP anywhere:",
        f"{full['any_tp'].mean():.2%}",
    )

    print(
        "Median positive anchor-task "
        "candidates/day @0.5:",
        f"{full['positive_candidates05'].median():.0f}",
    )

    print(
        "P10 positive candidates/day @0.5:",
        f"{full['positive_candidates05'].quantile(.10):.0f}",
    )

    # -----------------------------------
    # Which fixed task has the greatest
    # daily opportunity availability?
    # -----------------------------------

    task_summary = (
        task_daily
        .groupby(
            [
                "side",
                "horizon",
                "tp",
                "sl",
            ],
            as_index=False,
        )
        .agg(
            days=(
                "trading_day",
                "nunique",
            ),

            coverage05=(
                "best_net05",
                lambda x:
                    float(
                        (
                            x > 0
                        ).mean()
                    ),
            ),

            coverage10=(
                "best_net10",
                lambda x:
                    float(
                        (
                            x > 0
                        ).mean()
                    ),
            ),

            median_best05=(
                "best_net05",
                "median",
            ),

            median_positive05=(
                "positive05",
                "median",
            ),
        )
        .sort_values(
            [
                "coverage05",
                "median_best05",
            ],
            ascending=False,
        )
    )

    task_summary.to_csv(
        OUT
        / "task_daily_summary_v670a.csv",
        index=False,
    )

    print()
    print(
        "=" * 126
    )

    print(
        "TOP FIXED TASKS BY "
        "DAILY OPPORTUNITY COVERAGE"
    )

    print(
        "=" * 126
    )

    print(
        task_summary
        .head(
            18
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is an ORACLE FEASIBILITY "
        "CEILING, not model performance."
    )

    print(
        "Future outcomes are used only "
        "to answer whether a profitable "
        "opportunity existed that day."
    )

    print(
        "No V6.7 model or threshold is "
        "selected from this audit."
    )

    print(
        "2025 was not used."
    )

    print(
        "2026 was not used."
    )

    print(
        "Saved:",
        OUT,
    )


if __name__ == "__main__":
    main()
