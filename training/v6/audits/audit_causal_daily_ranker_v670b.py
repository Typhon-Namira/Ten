from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch

import training.v6.models.train_multisurface_technical_brain_v661 as brain
import training.v6.backtests.backtest_surface_policy_v662 as execmod


OUT = Path(
    "training/artifacts/v6/"
    "causal_daily_ranker_v670b"
)

TARGET = Path(
    "training/v6/data_lake/"
    "execution_aligned_targets_v670/"
    "execution_aligned_targets_v670.parquet"
)

SCORE_MODES = (
    "ptp_rank",
    "race_rank",
    "economic_rank",
    "hybrid",
)

ENTRY_THRESHOLDS = (
    0.70,
    0.80,
    0.90,
    0.95,
    0.98,
    0.99,
    0.995,
)

MAX_TRADES_PER_DAY = (
    1,
    2,
    3,
)

PERSISTENCES = (
    1,
    2,
)

SIDE_MARGINS = (
    0.00,
    0.02,
)

SELECTION_COST = 0.50

STRESS_COSTS = (
    0.0,
    0.5,
    1.0,
    2.0,
)

MIN_DAY_BARS = 100

# This is deliberately demanding.
MIN_SELECTION_COVERAGE = 0.80
MIN_SELECTION_TRADES = 150
MIN_HALF_TRADES = 50

STEP_NS = 300_000_000_000


def trading_day_ny(
    timestamps,
):
    x = pd.Series(
        pd.to_datetime(
            timestamps,
            utc=True,
        )
    )

    ny = x.dt.tz_convert(
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
        .to_numpy()
    )


def build_task_metadata():
    rows = []

    for j, meta in enumerate(
        brain.TASKS
    ):
        side = str(
            meta["side"]
        ).lower()

        side_id = (
            0
            if side == "long"
            else 1
        )

        horizon = int(
            meta["horizon"]
        )

        tp = int(
            meta["tp"]
        )

        sl = int(
            meta["sl"]
        )

        exec_task = execmod.TASK_LOOKUP[
            (
                side_id,
                horizon,
                tp,
                sl,
            )
        ]

        rows.append(
            {
                "brain_task":
                    j,

                "exec_task":
                    int(
                        exec_task
                    ),

                "side":
                    side,

                "side_id":
                    side_id,

                "horizon":
                    horizon,

                "tp":
                    tp,

                "sl":
                    sl,
            }
        )

    return rows


TASK_META = build_task_metadata()

LONG_TASKS = np.array(
    [
        x["brain_task"]
        for x in TASK_META
        if x["side_id"] == 0
    ],
    dtype=np.int64,
)

SHORT_TASKS = np.array(
    [
        x["brain_task"]
        for x in TASK_META
        if x["side_id"] == 1
    ],
    dtype=np.int64,
)


def raw_scores(
    race_prob,
):
    p_tp = race_prob[
        :,
        :,
        0
    ].astype(
        np.float64
    )

    p_sl = race_prob[
        :,
        :,
        1
    ].astype(
        np.float64
    )

    ptp = p_tp.copy()

    race = (
        p_tp
        / np.maximum(
            p_tp + p_sl,
            1e-12,
        )
    )

    econ = np.zeros_like(
        p_tp
    )

    for meta in TASK_META:
        j = meta[
            "brain_task"
        ]

        econ[
            :,
            j
        ] = (
            p_tp[
                :,
                j
            ]
            * meta[
                "tp"
            ]
            -
            p_sl[
                :,
                j
            ]
            * meta[
                "sl"
            ]
        )

    return {
        "ptp":
            ptp,

        "race":
            race,

        "economic":
            econ,
    }


def empirical_rank(
    reference,
    values,
):
    if (
        reference.shape[1]
        != values.shape[1]
    ):
        raise ValueError(
            "Task dimension mismatch"
        )

    out = np.empty(
        values.shape,
        dtype=np.float32,
    )

    for j in range(
        values.shape[1]
    ):
        ref = np.sort(
            reference[
                :,
                j
            ].astype(
                np.float64
            )
        )

        out[
            :,
            j
        ] = (
            np.searchsorted(
                ref,
                values[
                    :,
                    j
                ],
                side="right",
            )
            / len(
                ref
            )
        ).astype(
            np.float32
        )

    return out


def ranked_scores(
    ref_pred,
    cur_pred,
):
    ref_raw = raw_scores(
        ref_pred[
            "race_prob"
        ]
    )

    cur_raw = raw_scores(
        cur_pred[
            "race_prob"
        ]
    )

    ptp_rank = empirical_rank(
        ref_raw[
            "ptp"
        ],
        cur_raw[
            "ptp"
        ],
    )

    race_rank = empirical_rank(
        ref_raw[
            "race"
        ],
        cur_raw[
            "race"
        ],
    )

    economic_rank = empirical_rank(
        ref_raw[
            "economic"
        ],
        cur_raw[
            "economic"
        ],
    )

    # All components are now task-specific
    # percentiles, so they are comparable.
    hybrid = (
        0.35 * ptp_rank
        + 0.35 * race_rank
        + 0.30 * economic_rank
    ).astype(
        np.float32
    )

    return {
        "ptp_rank":
            ptp_rank,

        "race_rank":
            race_rank,

        "economic_rank":
            economic_rank,

        "hybrid":
            hybrid,
    }


def build_valid_matrix(
    target,
    rows,
):
    n = len(
        rows
    )

    valid = np.zeros(
        (
            n,
            len(
                TASK_META
            ),
        ),
        dtype=bool,
    )

    for meta in TASK_META:
        j = meta[
            "brain_task"
        ]

        horizon = meta[
            "horizon"
        ]

        valid[
            :,
            j
        ] = (
            target[
                f"horizon_valid_h{horizon}"
            ].to_numpy(
                np.uint8
            )[
                rows
            ]
            == 1
        )

    return valid


def metrics(
    trades,
    cost,
):
    if len(trades) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "net": 0.0,
            "pf": np.nan,
            "win": np.nan,
            "dd": np.nan,
            "tp_rate": np.nan,
        }

    pnl = (
        trades[
            "gross_bps"
        ].to_numpy(
            np.float64
        )
        - cost
    )

    gains = pnl[
        pnl > 0
    ].sum()

    losses = -pnl[
        pnl < 0
    ].sum()

    pf = (
        gains / losses
        if losses > 0
        else np.inf
    )

    equity = np.cumsum(
        pnl
    )

    peak = np.maximum.accumulate(
        np.r_[
            0.0,
            equity,
        ]
    )[1:]

    dd = float(
        -np.min(
            equity
            - peak
        )
    )

    return {
        "n":
            len(
                trades
            ),

        "mean":
            float(
                pnl.mean()
            ),

        "net":
            float(
                pnl.sum()
            ),

        "pf":
            float(
                pf
            ),

        "win":
            float(
                (
                    pnl > 0
                ).mean()
            ),

        "dd":
            dd,

        "tp_rate":
            float(
                (
                    trades[
                        "outcome"
                    ]
                    == "TP"
                ).mean()
            ),
    }


def half_metrics(
    trades,
    cost,
):
    if len(trades) == 0:
        empty = metrics(
            trades,
            cost,
        )

        return (
            empty,
            empty,
        )

    ts = pd.to_datetime(
        trades[
            "signal_timestamp"
        ],
        utc=True,
    )

    h1 = trades[
        ts.dt.month <= 6
    ]

    h2 = trades[
        ts.dt.month >= 7
    ]

    return (
        metrics(
            h1,
            cost,
        ),
        metrics(
            h2,
            cost,
        ),
    )


def full_days(
    day,
):
    x = pd.Series(
        day
    )

    counts = x.value_counts()

    return set(
        counts[
            counts
            >= MIN_DAY_BARS
        ].index
    )


def simulate(
    score,
    valid,
    source,
    timestamps_ns,
    day,
    m5,
    threshold,
    max_per_day,
    persistence,
    side_margin,
):
    n = len(
        source
    )

    eligible_days = full_days(
        day
    )

    trades = []

    trade_count = {}

    occupied_until = -1

    previous_task = -1
    previous_source = -10
    streak = 0

    stats = {
        "candidate_bars":
            0,

        "threshold_bars":
            0,

        "suppressed_open":
            0,

        "suppressed_daily_limit":
            0,

        "suppressed_side_margin":
            0,

        "invalid_execution":
            0,
    }

    for i in range(
        n
    ):
        s = int(
            source[
                i
            ]
        )

        d = pd.Timestamp(
            day[
                i
            ]
        )

        row_score = score[
            i
        ].copy()

        row_score[
            ~valid[
                i
            ]
        ] = -np.inf

        long_local = LONG_TASKS[
            np.argmax(
                row_score[
                    LONG_TASKS
                ]
            )
        ]

        short_local = SHORT_TASKS[
            np.argmax(
                row_score[
                    SHORT_TASKS
                ]
            )
        ]

        long_score = float(
            row_score[
                long_local
            ]
        )

        short_score = float(
            row_score[
                short_local
            ]
        )

        if (
            not np.isfinite(
                long_score
            )
            and not np.isfinite(
                short_score
            )
        ):
            previous_task = -1
            streak = 0
            continue

        if (
            long_score
            >= short_score
        ):
            task = int(
                long_local
            )

            best_score = (
                long_score
            )

            opposite = (
                short_score
            )

        else:
            task = int(
                short_local
            )

            best_score = (
                short_score
            )

            opposite = (
                long_score
            )

        stats[
            "candidate_bars"
        ] += 1

        if (
            best_score
            < threshold
        ):
            previous_task = -1
            previous_source = s
            streak = 0
            continue

        stats[
            "threshold_bars"
        ] += 1

        if (
            np.isfinite(
                opposite
            )
            and (
                best_score
                - opposite
            )
            < side_margin
        ):
            stats[
                "suppressed_side_margin"
            ] += 1

            previous_task = -1
            streak = 0
            continue

        contiguous = (
            s
            == previous_source + 1
            and (
                timestamps_ns[
                    s
                ]
                - timestamps_ns[
                    previous_source
                ]
                == STEP_NS
            )
            if (
                previous_source >= 0
            )
            else False
        )

        if (
            task == previous_task
            and contiguous
        ):
            streak += 1
        else:
            streak = 1

        previous_task = task
        previous_source = s

        if (
            streak
            < persistence
        ):
            continue

        if (
            s
            <= occupied_until
        ):
            stats[
                "suppressed_open"
            ] += 1
            continue

        if (
            d
            not in eligible_days
        ):
            continue

        used = trade_count.get(
            d,
            0,
        )

        if (
            used
            >= max_per_day
        ):
            stats[
                "suppressed_daily_limit"
            ] += 1
            continue

        meta = TASK_META[
            task
        ]

        result = execmod.execute_trade(
            m5,
            timestamps_ns,
            s,
            meta[
                "side_id"
            ],
            meta[
                "exec_task"
            ],
        )

        if result is None:
            stats[
                "invalid_execution"
            ] += 1
            continue

        entry_row = (
            s + 1
        )

        trades.append(
            {
                "signal_source":
                    s,

                "signal_timestamp":
                    m5[
                        "timestamp"
                    ][s],

                "entry_timestamp":
                    m5[
                        "timestamp"
                    ][entry_row],

                "exit_timestamp":
                    m5[
                        "timestamp"
                    ][
                        result[
                            "exit_row"
                        ]
                    ],

                "trading_day":
                    d,

                "side":
                    meta[
                        "side"
                    ].upper(),

                "horizon":
                    meta[
                        "horizon"
                    ],

                "tp_bps":
                    meta[
                        "tp"
                    ],

                "sl_bps":
                    meta[
                        "sl"
                    ],

                "task":
                    task,

                "score":
                    best_score,

                "outcome":
                    result[
                        "outcome"
                    ],

                "gross_bps":
                    float(
                        result[
                            "gross_bps"
                        ]
                    ),

                "hold_bars":
                    int(
                        result[
                            "hold_bars"
                        ]
                    ),
            }
        )

        trade_count[
            d
        ] = (
            used + 1
        )

        # One position maximum.
        occupied_until = (
            int(
                result[
                    "exit_row"
                ]
            )
            + 1
        )

    trades = pd.DataFrame(
        trades
    )

    traded_days = (
        set()
        if len(
            trades
        ) == 0
        else set(
            trades[
                "trading_day"
            ].unique()
        )
    )

    coverage = (
        len(
            traded_days
            & eligible_days
        )
        / len(
            eligible_days
        )
        if eligible_days
        else np.nan
    )

    return (
        trades,
        stats,
        coverage,
        len(
            eligible_days
        ),
    )


def selection_score(
    m,
    h1,
    h2,
    coverage,
):
    if (
        m["n"]
        < MIN_SELECTION_TRADES
        or coverage
        < MIN_SELECTION_COVERAGE
        or h1["n"]
        < MIN_HALF_TRADES
        or h2["n"]
        < MIN_HALF_TRADES
        or not np.isfinite(
            m["pf"]
        )
        or m["mean"]
        <= 0
        or m["pf"]
        <= 1.0
        or h1["mean"]
        <= 0
        or h2["mean"]
        <= 0
    ):
        return -np.inf

    return float(
        6.0 * m["win"]
        + 2.0 * coverage
        + 0.40 * np.log(
            max(
                m["pf"],
                1e-9,
            )
        )
        + 0.05 * m["mean"]
        + 0.50 * min(
            h1["win"],
            h2["win"],
        )
        - 0.0005 * m["dd"]
    )


def main():
    started = time.time()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.7.0B "
        "CAUSAL DAILY TECHNICAL-RANKER AUDIT"
    )

    print(
        "=" * 130
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        arrays,
        split,
        groups,
        names,
        mean,
        std,
    ) = brain.load_data()

    model = (
        brain.MultiSurfaceTechnicalBrain(
            groups
        )
        .to(
            device
        )
    )

    ckpt = torch.load(
        execmod.CKPT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        ckpt[
            "model"
        ]
    )

    meta = pd.read_parquet(
        brain.TARGET_FILE,
        columns=[
            "year",
        ],
    )

    year = meta[
        "year"
    ].to_numpy(
        np.int16
    )

    val = split[
        "val"
    ]

    rows2023 = val[
        year[val] == 2023
    ]

    rows2024 = val[
        year[val] == 2024
    ]

    rows2025 = split[
        "test2025"
    ]

    print(
        "Device:",
        device,
    )

    print(
        "Frozen Technical Brain epoch:",
        ckpt[
            "epoch"
        ],
    )

    print(
        "2023 reference:",
        f"{len(rows2023):,}",
    )

    print(
        "2024 policy search:",
        f"{len(rows2024):,}",
    )

    print(
        "2025 benchmark:",
        f"{len(rows2025):,}",
    )

    print(
        "2026 untouched by this audit:",
        f"{len(split['reserved2026']):,}",
    )

    def predict_rows(
        rows,
    ):
        loader = brain.make_loader(
            rows,
            False,
            arrays,
            mean,
            std,
        )

        return brain.predict(
            model,
            loader,
            device,
        )

    print()
    print(
        "Predicting 2023 ..."
    )

    pred23 = predict_rows(
        rows2023
    )

    print(
        "Predicting 2024 ..."
    )

    pred24 = predict_rows(
        rows2024
    )

    print(
        "Building 2024 previous-year ranks ..."
    )

    rank24 = ranked_scores(
        pred23,
        pred24,
    )

    target = pd.read_parquet(
        TARGET,
        columns=[
            "horizon_valid_h30",
            "horizon_valid_h60",
            "horizon_valid_h120",
        ],
    )

    valid24 = build_valid_matrix(
        target,
        rows2024,
    )

    print(
        "Loading executable M5 stream ..."
    )

    m5_df = pd.read_parquet(
        execmod.M5_FILE,
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

    m5_df[
        "timestamp"
    ] = pd.to_datetime(
        m5_df[
            "timestamp"
        ],
        utc=True,
    )

    timestamps_ns = (
        m5_df[
            "timestamp"
        ]
        .astype(
            "int64"
        )
        .to_numpy(
            np.int64
        )
    )

    m5 = {
        col:
            m5_df[
                col
            ].to_numpy()
        for col in m5_df.columns
    }

    source24 = arrays[
        "source"
    ][rows2024]

    day24 = trading_day_ny(
        m5_df[
            "timestamp"
        ].to_numpy()[
            source24
        ]
    )

    grid_rows = []

    best = None
    best_sel = -np.inf
    best_trades = None

    print()
    print(
        "=" * 130
    )

    print(
        "2024 CAUSAL POLICY SEARCH"
    )

    print(
        "=" * 130
    )

    for mode in SCORE_MODES:

        score = rank24[
            mode
        ]

        for threshold in (
            ENTRY_THRESHOLDS
        ):

            for max_day in (
                MAX_TRADES_PER_DAY
            ):

                for persistence in (
                    PERSISTENCES
                ):

                    for margin in (
                        SIDE_MARGINS
                    ):

                        (
                            trades,
                            stats,
                            coverage,
                            eligible_days,
                        ) = simulate(
                            score,
                            valid24,
                            source24,
                            timestamps_ns,
                            day24,
                            m5,
                            threshold,
                            max_day,
                            persistence,
                            margin,
                        )

                        m = metrics(
                            trades,
                            SELECTION_COST,
                        )

                        h1, h2 = half_metrics(
                            trades,
                            SELECTION_COST,
                        )

                        sel = selection_score(
                            m,
                            h1,
                            h2,
                            coverage,
                        )

                        target80 = (
                            m["n"] >= 150
                            and coverage >= 0.80
                            and m["win"] >= 0.80
                            and m["pf"] >= 1.50
                            and m["mean"] > 0
                        )

                        row = {
                            "mode":
                                mode,

                            "threshold":
                                threshold,

                            "max_trades_day":
                                max_day,

                            "persistence":
                                persistence,

                            "side_margin":
                                margin,

                            "eligible_days":
                                eligible_days,

                            "coverage":
                                coverage,

                            "trades":
                                m["n"],

                            "trades_per_day":
                                (
                                    m["n"]
                                    / eligible_days
                                    if eligible_days
                                    else np.nan
                                ),

                            "win":
                                m["win"],

                            "tp_rate":
                                m["tp_rate"],

                            "mean":
                                m["mean"],

                            "net":
                                m["net"],

                            "pf":
                                m["pf"],

                            "dd":
                                m["dd"],

                            "h1_n":
                                h1["n"],

                            "h1_win":
                                h1["win"],

                            "h1_mean":
                                h1["mean"],

                            "h1_pf":
                                h1["pf"],

                            "h2_n":
                                h2["n"],

                            "h2_win":
                                h2["win"],

                            "h2_mean":
                                h2["mean"],

                            "h2_pf":
                                h2["pf"],

                            "selection":
                                sel,

                            "target80":
                                target80,

                            "suppressed_open":
                                stats[
                                    "suppressed_open"
                                ],

                            "suppressed_daily_limit":
                                stats[
                                    "suppressed_daily_limit"
                                ],

                            "suppressed_margin":
                                stats[
                                    "suppressed_side_margin"
                                ],
                        }

                        grid_rows.append(
                            row
                        )

                        if (
                            np.isfinite(
                                sel
                            )
                            and sel
                            > best_sel
                        ):
                            best_sel = sel
                            best = row.copy()
                            best_trades = (
                                trades.copy()
                            )

    grid = pd.DataFrame(
        grid_rows
    )

    grid.to_csv(
        OUT
        / "policy_grid_2024_v670b.csv",
        index=False,
    )

    print()
    print(
        "=" * 130
    )

    print(
        "TOP 30 BY COVERAGE + PRECISION"
    )

    print(
        "=" * 130
    )

    diagnostic = grid.sort_values(
        [
            "win",
            "coverage",
            "pf",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    columns = [
        "mode",
        "threshold",
        "max_trades_day",
        "persistence",
        "side_margin",
        "coverage",
        "trades_per_day",
        "trades",
        "win",
        "mean",
        "pf",
        "dd",
        "h1_win",
        "h1_mean",
        "h2_win",
        "h2_mean",
        "target80",
    ]

    print(
        diagnostic[
            columns
        ]
        .head(
            30
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "80% + >=80% DAILY COVERAGE "
        "CANDIDATES:",
        int(
            grid[
                "target80"
            ].sum()
        ),
    )

    # Coverage frontier.
    print()
    print(
        "=" * 130
    )

    print(
        "2024 COVERAGE / PRECISION FRONTIER"
    )

    print(
        "=" * 130
    )

    for minimum_coverage in (
        1.00,
        0.95,
        0.90,
        0.80,
        0.60,
        0.40,
        0.20,
    ):
        z = grid[
            grid[
                "coverage"
            ]
            >= minimum_coverage
        ]

        if len(z) == 0:
            print(
                f"COVER>={minimum_coverage:.0%}: "
                "no policy"
            )
            continue

        z = z.sort_values(
            [
                "win",
                "pf",
                "mean",
            ],
            ascending=False,
        )

        r = z.iloc[
            0
        ]

        print(
            f"COVER>={minimum_coverage:>4.0%} | "
            f"WIN={r['win']:>6.2%} "
            f"PF={r['pf']:>6.3f} "
            f"MEAN={r['mean']:>+7.3f} "
            f"TR/DAY={r['trades_per_day']:>5.2f} "
            f"| {r['mode']} "
            f"T={r['threshold']:.3f} "
            f"K={int(r['max_trades_day'])} "
            f"P={int(r['persistence'])} "
            f"M={r['side_margin']:.2f}"
        )

    if best is None:
        print()
        print(
            "=" * 130
        )

        print(
            "TEN V6.7.0B RESULT:"
        )

        print(
            "NO PROFITABLE >=80% "
            "DAILY-COVERAGE BASELINE "
            "SURVIVED 2024."
        )

        print(
            "2025 NOT OPENED."
        )

        print(
            "NEXT: TRAIN V6.7.1 "
            "EXECUTION-ALIGNED "
            "PRECISION BRAIN."
        )

        print(
            "Elapsed:",
            f"{time.time() - started:.2f}s",
        )

        return

    print()
    print(
        "=" * 130
    )

    print(
        "FROZEN 2024 CAUSAL POLICY"
    )

    print(
        "=" * 130
    )

    print(
        json.dumps(
            {
                k:
                    (
                        v.item()
                        if isinstance(
                            v,
                            np.generic,
                        )
                        else v
                    )
                for k, v
                in best.items()
            },
            indent=2,
        )
    )

    best_trades.to_csv(
        OUT
        / "trades_2024_frozen_v670b.csv",
        index=False,
    )

    # ---------------------------------
    # 2025: previous-year reference.
    # 2025 is NOT used to select policy.
    # ---------------------------------

    print()
    print(
        "Predicting 2025 AFTER "
        "2024 policy freeze ..."
    )

    pred25 = predict_rows(
        rows2025
    )

    print(
        "Building 2025 ranks against "
        "full previous year 2024 ..."
    )

    rank25 = ranked_scores(
        pred24,
        pred25,
    )

    valid25 = build_valid_matrix(
        target,
        rows2025,
    )

    source25 = arrays[
        "source"
    ][rows2025]

    day25 = trading_day_ny(
        m5_df[
            "timestamp"
        ].to_numpy()[
            source25
        ]
    )

    (
        trades25,
        stats25,
        coverage25,
        eligible25,
    ) = simulate(
        rank25[
            best[
                "mode"
            ]
        ],
        valid25,
        source25,
        timestamps_ns,
        day25,
        m5,
        float(
            best[
                "threshold"
            ]
        ),
        int(
            best[
                "max_trades_day"
            ]
        ),
        int(
            best[
                "persistence"
            ]
        ),
        float(
            best[
                "side_margin"
            ]
        ),
    )

    print()
    print(
        "=" * 130
    )

    print(
        "2025 FROZEN CAUSAL DAILY EXECUTION"
    )

    print(
        "=" * 130
    )

    print(
        "Eligible trading days:",
        eligible25,
    )

    print(
        "Daily coverage:",
        f"{coverage25:.2%}",
    )

    for cost in STRESS_COSTS:
        m = metrics(
            trades25,
            cost,
        )

        print(
            f"COST={cost:>3.1f}bps | "
            f"N={m['n']:>4} "
            f"TR/DAY="
            f"{m['n']/eligible25:>5.2f} "
            f"WIN={m['win']:>6.2%} "
            f"TP={m['tp_rate']:>6.2%} "
            f"MEAN={m['mean']:>+7.3f} "
            f"NET={m['net']:>+9.2f} "
            f"PF={m['pf']:>6.3f} "
            f"DD={m['dd']:>8.2f}"
        )

    if len(
        trades25
    ):
        print()
        print(
            "Sides:"
        )

        print(
            trades25[
                "side"
            ].value_counts().to_dict()
        )

        print()
        print(
            "Task usage:"
        )

        print(
            trades25.groupby(
                [
                    "side",
                    "horizon",
                    "tp_bps",
                    "sl_bps",
                ]
            ).size().sort_values(
                ascending=False
            ).head(
                15
            )
        )

    trades25.to_csv(
        OUT
        / "trades_2025_v670b.csv",
        index=False,
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is a causal ranker baseline, "
        "not the V6.7.1 trained precision brain."
    )

    print(
        "2024 selects the policy."
    )

    print(
        "2025 only evaluates the frozen policy."
    )

    print(
        "2026 remains excluded."
    )

    print(
        "Saved:",
        OUT,
    )

    print(
        "Elapsed:",
        f"{time.time() - started:.2f}s",
    )


if __name__ == "__main__":
    main()
