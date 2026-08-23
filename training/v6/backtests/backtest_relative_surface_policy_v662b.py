from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

import training.v6.models.train_multisurface_technical_brain_v661 as brain
import training.v6.backtests.backtest_surface_policy_v662 as old


OUT = Path(
    "training/artifacts/v6/"
    "relative_surface_execution_v662b"
)

ENTRY_QS = (
    0.950,
    0.980,
    0.990,
    0.995,
)

MIN_AGREEMENTS = (
    2,
    3,
    4,
)

PERSISTENCES = (
    1,
    2,
)

SIDE_MARGINS = (
    0.00,
    0.02,
    0.05,
)

# Individual surface needs to be
# at least unusually strong,
# not necessarily absolute-positive.
TASK_AGREE_Q = 0.90
BEST_TASK_Q = 0.90

# Re-arm only after the previous
# directional cluster has cooled.
RESET_Q = 0.80

SELECTION_EXTRA_COST_BPS = 0.50


def percentile(
    reference_sorted,
    values,
):
    return (
        np.searchsorted(
            reference_sorted,
            values,
            side="right",
        )
        / len(
            reference_sorted
        )
    )


def fit_task_rank_reference(
    raw_state,
):
    score = raw_state[
        "task_score"
    ]

    return [
        np.sort(
            score[
                :,
                task_id
            ].astype(
                np.float64
            )
        )
        for task_id in range(
            brain.N_TASKS
        )
    ]


def rank_state(
    raw_state,
    references,
):
    raw_task = raw_state[
        "task_score"
    ]

    n = raw_task.shape[
        0
    ]

    rank = np.zeros_like(
        raw_task,
        dtype=np.float64,
    )

    for task_id in range(
        brain.N_TASKS
    ):
        rank[
            :,
            task_id
        ] = percentile(
            references[
                task_id
            ],
            raw_task[
                :,
                task_id
            ],
        )

    side_score = np.zeros(
        (
            n,
            2,
        ),
        dtype=np.float64,
    )

    agreement = np.zeros(
        (
            n,
            2,
        ),
        dtype=np.int16,
    )

    best_task = np.full(
        (
            n,
            2,
        ),
        -1,
        dtype=np.int16,
    )

    best_task_rank = np.zeros(
        (
            n,
            2,
        ),
        dtype=np.float64,
    )

    best_task_raw = np.full(
        (
            n,
            2,
        ),
        -np.inf,
        dtype=np.float64,
    )

    for side_id in range(
        2
    ):
        all_ids = [
            task_id
            for task_id, meta
            in enumerate(
                brain.TASKS
            )
            if meta[
                "side_id"
            ]
            == side_id
        ]

        r = rank[
            :,
            all_ids
        ]

        sorted_r = np.sort(
            r,
            axis=1,
        )

        top3 = sorted_r[
            :,
            -3:
        ].mean(
            axis=1
        )

        median = np.median(
            r,
            axis=1,
        )

        agree = (
            r
            >= TASK_AGREE_Q
        ).sum(
            axis=1
        )

        agreement[
            :,
            side_id
        ] = agree

        breadth = (
            agree
            / len(
                all_ids
            )
        )

        # Consensus is now relative:
        # top surfaces + broad support.
        side_score[
            :,
            side_id
        ] = (
            0.55
            * top3

            + 0.25
            * median

            + 0.20
            * breadth
        )

        trade_ids = [
            old.TASK_LOOKUP[
                (
                    side_id,
                    horizon,
                    tp,
                    sl,
                )
            ]
            for (
                horizon,
                tp,
                sl,
            ) in old.TRADEABLE
        ]

        trade_rank = rank[
            :,
            trade_ids
        ]

        trade_raw = raw_task[
            :,
            trade_ids
        ]

        # Primary criterion:
        # historical rarity/rank.
        #
        # Tiny raw-score tie-break
        # preserves economic ordering.
        choice_score = (
            trade_rank
            + 1e-4
            * np.tanh(
                trade_raw
            )
        )

        local = np.argmax(
            choice_score,
            axis=1,
        )

        trade_ids_np = np.asarray(
            trade_ids,
            dtype=np.int16,
        )

        selected = trade_ids_np[
            local
        ]

        best_task[
            :,
            side_id
        ] = selected

        best_task_rank[
            :,
            side_id
        ] = trade_rank[
            np.arange(
                n
            ),
            local,
        ]

        best_task_raw[
            :,
            side_id
        ] = trade_raw[
            np.arange(
                n
            ),
            local,
        ]

    return {
        "task_score":
            raw_task,

        "task_rank":
            rank,

        "side_score":
            side_score,

        "agreement":
            agreement,

        "best_task":
            best_task,

        "best_task_rank":
            best_task_rank,

        # old.simulate expects this key.
        "best_task_score":
            best_task_raw,
    }


def build_signals(
    source,
    timestamps_ns,
    state,
    thresholds,
    min_agreement,
    persistence,
    side_margin,
):
    n = len(
        source
    )

    raw_signal = np.full(
        n,
        -1,
        dtype=np.int8,
    )

    score = state[
        "side_score"
    ]

    agreement = state[
        "agreement"
    ]

    best_rank = state[
        "best_task_rank"
    ]

    for i in range(
        n
    ):
        qualified = []

        for side in range(
            2
        ):
            opposite = (
                1 - side
            )

            margin = (
                score[
                    i,
                    side
                ]
                - score[
                    i,
                    opposite
                ]
            )

            if (
                score[
                    i,
                    side
                ]
                >= thresholds[
                    side
                ]

                and agreement[
                    i,
                    side
                ]
                >= min_agreement

                and best_rank[
                    i,
                    side
                ]
                >= BEST_TASK_Q

                and margin
                >= side_margin
            ):
                qualified.append(
                    side
                )

        if len(
            qualified
        ) == 1:
            raw_signal[
                i
            ] = qualified[
                0
            ]

        elif len(
            qualified
        ) == 2:
            raw_signal[
                i
            ] = int(
                np.argmax(
                    score[
                        i
                    ]
                )
            )

    if persistence <= 1:
        return raw_signal

    signal = np.full_like(
        raw_signal,
        -1,
    )

    for i in range(
        persistence - 1,
        n,
    ):
        side = raw_signal[
            i
        ]

        if side < 0:
            continue

        stable = True

        for lag in range(
            1,
            persistence
        ):
            j = (
                i - lag
            )

            if (
                raw_signal[
                    j
                ]
                != side
            ):
                stable = False
                break

            if (
                source[
                    j + 1
                ]
                != source[
                    j
                ]
                + 1
            ):
                stable = False
                break

            if (
                timestamps_ns[
                    j + 1
                ]
                - timestamps_ns[
                    j
                ]
                != old.STEP_NS
            ):
                stable = False
                break

        if stable:
            signal[
                i
            ] = side

    return signal


def print_state_diagnostics(
    name,
    state,
):
    print()
    print(
        name
    )

    print(
        "-" * 120
    )

    for side_id, side in enumerate(
        (
            "LONG",
            "SHORT",
        )
    ):
        raw = state[
            "task_score"
        ][
            :,
            [
                i
                for i, m
                in enumerate(
                    brain.TASKS
                )
                if m[
                    "side_id"
                ]
                == side_id
            ]
        ]

        score = state[
            "side_score"
        ][
            :,
            side_id
        ]

        agree = state[
            "agreement"
        ][
            :,
            side_id
        ]

        best_rank = state[
            "best_task_rank"
        ][
            :,
            side_id
        ]

        print(
            side,
            "| raw task score "
            f"max median="
            f"{np.median(np.max(raw, axis=1)):+.4f}"
        )

        print(
            "   side score quantiles:",
            {
                q:
                    round(
                        float(
                            np.quantile(
                                score,
                                q,
                            )
                        ),
                        5,
                    )
                for q in (
                    0.50,
                    0.90,
                    0.95,
                    0.98,
                    0.99,
                    0.995,
                )
            },
        )

        print(
            "   agreement >=2/3/4:",
            f"{(agree >= 2).mean():.2%}",
            f"{(agree >= 3).mean():.2%}",
            f"{(agree >= 4).mean():.2%}",
        )

        print(
            "   best tradeable rank >=.90/.95/.99:",
            f"{(best_rank >= .90).mean():.2%}",
            f"{(best_rank >= .95).mean():.2%}",
            f"{(best_rank >= .99).mean():.2%}",
        )


def quarterly_report(
    title,
    trades,
    cost,
):
    print()
    print(
        title,
        f"| EXTRA COST={cost:.1f}bps"
    )

    print(
        "-" * 120
    )

    if not len(
        trades
    ):
        print(
            "NO TRADES"
        )
        return

    ts = pd.to_datetime(
        trades[
            "entry_timestamp"
        ],
        utc=True,
    )

    quarter = (
        (
            ts.dt.month
            - 1
        )
        // 3
        + 1
    )

    for q in range(
        1,
        5,
    ):
        x = trades[
            quarter == q
        ]

        m = old.trade_metrics(
            x,
            cost,
        )

        print(
            f"Q{q} "
            f"n={m['n']:>4} "
            f"mean={m['mean']:>+7.3f} "
            f"net={m['sum']:>+8.2f} "
            f"PF={m['pf']:>6.3f} "
            f"WIN={m['win']:>6.2%} "
            f"DD={m['max_dd']:>7.2f}"
        )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "TEN V6.6.2b "
        "RELATIVE SURFACE CONSENSUS "
        "+ STATEFUL EXECUTION"
    )

    print(
        "=" * 126
    )

    print(
        "Device:",
        device,
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

    checkpoint = torch.load(
        old.CKPT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model"
        ]
    )

    print(
        "Frozen Technical Brain epoch:",
        checkpoint[
            "epoch"
        ],
    )

    target_meta = pd.read_parquet(
        brain.TARGET_FILE,
        columns=[
            "year",
            "source_row",
        ],
    )

    year = target_meta[
        "year"
    ].to_numpy(
        np.int16
    )

    val_rows = split[
        "val"
    ]

    rows2023 = val_rows[
        year[
            val_rows
        ]
        == 2023
    ]

    rows2024 = val_rows[
        year[
            val_rows
        ]
        == 2024
    ]

    rows2025 = split[
        "test2025"
    ]

    print(
        "2023 reference/calibration:",
        f"{len(rows2023):,}",
    )

    print(
        "2024 policy selection:",
        f"{len(rows2024):,}",
    )

    print(
        "2025 OOT execution benchmark:",
        f"{len(rows2025):,}",
    )

    print(
        "2026 RESERVED:",
        f"{len(split['reserved2026']):,}",
        "(NOT EVALUATED)",
    )

    def predict_rows(
        rows,
    ):
        dl = brain.make_loader(
            rows,
            False,
            arrays,
            mean,
            std,
        )

        return brain.predict(
            model,
            dl,
            device,
        )

    print()
    print(
        "Predicting 2023 ..."
    )

    pred2023 = predict_rows(
        rows2023
    )

    print(
        "Predicting 2024 ..."
    )

    pred2024 = predict_rows(
        rows2024
    )

    print(
        "Predicting 2025 ..."
    )

    pred2025 = predict_rows(
        rows2025
    )

    print(
        "Fitting probability "
        "calibrators on 2023 ..."
    )

    calibrators = (
        old.fit_calibrators(
            pred2023
        )
    )

    with open(
        OUT
        / "calibrators_v662b.json",
        "w",
    ) as f:
        json.dump(
            calibrators,
            f,
            indent=2,
        )

    cal2023 = old.calibrate(
        pred2023,
        calibrators,
    )

    cal2024 = old.calibrate(
        pred2024,
        calibrators,
    )

    cal2025 = old.calibrate(
        pred2025,
        calibrators,
    )

    raw2023 = old.surface_state(
        cal2023
    )

    raw2024 = old.surface_state(
        cal2024
    )

    raw2025 = old.surface_state(
        cal2025
    )

    references = (
        fit_task_rank_reference(
            raw2023
        )
    )

    state2023 = rank_state(
        raw2023,
        references,
    )

    state2024 = rank_state(
        raw2024,
        references,
    )

    state2025 = rank_state(
        raw2025,
        references,
    )

    print_state_diagnostics(
        "2023 REFERENCE STATE",
        state2023,
    )

    print_state_diagnostics(
        "2024 POLICY STATE",
        state2024,
    )

    print_state_diagnostics(
        "2025 OOT STATE",
        state2025,
    )

    print()
    print(
        "Loading executable M5 stream ..."
    )

    m5_df = pd.read_parquet(
        old.M5_FILE,
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

    m5 = {
        col:
            m5_df[
                col
            ].to_numpy()
        for col in m5_df.columns
    }

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

    source2024 = arrays[
        "source"
    ][
        rows2024
    ]

    source2025 = arrays[
        "source"
    ][
        rows2025
    ]

    ts2024 = timestamps_ns[
        source2024
    ]

    ts2025 = timestamps_ns[
        source2025
    ]

    reset_thresholds = [
        float(
            np.quantile(
                state2023[
                    "side_score"
                ][
                    :,
                    side
                ],
                RESET_Q,
            )
        )
        for side in range(
            2
        )
    ]

    print()
    print(
        "Re-arm thresholds:",
        reset_thresholds,
    )

    grid_rows = []

    best = None
    best_selection = -np.inf

    print()
    print(
        "=" * 126
    )

    print(
        "2024 POLICY SEARCH"
    )

    print(
        "=" * 126
    )

    for entry_q in ENTRY_QS:
        thresholds = [
            float(
                np.quantile(
                    state2023[
                        "side_score"
                    ][
                        :,
                        side
                    ],
                    entry_q,
                )
            )
            for side in range(
                2
            )
        ]

        for min_agreement in (
            MIN_AGREEMENTS
        ):
            for persistence in (
                PERSISTENCES
            ):
                for margin in (
                    SIDE_MARGINS
                ):
                    signal = build_signals(
                        source2024,
                        ts2024,
                        state2024,
                        thresholds,
                        min_agreement,
                        persistence,
                        margin,
                    )

                    trades, stats = (
                        old.simulate(
                            rows2024,
                            source2024,
                            signal,
                            state2024,
                            m5,
                            timestamps_ns,
                            reset_thresholds,
                        )
                    )

                    metrics = (
                        old.trade_metrics(
                            trades,
                            SELECTION_EXTRA_COST_BPS,
                        )
                    )

                    h1, h2 = (
                        old.half_metrics(
                            trades,
                            SELECTION_EXTRA_COST_BPS,
                        )
                    )

                    selection = (
                        old.selection_score(
                            metrics,
                            h1,
                            h2,
                        )
                    )

                    grid_rows.append(
                        {
                            "entry_q":
                                entry_q,

                            "min_agreement":
                                min_agreement,

                            "persistence":
                                persistence,

                            "side_margin":
                                margin,

                            "long_threshold":
                                thresholds[
                                    0
                                ],

                            "short_threshold":
                                thresholds[
                                    1
                                ],

                            "selection":
                                selection,

                            "trades":
                                metrics[
                                    "n"
                                ],

                            "mean_bps":
                                metrics[
                                    "mean"
                                ],

                            "net_bps":
                                metrics[
                                    "sum"
                                ],

                            "pf":
                                metrics[
                                    "pf"
                                ],

                            "win":
                                metrics[
                                    "win"
                                ],

                            "max_dd":
                                metrics[
                                    "max_dd"
                                ],

                            "h1_n":
                                h1[
                                    "n"
                                ],

                            "h1_mean":
                                h1[
                                    "mean"
                                ],

                            "h1_pf":
                                h1[
                                    "pf"
                                ],

                            "h2_n":
                                h2[
                                    "n"
                                ],

                            "h2_mean":
                                h2[
                                    "mean"
                                ],

                            "h2_pf":
                                h2[
                                    "pf"
                                ],

                            "persistent_signals":
                                stats[
                                    "persistent_signals"
                                ],

                            "suppressed_open":
                                stats[
                                    "suppressed_open"
                                ],

                            "suppressed_rearm":
                                stats[
                                    "suppressed_rearm"
                                ],
                        }
                    )

                    print(
                        f"Q={entry_q:.3f} "
                        f"A={min_agreement} "
                        f"P={persistence} "
                        f"M={margin:.2f} | "
                        f"trades={metrics['n']:>4} "
                        f"mean={metrics['mean']:>+7.3f} "
                        f"PF={metrics['pf']:>6.3f} "
                        f"DD={metrics['max_dd']:>7.2f} "
                        f"sel={selection:>+8.4f}"
                    )

                    if (
                        selection
                        > best_selection
                    ):
                        best_selection = (
                            selection
                        )

                        best = {
                            "entry_q":
                                entry_q,

                            "min_agreement":
                                min_agreement,

                            "persistence":
                                persistence,

                            "side_margin":
                                margin,

                            "thresholds":
                                thresholds,
                        }

    grid = pd.DataFrame(
        grid_rows
    ).sort_values(
        "selection",
        ascending=False,
    )

    grid.to_csv(
        OUT
        / "policy_grid_2024_v662b.csv",
        index=False,
    )

    print()
    print(
        "TOP 15 2024 POLICIES"
    )

    print(
        "-" * 126
    )

    print(
        grid.head(
            15
        ).to_string(
            index=False
        )
    )

    if (
        best is None
        or not np.isfinite(
            best_selection
        )
    ):
        raise RuntimeError(
            "No eligible relative-surface "
            "policy survived."
        )

    frozen = {
        **best,

        "selection_score":
            float(
                best_selection
            ),

        "task_agree_q":
            TASK_AGREE_Q,

        "best_task_q":
            BEST_TASK_Q,

        "reset_q":
            RESET_Q,

        "reset_thresholds":
            reset_thresholds,

        "selection_extra_cost_bps":
            SELECTION_EXTRA_COST_BPS,

        "cooldown_bars":
            old.COOLDOWN_BARS,
    }

    with open(
        OUT
        / "frozen_policy_v662b.json",
        "w",
    ) as f:
        json.dump(
            frozen,
            f,
            indent=2,
        )

    print()
    print(
        "=" * 126
    )

    print(
        "FROZEN V6.6.2b POLICY"
    )

    print(
        "=" * 126
    )

    print(
        json.dumps(
            frozen,
            indent=2,
        )
    )

    signal2024 = build_signals(
        source2024,
        ts2024,
        state2024,
        best[
            "thresholds"
        ],
        best[
            "min_agreement"
        ],
        best[
            "persistence"
        ],
        best[
            "side_margin"
        ],
    )

    trades2024, stats2024 = (
        old.simulate(
            rows2024,
            source2024,
            signal2024,
            state2024,
            m5,
            timestamps_ns,
            reset_thresholds,
        )
    )

    old.print_metrics(
        "2024 FROZEN POLICY BACKTEST",
        trades2024,
        stats2024,
    )

    trades2024.to_csv(
        OUT
        / "trades_2024_v662b.csv",
        index=False,
    )

    print()
    print(
        "=" * 126
    )

    print(
        "2025 FROZEN OOT EXECUTION"
    )

    print(
        "=" * 126
    )

    signal2025 = build_signals(
        source2025,
        ts2025,
        state2025,
        best[
            "thresholds"
        ],
        best[
            "min_agreement"
        ],
        best[
            "persistence"
        ],
        best[
            "side_margin"
        ],
    )

    trades2025, stats2025 = (
        old.simulate(
            rows2025,
            source2025,
            signal2025,
            state2025,
            m5,
            timestamps_ns,
            reset_thresholds,
        )
    )

    old.print_metrics(
        "2025 V6.6.2b "
        "STATEFUL EXECUTION",
        trades2025,
        stats2025,
    )

    quarterly_report(
        "2025 QUARTERLY ROBUSTNESS",
        trades2025,
        0.5,
    )

    quarterly_report(
        "2025 QUARTERLY ROBUSTNESS",
        trades2025,
        1.0,
    )

    trades2025.to_csv(
        OUT
        / "trades_2025_v662b.csv",
        index=False,
    )

    print()
    print(
        "Spread: embedded in BID/ASK."
    )

    print(
        "Entry: NEXT M5 OPEN."
    )

    print(
        "One position maximum."
    )

    print(
        "Persistent duplicate signals "
        "are suppressed."
    )

    print(
        "Same-side re-entry requires "
        "hysteresis reset."
    )

    print(
        "TP+SL same bar => SL."
    )

    print(
        "Stop gaps use actual "
        "next executable price."
    )

    print(
        "2025 was NOT used in "
        "V6.6.2b policy search."
    )

    print(
        "2026 RESERVED: NOT EVALUATED."
    )

    print(
        "Saved:",
        OUT,
    )


if __name__ == "__main__":
    main()
