from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

import training.v6.models.train_multisurface_technical_brain_v661 as brain
import training.v6.backtests.backtest_surface_policy_v662 as execmod


OUT = Path(
    "training/artifacts/v6/"
    "task_specific_tail_audit_v662c"
)

CANDIDATES = (
    ("SHORT", 60, 40, 20),
    ("SHORT", 120, 40, 20),
    ("SHORT", 120, 60, 30),

    # Controls
    ("LONG", 60, 40, 20),
    ("LONG", 120, 40, 20),
)

SCORE_MODES = (
    "ptp",
    "race",
    "economic",
)

ENTRY_QS = (
    0.90,
    0.95,
    0.98,
    0.99,
    0.995,
    0.998,
)

PERSISTENCES = (
    1,
    2,
    3,
)

REARM_QS = (
    0.50,
    0.80,
    0.90,
)

SELECTION_COST = 0.50

STRESS_COSTS = (
    0.0,
    0.5,
    1.0,
    2.0,
)

MIN_TRADES = 20
MIN_HALF_TRADES = 5

TARGET_WIN = 0.80
TARGET_PF = 1.20

STEP_NS = 300_000_000_000


def task_id_for(
    side,
    horizon,
    tp,
    sl,
):
    side_id = (
        0
        if side == "LONG"
        else 1
    )

    return execmod.TASK_LOOKUP[
        (
            side_id,
            horizon,
            tp,
            sl,
        )
    ]


def score_task(
    prob,
    task_id,
    mode,
):
    p = prob[
        :,
        task_id
    ]

    p_tp = p[
        :,
        0
    ]

    p_sl = p[
        :,
        1
    ]

    if mode == "ptp":
        return p_tp.copy()

    if mode == "race":
        return (
            p_tp
            / np.maximum(
                p_tp + p_sl,
                1e-12,
            )
        )

    if mode == "economic":
        meta = brain.TASKS[
            task_id
        ]

        tp = float(
            meta["tp"]
        )

        sl = float(
            meta["sl"]
        )

        # Timeout intentionally contributes
        # zero to this ranking proxy.
        return (
            p_tp * tp
            - p_sl * sl
        )

    raise ValueError(
        mode
    )


def persistent_signal(
    score,
    source,
    timestamps_ns,
    threshold,
    persistence,
):
    raw = (
        score >= threshold
    )

    if persistence <= 1:
        return raw

    out = np.zeros(
        len(score),
        dtype=bool,
    )

    for i in range(
        persistence - 1,
        len(score),
    ):
        if not raw[i]:
            continue

        ok = True

        for lag in range(
            1,
            persistence
        ):
            j = i - lag

            if not raw[j]:
                ok = False
                break

            if (
                source[j + 1]
                != source[j] + 1
            ):
                ok = False
                break

            if (
                timestamps_ns[j + 1]
                - timestamps_ns[j]
                != STEP_NS
            ):
                ok = False
                break

        if ok:
            out[i] = True

    return out


def simulate_fixed_task(
    signal,
    score,
    source,
    m5,
    timestamps_ns,
    task_id,
    rearm_threshold,
):
    meta = brain.TASKS[
        task_id
    ]

    side_id = int(
        meta["side_id"]
    )

    trades = []

    occupied_until = -1
    blocked = False

    stats = {
        "signals":
            int(
                signal.sum()
            ),

        "suppressed_open":
            0,

        "suppressed_rearm":
            0,

        "invalid_execution":
            0,
    }

    for i in range(
        len(source)
    ):
        s = int(
            source[i]
        )

        if (
            blocked
            and s > occupied_until
            and score[i] < rearm_threshold
        ):
            blocked = False

        if not signal[i]:
            continue

        if s <= occupied_until:
            stats[
                "suppressed_open"
            ] += 1
            continue

        if blocked:
            stats[
                "suppressed_rearm"
            ] += 1
            continue

        result = execmod.execute_trade(
            m5,
            timestamps_ns,
            s,
            side_id,
            task_id,
        )

        if result is None:
            stats[
                "invalid_execution"
            ] += 1
            continue

        entry_row = s + 1

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

                "side":
                    meta[
                        "side"
                    ].upper(),

                "horizon":
                    int(
                        meta[
                            "horizon"
                        ]
                    ),

                "tp_bps":
                    float(
                        meta[
                            "tp"
                        ]
                    ),

                "sl_bps":
                    float(
                        meta[
                            "sl"
                        ]
                    ),

                "score":
                    float(
                        score[i]
                    ),

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

        occupied_until = (
            int(
                result[
                    "exit_row"
                ]
            )
            + 1
        )

        blocked = True

    return (
        pd.DataFrame(
            trades
        ),
        stats,
    )


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
            "tp_rate": np.nan,
            "resolved_tp": np.nan,
            "dd": np.nan,
        }

    pnl = (
        trades[
            "gross_bps"
        ].to_numpy(
            np.float64
        )
        - cost
    )

    positive = pnl > 0

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
            equity
        ]
    )[1:]

    dd = float(
        -np.min(
            equity - peak
        )
    )

    outcome = trades[
        "outcome"
    ].astype(str)

    tp_mask = (
        outcome == "TP"
    )

    sl_mask = (
        outcome.str.startswith(
            "SL"
        )
    )

    resolved = (
        tp_mask.sum()
        + sl_mask.sum()
    )

    resolved_tp = (
        float(
            tp_mask.sum()
            / resolved
        )
        if resolved > 0
        else np.nan
    )

    return {
        "n":
            len(trades),

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

        # Real net-positive trade rate.
        "win":
            float(
                positive.mean()
            ),

        "tp_rate":
            float(
                tp_mask.mean()
            ),

        "resolved_tp":
            resolved_tp,

        "dd":
            dd,
    }


def half_metrics(
    trades,
    cost,
):
    if len(trades) == 0:
        return (
            metrics(
                trades,
                cost,
            ),
            metrics(
                trades,
                cost,
            ),
        )

    month = pd.to_datetime(
        trades[
            "entry_timestamp"
        ],
        utc=True,
    ).dt.month

    return (
        metrics(
            trades[
                month <= 6
            ],
            cost,
        ),

        metrics(
            trades[
                month >= 7
            ],
            cost,
        ),
    )


def robust_survivor(
    m,
    h1,
    h2,
):
    return (
        m["n"] >= MIN_TRADES

        and h1["n"] >= MIN_HALF_TRADES
        and h2["n"] >= MIN_HALF_TRADES

        and m["mean"] > 0
        and m["pf"] > 1.0

        and h1["mean"] > 0
        and h2["mean"] > 0
    )


def selection_score(
    m,
    h1,
    h2,
):
    if not robust_survivor(
        m,
        h1,
        h2,
    ):
        return -np.inf

    # We explicitly reward precision,
    # but only AFTER profitability
    # and temporal robustness survive.
    return float(
        5.0 * m["win"]
        + 0.50 * np.log(
            max(
                m["pf"],
                1e-6,
            )
        )
        + 0.10 * m["mean"]
        + 0.50 * min(
            h1["win"],
            h2["win"],
        )
        - 0.001 * m["dd"]
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
        "TEN V6.6.2c "
        "TASK-SPECIFIC STATEFUL TAIL AUDIT"
    )

    print(
        "=" * 126
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
        execmod.CKPT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model"
        ]
    )

    print(
        "Device:",
        device,
    )

    print(
        "Frozen Brain epoch:",
        checkpoint[
            "epoch"
        ],
    )

    meta = pd.read_parquet(
        brain.TARGET_FILE,
        columns=[
            "year",
            "source_row",
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
        "2023 reference:",
        f"{len(rows2023):,}",
    )

    print(
        "2024 policy selection:",
        f"{len(rows2024):,}",
    )

    print(
        "2025 held from this selection:",
        f"{len(rows2025):,}",
    )

    print(
        "2026 RESERVED:",
        f"{len(split['reserved2026']):,}",
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
        "Loading executable M5 ..."
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

    source2023 = arrays[
        "source"
    ][rows2023]

    source2024 = arrays[
        "source"
    ][rows2024]

    ts2024 = timestamps_ns[
        source2024
    ]

    grid_rows = []

    best = None
    best_score = -np.inf

    print()
    print(
        "=" * 126
    )

    print(
        "2024 TASK-SPECIFIC SEARCH"
    )

    print(
        "=" * 126
    )

    for (
        side,
        horizon,
        tp,
        sl,
    ) in CANDIDATES:

        task_id = task_id_for(
            side,
            horizon,
            tp,
            sl,
        )

        print()
        print(
            f"{side} "
            f"H{horizon} "
            f"TP{tp}/SL{sl}"
        )

        print(
            "-" * 100
        )

        for mode in SCORE_MODES:

            s23 = score_task(
                pred2023[
                    "race_prob"
                ],
                task_id,
                mode,
            )

            s24 = score_task(
                pred2024[
                    "race_prob"
                ],
                task_id,
                mode,
            )

            for entry_q in ENTRY_QS:

                entry_threshold = float(
                    np.quantile(
                        s23,
                        entry_q,
                    )
                )

                for rearm_q in REARM_QS:

                    if rearm_q >= entry_q:
                        continue

                    rearm_threshold = float(
                        np.quantile(
                            s23,
                            rearm_q,
                        )
                    )

                    for persistence in PERSISTENCES:

                        signal = persistent_signal(
                            s24,
                            source2024,
                            ts2024,
                            entry_threshold,
                            persistence,
                        )

                        trades, stats = (
                            simulate_fixed_task(
                                signal,
                                s24,
                                source2024,
                                m5,
                                timestamps_ns,
                                task_id,
                                rearm_threshold,
                            )
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
                        )

                        target80 = (
                            m["n"] >= MIN_TRADES
                            and m["win"] >= TARGET_WIN
                            and m["pf"] >= TARGET_PF
                        )

                        row = {
                            "side":
                                side,

                            "horizon":
                                horizon,

                            "tp":
                                tp,

                            "sl":
                                sl,

                            "task_id":
                                task_id,

                            "score_mode":
                                mode,

                            "entry_q":
                                entry_q,

                            "rearm_q":
                                rearm_q,

                            "persistence":
                                persistence,

                            "entry_threshold":
                                entry_threshold,

                            "rearm_threshold":
                                rearm_threshold,

                            "selection":
                                sel,

                            "target80":
                                target80,

                            "trades":
                                m["n"],

                            "win":
                                m["win"],

                            "tp_rate":
                                m["tp_rate"],

                            "resolved_tp":
                                m["resolved_tp"],

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

                            "signals":
                                stats["signals"],

                            "suppressed_open":
                                stats[
                                    "suppressed_open"
                                ],

                            "suppressed_rearm":
                                stats[
                                    "suppressed_rearm"
                                ],
                        }

                        grid_rows.append(
                            row
                        )

                        if np.isfinite(
                            sel
                        ) and sel > best_score:
                            best_score = sel
                            best = row.copy()

            # Best diagnostic row for
            # this task/score printed later.

    grid = pd.DataFrame(
        grid_rows
    )

    grid.to_csv(
        OUT
        / "task_specific_grid_2024_v662c.csv",
        index=False,
    )

    valid = grid[
        np.isfinite(
            grid[
                "selection"
            ]
        )
    ].sort_values(
        "selection",
        ascending=False,
    )

    print()
    print(
        "=" * 126
    )

    print(
        "TOP ROBUST 2024 POLICIES"
    )

    print(
        "=" * 126
    )

    cols = [
        "side",
        "horizon",
        "tp",
        "sl",
        "score_mode",
        "entry_q",
        "rearm_q",
        "persistence",
        "trades",
        "win",
        "tp_rate",
        "resolved_tp",
        "mean",
        "pf",
        "dd",
        "h1_mean",
        "h2_mean",
        "target80",
    ]

    if len(valid):
        print(
            valid[
                cols
            ]
            .head(
                25
            )
            .to_string(
                index=False
            )
        )

    else:
        print(
            "NO ROBUST POSITIVE POLICY "
            "SURVIVED 2024."
        )

    target80 = valid[
        valid[
            "target80"
        ]
    ]

    print()
    print(
        "80% REAL-WIN CANDIDATES:",
        len(
            target80
        ),
    )

    if len(target80):
        print(
            target80[
                cols
            ]
            .head(
                20
            )
            .to_string(
                index=False
            )
        )

    if best is None:
        print()
        print(
            "=" * 126
        )

        print(
            "TEN V6.6.2c RESULT:"
        )

        print(
            "NO POSITIVE TASK-SPECIFIC "
            "ENTRY POLICY SURVIVED 2024."
        )

        print(
            "2025 IS NOT EVALUATED "
            "BY THIS AUDIT."
        )

        print(
            "NEXT STEP: "
            "SELECTIVE PRECISION GATE."
        )

        return

    frozen = {
        key:
            (
                value.item()
                if isinstance(
                    value,
                    np.generic,
                )
                else value
            )
        for key, value
        in best.items()
    }

    with open(
        OUT
        / "frozen_task_policy_v662c.json",
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
        "FROZEN TASK-SPECIFIC POLICY"
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

    # Only now is 2025 opened.
    print()
    print(
        "Predicting 2025 after "
        "policy freeze ..."
    )

    pred2025 = predict_rows(
        rows2025
    )

    source2025 = arrays[
        "source"
    ][rows2025]

    ts2025 = timestamps_ns[
        source2025
    ]

    task_id = int(
        best[
            "task_id"
        ]
    )

    mode = best[
        "score_mode"
    ]

    s25 = score_task(
        pred2025[
            "race_prob"
        ],
        task_id,
        mode,
    )

    # Important:
    # thresholds remain fitted on 2023.
    signal25 = persistent_signal(
        s25,
        source2025,
        ts2025,
        float(
            best[
                "entry_threshold"
            ]
        ),
        int(
            best[
                "persistence"
            ]
        ),
    )

    trades25, stats25 = (
        simulate_fixed_task(
            signal25,
            s25,
            source2025,
            m5,
            timestamps_ns,
            task_id,
            float(
                best[
                    "rearm_threshold"
                ]
            ),
        )
    )

    print()
    print(
        "=" * 126
    )

    print(
        "2025 FROZEN TASK-SPECIFIC "
        "STATEFUL EXECUTION"
    )

    print(
        "=" * 126
    )

    print(
        "Signals:",
        stats25[
            "signals"
        ],
    )

    print(
        "Suppressed OPEN:",
        stats25[
            "suppressed_open"
        ],
    )

    print(
        "Suppressed REARM:",
        stats25[
            "suppressed_rearm"
        ],
    )

    print(
        "Invalid:",
        stats25[
            "invalid_execution"
        ],
    )

    for cost in STRESS_COSTS:
        m = metrics(
            trades25,
            cost,
        )

        print(
            f"COST={cost:>3.1f}bps | "
            f"N={m['n']:>4} "
            f"WIN={m['win']:>6.2%} "
            f"TP={m['tp_rate']:>6.2%} "
            f"TP|RES={m['resolved_tp']:>6.2%} "
            f"MEAN={m['mean']:>+7.3f} "
            f"NET={m['net']:>+9.2f} "
            f"PF={m['pf']:>6.3f} "
            f"DD={m['dd']:>7.2f}"
        )

    trades25.to_csv(
        OUT
        / "trades_2025_v662c.csv",
        index=False,
    )

    print()
    print(
        "2026 RESERVED: NOT EVALUATED."
    )

    print(
        "Saved:",
        OUT,
    )


if __name__ == "__main__":
    main()
