from pathlib import Path

import numpy as np
import pandas as pd
import torch

import training.v6.models.train_multisurface_technical_brain_v661 as brain
import training.v6.backtests.backtest_surface_policy_v662 as execmod
import training.v6.models.train_execution_precision_brain_v671 as v671
import training.v6.models.train_daily_opportunity_brain_v672 as v672


OUT = Path(
    "training/artifacts/v6/"
    "when_what_decomposition_v672a"
)

COST = 0.5
MIN_DAY_BARS = 100


def profit_factor(
    pnl,
):
    pnl = np.asarray(
        pnl,
        np.float64,
    )

    gains = pnl[
        pnl > 0
    ].sum()

    losses = -pnl[
        pnl < 0
    ].sum()

    if losses <= 0:
        return np.inf

    return float(
        gains / losses
    )


def metrics(
    pnl,
):
    pnl = np.asarray(
        pnl,
        np.float64,
    )

    if len(pnl) == 0:
        return {
            "n": 0,
            "win": np.nan,
            "mean": np.nan,
            "pf": np.nan,
            "net": 0.0,
        }

    return {
        "n":
            len(pnl),

        "win":
            float(
                (
                    pnl > 0
                ).mean()
            ),

        "mean":
            float(
                pnl.mean()
            ),

        "pf":
            profit_factor(
                pnl
            ),

        "net":
            float(
                pnl.sum()
            ),
    }


def build_model(
    groups,
    device,
):
    base = (
        brain.MultiSurfaceTechnicalBrain(
            groups
        )
        .to(
            device
        )
    )

    old = torch.load(
        execmod.CKPT,
        map_location=device,
        weights_only=False,
    )

    base.load_state_dict(
        old[
            "model"
        ]
    )

    execution = (
        v671.ExecutionPrecisionBrainV671(
            base
        )
        .to(
            device
        )
    )

    c671 = torch.load(
        v672.V671_CHAMPION,
        map_location=device,
        weights_only=False,
    )

    execution.load_state_dict(
        c671[
            "model"
        ]
    )

    model = (
        v672.DailyOpportunityBrainV672(
            execution
        )
        .to(
            device
        )
    )

    c672 = torch.load(
        v672.CHAMPION,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        c672[
            "model"
        ]
    )

    model.eval()

    return (
        model,
        c672,
    )


def prepare_year(
    pred,
    rows,
    arrays,
    daily,
    execution_targets,
):
    source = arrays[
        "source"
    ][
        rows
    ]

    valid = execution_targets[
        "valid"
    ][
        source
    ]

    gross = execution_targets[
        "gross"
    ][
        source
    ]

    actual_net = (
        gross
        - COST
    )

    safe_actual = np.where(
        valid,
        actual_net,
        -np.inf,
    )

    oracle_task = np.argmax(
        safe_actual,
        axis=1,
    )

    idx = np.arange(
        len(
            rows
        ),
        dtype=np.int64,
    )

    oracle_net = safe_actual[
        idx,
        oracle_task
    ]

    # This exactly follows the current
    # V6.7.2 action-selection rule.
    learned_task_score = (
        pred[
            "p05"
        ]
        + 0.01
        * np.tanh(
            pred[
                "exec_net"
            ]
            / 10.0
        )
    )

    learned_task_score[
        ~valid
    ] = -np.inf

    learned_task = np.argmax(
        learned_task_score,
        axis=1,
    )

    learned_net = safe_actual[
        idx,
        learned_task
    ]

    regret = (
        oracle_net
        - learned_net
    )

    day = daily[
        "day_ns"
    ][
        source
    ]

    top = daily[
        "top"
    ][
        source
    ]

    return {
        "source":
            source,

        "day":
            day,

        "score":
            pred[
                "score"
            ],

        "learned_task":
            learned_task,

        "oracle_task":
            oracle_task,

        "learned_net":
            learned_net,

        "oracle_net":
            oracle_net,

        "regret":
            regret,

        "top":
            top,
    }


def eligible_days(
    day,
):
    counts = pd.Series(
        day
    ).value_counts()

    return set(
        int(x)
        for x in counts[
            counts
            >= MIN_DAY_BARS
        ].index
    )


def first_trigger_indices(
    state,
    threshold,
):
    days = eligible_days(
        state[
            "day"
        ]
    )

    used = set()
    chosen = []

    for i in range(
        len(
            state[
                "score"
            ]
        )
    ):
        d = int(
            state[
                "day"
            ][
                i
            ]
        )

        if d not in days:
            continue

        if d in used:
            continue

        if (
            state[
                "score"
            ][
                i
            ]
            < threshold
        ):
            continue

        used.add(
            d
        )

        chosen.append(
            i
        )

    return (
        np.asarray(
            chosen,
            np.int64,
        ),
        len(
            days
        ),
    )


def daymax_score_indices(
    state,
):
    days = eligible_days(
        state[
            "day"
        ]
    )

    chosen = []

    for d in sorted(
        days
    ):
        idx = np.flatnonzero(
            state[
                "day"
            ]
            == d
        )

        if len(idx) == 0:
            continue

        j = idx[
            np.argmax(
                state[
                    "score"
                ][
                    idx
                ]
            )
        ]

        chosen.append(
            j
        )

    return np.asarray(
        chosen,
        np.int64,
    )


def oracle_bestbar_indices(
    state,
):
    days = eligible_days(
        state[
            "day"
        ]
    )

    chosen = []

    for d in sorted(
        days
    ):
        idx = np.flatnonzero(
            state[
                "day"
            ]
            == d
        )

        if len(idx) == 0:
            continue

        j = idx[
            np.argmax(
                state[
                    "oracle_net"
                ][
                    idx
                ]
            )
        ]

        chosen.append(
            j
        )

    return np.asarray(
        chosen,
        np.int64,
    )


def print_pair(
    name,
    learned,
    oracle,
):
    lm = metrics(
        learned
    )

    om = metrics(
        oracle
    )

    print(
        f"{name:<34} | "
        f"LEARNED WHAT "
        f"N={lm['n']:>4} "
        f"WIN={lm['win']:>6.2%} "
        f"MEAN={lm['mean']:>+7.3f} "
        f"PF={lm['pf']:>6.3f} "
        f"|| ORACLE WHAT "
        f"WIN={om['win']:>6.2%} "
        f"MEAN={om['mean']:>+7.3f} "
        f"PF={om['pf']:>7.3f}"
    )


def audit_year(
    label,
    state,
    thresholds,
):
    print()
    print(
        "=" * 138
    )

    print(
        label
    )

    print(
        "=" * 138
    )

    # ========================================================
    # WHAT SELECTOR QUALITY
    # ========================================================

    learned = state[
        "learned_net"
    ]

    oracle = state[
        "oracle_net"
    ]

    regret = state[
        "regret"
    ]

    print()
    print(
        "ACTION SELECTOR QUALITY"
    )

    print(
        "-" * 138
    )

    lm = metrics(
        learned
    )

    om = metrics(
        oracle
    )

    print(
        f"ALL ROWS | "
        f"LEARNED WHAT "
        f"WIN={lm['win']:.2%} "
        f"MEAN={lm['mean']:+.3f} "
        f"PF={lm['pf']:.3f}"
    )

    print(
        f"ALL ROWS | "
        f"ORACLE WHAT  "
        f"WIN={om['win']:.2%} "
        f"MEAN={om['mean']:+.3f} "
        f"PF={om['pf']:.3f}"
    )

    print(
        "Median action regret:",
        f"{np.median(regret):+.3f} bps",
    )

    print(
        "Mean action regret:",
        f"{np.mean(regret):+.3f} bps",
    )

    print(
        "Learned within 1bps of best:",
        f"{(regret <= 1.0).mean():.2%}",
    )

    print(
        "Learned within 3bps of best:",
        f"{(regret <= 3.0).mean():.2%}",
    )

    print(
        "Learned within 5bps of best:",
        f"{(regret <= 5.0).mean():.2%}",
    )

    # Conditional on TRUE strong opportunity bars.
    for column, name in (
        (
            1,
            "TRUE TOP10",
        ),
        (
            2,
            "TRUE TOP5",
        ),
        (
            3,
            "TRUE TOP2",
        ),
    ):

        mask = (
            state[
                "top"
            ][
                :,
                column
            ]
            > 0.5
        )

        if not mask.any():
            continue

        m_l = metrics(
            learned[
                mask
            ]
        )

        m_o = metrics(
            oracle[
                mask
            ]
        )

        print(
            f"{name:<10} | "
            f"N={mask.sum():>5} | "
            f"LEARNED WHAT "
            f"WIN={m_l['win']:>6.2%} "
            f"MEAN={m_l['mean']:>+7.3f} "
            f"PF={m_l['pf']:>6.3f} "
            f"| ORACLE WHAT "
            f"WIN={m_o['win']:>6.2%}"
        )

    # ========================================================
    # CAUSAL WHEN / WHAT DECOMPOSITION
    # ========================================================

    print()
    print(
        "CAUSAL FIRST-TRIGGER DECOMPOSITION"
    )

    print(
        "-" * 138
    )

    rows = []

    for q in v672.ENTRY_QUANTILES:

        threshold = thresholds[
            float(
                q
            )
        ]

        idx, n_days = (
            first_trigger_indices(
                state,
                threshold,
            )
        )

        coverage = (
            len(
                idx
            )
            / n_days
            if n_days
            else np.nan
        )

        learned_pnl = state[
            "learned_net"
        ][
            idx
        ]

        oracle_pnl = state[
            "oracle_net"
        ][
            idx
        ]

        lm = metrics(
            learned_pnl
        )

        om = metrics(
            oracle_pnl
        )

        rows.append(
            {
                "quantile":
                    q,

                "threshold":
                    threshold,

                "coverage":
                    coverage,

                "trades":
                    len(
                        idx
                    ),

                "learned_win":
                    lm[
                        "win"
                    ],

                "learned_mean":
                    lm[
                        "mean"
                    ],

                "learned_pf":
                    lm[
                        "pf"
                    ],

                "oracle_what_win":
                    om[
                        "win"
                    ],

                "oracle_what_mean":
                    om[
                        "mean"
                    ],

                "oracle_what_pf":
                    om[
                        "pf"
                    ],
            }
        )

        print(
            f"Q={q:>4.2f} "
            f"COVER={coverage:>6.2%} "
            f"N={len(idx):>3} | "
            f"LEARNED WHAT "
            f"WIN={lm['win']:>6.2%} "
            f"MEAN={lm['mean']:>+7.3f} "
            f"PF={lm['pf']:>6.3f} "
            f"|| ORACLE WHAT "
            f"WIN={om['win']:>6.2%} "
            f"MEAN={om['mean']:>+7.3f} "
            f"PF={om['pf']:>7.3f}"
        )

    # ========================================================
    # NON-CAUSAL DIAGNOSTICS
    # ========================================================

    print()
    print(
        "DIAGNOSTIC CEILINGS"
    )

    print(
        "(These use future information and "
        "are NOT executable strategies.)"
    )

    print(
        "-" * 138
    )

    # Best predicted WHEN of the whole day.
    # This removes the optimal-stopping problem
    # but still uses the learned ranking.
    idx = daymax_score_indices(
        state
    )

    print_pair(
        "DAY-MAX PREDICTED WHEN",
        state[
            "learned_net"
        ][
            idx
        ],
        state[
            "oracle_net"
        ][
            idx
        ],
    )

    # True best bar of day.
    # This isolates WHAT performance when WHEN
    # is made perfect by an oracle.
    idx = oracle_bestbar_indices(
        state
    )

    print_pair(
        "ORACLE WHEN",
        state[
            "learned_net"
        ][
            idx
        ],
        state[
            "oracle_net"
        ][
            idx
        ],
    )

    return pd.DataFrame(
        rows
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.7.2A "
        "WHEN / WHAT DECOMPOSITION AUDIT"
    )

    print(
        "=" * 138
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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

    execution_targets = (
        v671.load_execution_targets()
    )

    daily = (
        v672.load_daily_targets()
    )

    model, champion = (
        build_model(
            groups,
            device,
        )
    )

    years = daily[
        "year"
    ][
        arrays[
            "source"
        ]
    ]

    val = split[
        "val"
    ]

    rows23 = val[
        years[
            val
        ]
        == 2023
    ]

    rows24 = val[
        years[
            val
        ]
        == 2024
    ]

    print(
        "Champion epoch:",
        champion[
            "epoch"
        ],
    )

    print(
        "Champion phase:",
        champion[
            "phase"
        ],
    )

    print(
        "2023:",
        f"{len(rows23):,}",
    )

    print(
        "2024:",
        f"{len(rows24):,}",
    )

    print(
        "2025: NOT EVALUATED"
    )

    print(
        "2026: NOT EVALUATED"
    )

    print()
    print(
        "Predicting 2023 ..."
    )

    pred23 = v672.predict(
        model,
        rows23,
        arrays,
        mean,
        std,
        device,
    )

    print(
        "Predicting 2024 ..."
    )

    pred24 = v672.predict(
        model,
        rows24,
        arrays,
        mean,
        std,
        device,
    )

    state23 = prepare_year(
        pred23,
        rows23,
        arrays,
        daily,
        execution_targets,
    )

    state24 = prepare_year(
        pred24,
        rows24,
        arrays,
        daily,
        execution_targets,
    )

    thresholds = {
        float(
            k
        ):
            float(
                v
            )
        for k, v in (
            champion[
                "thresholds_2023"
            ].items()
        )
    }

    out23 = audit_year(
        "2023 VALIDATION DECOMPOSITION",
        state23,
        thresholds,
    )

    out24 = audit_year(
        "2024 FROZEN DECOMPOSITION",
        state24,
        thresholds,
    )

    out23.to_csv(
        OUT
        / "decomposition_2023.csv",
        index=False,
    )

    out24.to_csv(
        OUT
        / "decomposition_2024.csv",
        index=False,
    )

    print()
    print(
        "=" * 138
    )

    print(
        "INTERPRETATION RULES"
    )

    print(
        "=" * 138
    )

    print(
        "1) If LEARNED WHEN + ORACLE WHAT "
        "is strong, action selection is the "
        "primary bottleneck."
    )

    print(
        "2) If DAY-MAX predicted WHEN is much "
        "better than causal first-trigger, "
        "optimal stopping is also a bottleneck."
    )

    print(
        "3) If ORACLE WHEN + LEARNED WHAT is "
        "still weak, V6.7.3 must rebuild the "
        "18-action utility selector."
    )

    print(
        "4) ORACLE WHEN + ORACLE WHAT is only "
        "a feasibility ceiling."
    )

    print()
    print(
        "2025: untouched."
    )

    print(
        "2026: untouched."
    )

    print(
        "Saved:",
        OUT,
    )


if __name__ == "__main__":
    main()
