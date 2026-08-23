from pathlib import Path

import numpy as np
import pandas as pd
import torch

import training.v6.models.train_multisurface_technical_brain_v661 as brain
import training.v6.models.train_execution_precision_brain_v671 as v671
import training.v6.models.train_daily_opportunity_brain_v672 as v672
import training.v6.audits.audit_when_what_decomposition_v672a as decomp


OUT = Path(
    "training/artifacts/v6/"
    "action_anatomy_v672b"
)

COST = 0.5

LONG = np.arange(
    0,
    9,
    dtype=np.int64,
)

SHORT = np.arange(
    9,
    18,
    dtype=np.int64,
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
        }

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
            float(
                pf
            ),
    }


def choose_best_in_side(
    values,
    side,
):
    n = len(
        values
    )

    out = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    long_mask = (
        side == 0
    )

    short_mask = (
        side == 1
    )

    if long_mask.any():

        local = np.argmax(
            values[
                long_mask
            ][
                :,
                LONG
            ],
            axis=1,
        )

        out[
            long_mask
        ] = LONG[
            local
        ]

    if short_mask.any():

        local = np.argmax(
            values[
                short_mask
            ][
                :,
                SHORT
            ],
            axis=1,
        )

        out[
            short_mask
        ] = SHORT[
            local
        ]

    return out


def prepare(
    pred,
    rows,
    arrays,
    execution_targets,
    daily,
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

    actual = (
        gross
        - COST
    )

    actual = np.where(
        valid,
        actual,
        -np.inf,
    )

    learned_score = (
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

    learned_score = np.where(
        valid,
        learned_score,
        -np.inf,
    )

    row = np.arange(
        len(
            rows
        )
    )

    oracle_task = np.argmax(
        actual,
        axis=1,
    )

    learned_task = np.argmax(
        learned_score,
        axis=1,
    )

    best_long = np.max(
        actual[
            :,
            LONG
        ],
        axis=1,
    )

    best_short = np.max(
        actual[
            :,
            SHORT
        ],
        axis=1,
    )

    oracle_side = (
        best_short
        > best_long
    ).astype(
        np.int8
    )

    side_gap = np.abs(
        best_short
        - best_long
    )

    # Explicit V6.7.2 direction head.
    head_side = (
        pred[
            "side"
        ]
        >= 0.5
    ).astype(
        np.int8
    )

    # Implicit side from strongest
    # learned action score.
    learned_best_long = np.max(
        learned_score[
            :,
            LONG
        ],
        axis=1,
    )

    learned_best_short = np.max(
        learned_score[
            :,
            SHORT
        ],
        axis=1,
    )

    score_side = (
        learned_best_short
        > learned_best_long
    ).astype(
        np.int8
    )

    # --------------------------------
    # ISOLATE SIDE ERROR
    # --------------------------------

    # Give model perfect side,
    # but force it to select the
    # horizon/barrier itself.
    oracle_side_learned_task = (
        choose_best_in_side(
            learned_score,
            oracle_side,
        )
    )

    # --------------------------------
    # ISOLATE WITHIN-SIDE ERROR
    # --------------------------------

    # Give model's explicit side,
    # then use oracle horizon/barrier
    # within that side.
    head_side_oracle_task = (
        choose_best_in_side(
            actual,
            head_side,
        )
    )

    # Same experiment but side is
    # inferred from action scores.
    score_side_oracle_task = (
        choose_best_in_side(
            actual,
            score_side,
        )
    )

    # Full oracle.
    oracle_pnl = actual[
        row,
        oracle_task,
    ]

    learned_pnl = actual[
        row,
        learned_task,
    ]

    oracle_side_learned_pnl = actual[
        row,
        oracle_side_learned_task,
    ]

    head_side_oracle_pnl = actual[
        row,
        head_side_oracle_task,
    ]

    score_side_oracle_pnl = actual[
        row,
        score_side_oracle_task,
    ]

    return {
        "source":
            source,

        "day":
            daily[
                "day_ns"
            ][
                source
            ],

        "top":
            daily[
                "top"
            ][
                source
            ],

        "score":
            pred[
                "score"
            ],

        "oracle_side":
            oracle_side,

        "head_side":
            head_side,

        "score_side":
            score_side,

        "side_gap":
            side_gap,

        "oracle_task":
            oracle_task,

        "learned_task":
            learned_task,

        "learned":
            learned_pnl,

        "oracle":
            oracle_pnl,

        "oracle_side_learned_task":
            oracle_side_learned_pnl,

        "head_side_oracle_task":
            head_side_oracle_pnl,

        "score_side_oracle_task":
            score_side_oracle_pnl,
    }


def print_metric(
    name,
    pnl,
):
    m = metrics(
        pnl
    )

    print(
        f"{name:<34} "
        f"N={m['n']:>6} "
        f"WIN={m['win']:>7.2%} "
        f"MEAN={m['mean']:>+8.3f} "
        f"PF={m['pf']:>7.3f}"
    )

    return m


def analyze_subset(
    label,
    state,
    mask,
):
    print()
    print(
        label
    )

    print(
        "-" * 125
    )

    rows = []

    variants = (
        (
            "LEARNED SIDE + LEARNED TASK",
            "learned",
        ),

        (
            "ORACLE SIDE + LEARNED TASK",
            "oracle_side_learned_task",
        ),

        (
            "HEAD SIDE + ORACLE TASK",
            "head_side_oracle_task",
        ),

        (
            "SCORE SIDE + ORACLE TASK",
            "score_side_oracle_task",
        ),

        (
            "ORACLE SIDE + ORACLE TASK",
            "oracle",
        ),
    )

    for name, key in variants:

        pnl = state[
            key
        ][
            mask
        ]

        m = print_metric(
            name,
            pnl,
        )

        rows.append(
            {
                "subset":
                    label,

                "variant":
                    name,

                **m,
            }
        )

    # Only evaluate side accuracy when
    # economic difference between LONG
    # and SHORT is meaningful.
    side_mask = (
        mask
        & (
            state[
                "side_gap"
            ]
            >= 1.0
        )
    )

    if side_mask.any():

        head_acc = (
            state[
                "head_side"
            ][
                side_mask
            ]
            == state[
                "oracle_side"
            ][
                side_mask
            ]
        ).mean()

        score_acc = (
            state[
                "score_side"
            ][
                side_mask
            ]
            == state[
                "oracle_side"
            ][
                side_mask
            ]
        ).mean()

        print()
        print(
            "Meaningful SIDE cases:",
            f"{side_mask.sum():,}",
        )

        print(
            "Explicit side-head accuracy:",
            f"{head_acc:.2%}",
        )

        print(
            "Action-score side accuracy:",
            f"{score_acc:.2%}",
        )

    return rows


def analyze_year(
    label,
    state,
    thresholds,
):
    print()
    print(
        "=" * 125
    )

    print(
        label
    )

    print(
        "=" * 125
    )

    all_mask = np.ones(
        len(
            state[
                "learned"
            ]
        ),
        dtype=bool,
    )

    rows = analyze_subset(
        "ALL ROWS",
        state,
        all_mask,
    )

    for j, name in (
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
                j
            ]
            > 0.5
        )

        rows += analyze_subset(
            name,
            state,
            mask,
        )

    print()
    print(
        "CAUSAL DAILY ACTION ANATOMY"
    )

    print(
        "-" * 125
    )

    for q in (
        0.25,
        0.50,
        0.90,
    ):

        threshold = thresholds[
            float(
                q
            )
        ]

        idx, total_days = (
            decomp.first_trigger_indices(
                state,
                threshold,
            )
        )

        coverage = (
            len(
                idx
            )
            / total_days
        )

        print()
        print(
            f"Q={q:.2f} "
            f"COVER={coverage:.2%} "
            f"N={len(idx)}"
        )

        print_metric(
            "LEARNED ALL",
            state[
                "learned"
            ][
                idx
            ],
        )

        print_metric(
            "ORACLE SIDE / LEARNED TASK",
            state[
                "oracle_side_learned_task"
            ][
                idx
            ],
        )

        print_metric(
            "HEAD SIDE / ORACLE TASK",
            state[
                "head_side_oracle_task"
            ][
                idx
            ],
        )

        print_metric(
            "SCORE SIDE / ORACLE TASK",
            state[
                "score_side_oracle_task"
            ][
                idx
            ],
        )

        print_metric(
            "FULL ORACLE WHAT",
            state[
                "oracle"
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
        "TEN V6.7.2B ACTION ANATOMY AUDIT"
    )

    print(
        "=" * 125
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

    execution_targets = (
        v671.load_execution_targets()
    )

    daily = (
        v672.load_daily_targets()
    )

    model, champion = (
        decomp.build_model(
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
        "Champion:",
        champion[
            "epoch"
        ],
        "phase",
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

    state23 = prepare(
        pred23,
        rows23,
        arrays,
        execution_targets,
        daily,
    )

    state24 = prepare(
        pred24,
        rows24,
        arrays,
        execution_targets,
        daily,
    )

    thresholds = {
        float(k):
            float(v)
        for k, v
        in champion[
            "thresholds_2023"
        ].items()
    }

    out23 = analyze_year(
        "2023 ACTION ANATOMY",
        state23,
        thresholds,
    )

    out24 = analyze_year(
        "2024 FROZEN ACTION ANATOMY",
        state24,
        thresholds,
    )

    out23.to_csv(
        OUT
        / "action_anatomy_2023.csv",
        index=False,
    )

    out24.to_csv(
        OUT
        / "action_anatomy_2024.csv",
        index=False,
    )

    print()
    print(
        "=" * 125
    )

    print(
        "READING THE RESULT"
    )

    print(
        "=" * 125
    )

    print(
        "If ORACLE SIDE + LEARNED TASK "
        "jumps sharply, SIDE is the major bottleneck."
    )

    print(
        "If LEARNED SIDE + ORACLE TASK "
        "jumps sharply, within-side horizon/barrier "
        "selection is the major bottleneck."
    )

    print(
        "If both jump, V6.7.3 must solve "
        "direction and action utility jointly."
    )

    print(
        "2025/2026 remain untouched."
    )

    print(
        "Saved:",
        OUT,
    )


if __name__ == "__main__":
    main()
