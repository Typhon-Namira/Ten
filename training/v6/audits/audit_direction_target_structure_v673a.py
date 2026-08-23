from pathlib import Path

import numpy as np
import pandas as pd

import training.v6.models.train_multisurface_technical_brain_v661 as brain
import training.v6.models.train_execution_precision_brain_v671 as v671
import training.v6.models.train_daily_opportunity_brain_v672 as v672


OUT = Path(
    "training/artifacts/v6/"
    "direction_target_structure_v673a"
)

STEP_NS = 300_000_000_000

START_YEAR = 2016
END_YEAR = 2024

LAGS = (
    1,
    3,
    6,
    12,
)

ORACLE_GAPS = (
    0.0,
    3.0,
    10.0,
    20.0,
)


def continuity(
    timestamps_ns,
    lag,
):
    n = len(
        timestamps_ns
    )

    out = np.zeros(
        n,
        dtype=bool,
    )

    if lag >= n:
        return out

    idx = np.arange(
        lag,
        n,
        dtype=np.int64,
    )

    out[
        idx
    ] = (
        timestamps_ns[
            idx
        ]
        - timestamps_ns[
            idx - lag
        ]
        == lag * STEP_NS
    )

    return out


def flip_rate(
    labels,
    valid,
    timestamps_ns,
    lag,
):
    cont = continuity(
        timestamps_ns,
        lag,
    )

    prev_valid = np.zeros(
        len(
            valid
        ),
        dtype=bool,
    )

    prev_valid[
        lag:
    ] = valid[
        :-lag
    ]

    mask = (
        valid
        & prev_valid
        & cont
    )

    idx = np.flatnonzero(
        mask
    )

    if len(idx) == 0:
        return np.nan

    return float(
        (
            labels[
                idx
            ]
            != labels[
                idx - lag
            ]
        ).mean()
    )


def describe_label(
    name,
    label,
    valid,
    timestamps_ns,
):
    x = label[
        valid
    ]

    row = {
        "name":
            name,

        "n":
            int(
                valid.sum()
            ),

        "short_rate":
            float(
                x.mean()
            ),
    }

    for lag in LAGS:
        row[
            f"flip_{lag}"
        ] = flip_rate(
            label,
            valid,
            timestamps_ns,
            lag,
        )

    return row


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.7.3A "
        "DIRECTION TARGET STRUCTURE AUDIT"
    )

    print(
        "=" * 130
    )

    execution = (
        v671.load_execution_targets()
    )

    daily = (
        v672.load_daily_targets()
    )

    timestamps_ns = (
        pd.to_datetime(
            execution[
                "timestamp"
            ],
            utc=True,
        )
        .astype(
            "int64"
        )
        .to_numpy(
            np.int64
        )
    )

    years = execution[
        "year"
    ]

    design = (
        (years >= START_YEAR)
        & (years <= END_YEAR)
    )

    oracle_side = daily[
        "side"
    ].astype(
        np.int8
    )

    oracle_gap = daily[
        "direction_gap"
    ].astype(
        np.float32
    )

    oracle_valid = (
        design
        & (
            oracle_side >= 0
        )
        & np.isfinite(
            oracle_gap
        )
    )

    print(
        "Rows total:",
        f"{len(years):,}",
    )

    print(
        "Design rows 2016-2024:",
        f"{design.sum():,}",
    )

    print(
        "2025 excluded."
    )

    print(
        "2026 excluded."
    )

    print()
    print(
        "=" * 130
    )

    print(
        "CURRENT MAX-OF-MAX ORACLE SIDE"
    )

    print(
        "=" * 130
    )

    oracle_rows = []

    for min_gap in ORACLE_GAPS:

        valid = (
            oracle_valid
            & (
                np.abs(
                    oracle_gap
                )
                >= min_gap
            )
        )

        row = describe_label(
            f"oracle_gap_{min_gap:g}",
            oracle_side,
            valid,
            timestamps_ns,
        )

        row[
            "median_abs_gap"
        ] = float(
            np.median(
                np.abs(
                    oracle_gap[
                        valid
                    ]
                )
            )
        )

        oracle_rows.append(
            row
        )

        print(
            f"|GAP|>={min_gap:>4.0f}bps "
            f"N={row['n']:>7,} "
            f"SHORT={row['short_rate']:>6.2%} "
            f"FLIP1={row['flip_1']:>6.2%} "
            f"FLIP3={row['flip_3']:>6.2%} "
            f"FLIP6={row['flip_6']:>6.2%} "
            f"FLIP12={row['flip_12']:>6.2%} "
            f"MED|GAP|={row['median_abs_gap']:.2f}"
        )

    pd.DataFrame(
        oracle_rows
    ).to_csv(
        OUT
        / "oracle_side_structure.csv",
        index=False,
    )

    print()
    print(
        "=" * 130
    )

    print(
        "FIXED TASK LONG-vs-SHORT LABELS"
    )

    print(
        "=" * 130
    )

    fixed_rows = []

    pair_labels = []
    pair_valids = []

    for j in range(
        9
    ):

        meta = brain.TASKS[
            j
        ]

        short_j = (
            j + 9
        )

        long_net = (
            execution[
                "gross"
            ][
                :,
                j
            ]
            - 0.5
        )

        short_net = (
            execution[
                "gross"
            ][
                :,
                short_j
            ]
            - 0.5
        )

        valid = (
            design
            & execution[
                "valid"
            ][
                :,
                j
            ]
            & execution[
                "valid"
            ][
                :,
                short_j
            ]
            & np.isfinite(
                long_net
            )
            & np.isfinite(
                short_net
            )
        )

        gap = (
            short_net
            - long_net
        )

        # Remove exact economic ties.
        valid &= (
            np.abs(
                gap
            )
            > 1e-6
        )

        label = np.zeros(
            len(
                years
            ),
            dtype=np.int8,
        )

        label[
            valid
        ] = (
            short_net[
                valid
            ]
            > long_net[
                valid
            ]
        ).astype(
            np.int8
        )

        pair_labels.append(
            label
        )

        pair_valids.append(
            valid
        )

        row = describe_label(
            (
                f"H{meta['horizon']}_"
                f"TP{meta['tp']}_"
                f"SL{meta['sl']}"
            ),
            label,
            valid,
            timestamps_ns,
        )

        compare = (
            valid
            & oracle_valid
        )

        row[
            "oracle_agreement"
        ] = float(
            (
                label[
                    compare
                ]
                == oracle_side[
                    compare
                ]
            ).mean()
        )

        compare10 = (
            compare
            & (
                np.abs(
                    oracle_gap
                )
                >= 10.0
            )
        )

        row[
            "oracle_agreement_gap10"
        ] = (
            float(
                (
                    label[
                        compare10
                    ]
                    == oracle_side[
                        compare10
                    ]
                ).mean()
            )
            if compare10.any()
            else np.nan
        )

        row[
            "median_abs_pair_gap"
        ] = float(
            np.median(
                np.abs(
                    gap[
                        valid
                    ]
                )
            )
        )

        fixed_rows.append(
            row
        )

        print(
            f"{row['name']:<18} "
            f"N={row['n']:>7,} "
            f"SHORT={row['short_rate']:>6.2%} "
            f"ORACLE={row['oracle_agreement']:>6.2%} "
            f"ORACLE10={row['oracle_agreement_gap10']:>6.2%} "
            f"FLIP1={row['flip_1']:>6.2%} "
            f"FLIP6={row['flip_6']:>6.2%}"
        )

    pd.DataFrame(
        fixed_rows
    ).to_csv(
        OUT
        / "fixed_task_direction_structure.csv",
        index=False,
    )

    print()
    print(
        "=" * 130
    )

    print(
        "CONSENSUS DIRECTION ACROSS 9 "
        "FIXED ACTION PAIRS"
    )

    print(
        "=" * 130
    )

    labels = np.stack(
        pair_labels,
        axis=1,
    )

    valids = np.stack(
        pair_valids,
        axis=1,
    )

    short_votes = (
        labels
        * valids.astype(
            np.int8
        )
    ).sum(
        axis=1
    )

    vote_count = valids.sum(
        axis=1
    )

    consensus_valid = (
        design
        & (
            vote_count > 0
        )
        & (
            short_votes * 2
            != vote_count
        )
    )

    consensus = np.zeros(
        len(
            years
        ),
        dtype=np.int8,
    )

    consensus[
        consensus_valid
    ] = (
        short_votes[
            consensus_valid
        ]
        * 2
        > vote_count[
            consensus_valid
        ]
    ).astype(
        np.int8
    )

    row = describe_label(
        "fixed_pair_majority",
        consensus,
        consensus_valid,
        timestamps_ns,
    )

    compare = (
        consensus_valid
        & oracle_valid
    )

    agreement = (
        consensus[
            compare
        ]
        == oracle_side[
            compare
        ]
    ).mean()

    print(
        "Rows:",
        f"{row['n']:,}",
    )

    print(
        "SHORT rate:",
        f"{row['short_rate']:.2%}",
    )

    print(
        "Oracle agreement:",
        f"{agreement:.2%}",
    )

    print(
        "Flip 1 bar:",
        f"{row['flip_1']:.2%}",
    )

    print(
        "Flip 3 bars:",
        f"{row['flip_3']:.2%}",
    )

    print(
        "Flip 6 bars:",
        f"{row['flip_6']:.2%}",
    )

    print(
        "Flip 12 bars:",
        f"{row['flip_12']:.2%}",
    )

    print()
    print(
        "=" * 130
    )

    print(
        "YEARLY ORACLE SIDE STABILITY"
    )

    print(
        "=" * 130
    )

    yearly = []

    for yr in range(
        START_YEAR,
        END_YEAR + 1,
    ):

        valid = (
            oracle_valid
            & (
                years == yr
            )
        )

        r = describe_label(
            str(
                yr
            ),
            oracle_side,
            valid,
            timestamps_ns,
        )

        yearly.append(
            {
                "year":
                    yr,

                **r,
            }
        )

        print(
            f"{yr} "
            f"N={r['n']:>7,} "
            f"SHORT={r['short_rate']:>6.2%} "
            f"FLIP1={r['flip_1']:>6.2%} "
            f"FLIP3={r['flip_3']:>6.2%} "
            f"FLIP6={r['flip_6']:>6.2%}"
        )

    pd.DataFrame(
        yearly
    ).to_csv(
        OUT
        / "yearly_oracle_side_structure.csv",
        index=False,
    )

    print()
    print(
        "=" * 130
    )

    print(
        "INTERPRETATION"
    )

    print(
        "=" * 130
    )

    print(
        "If max-of-max oracle direction "
        "flips much faster than fixed-task "
        "or consensus direction, the current "
        "side label is structurally noisy."
    )

    print(
        "If fixed H30/H60/H120 labels are "
        "substantially more stable, V6.7.4 "
        "should learn horizon-specific "
        "direction first, then utility."
    )

    print(
        "If all direction labels are equally "
        "unstable, the next step is longer "
        "historical/micro context rather than "
        "another side head."
    )

    print(
        "2025: not evaluated."
    )

    print(
        "2026: not evaluated."
    )

    print(
        "Saved:",
        OUT,
    )


if __name__ == "__main__":
    main()
