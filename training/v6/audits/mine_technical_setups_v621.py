from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd

SETUP_DIR = Path(
    "training/v6/data_lake/technical_setup_v620"
)

TARGET = Path(
    "training/v6/data_lake/large_move_v60/"
    "large_move_targets_v60.parquet"
)

OUT = Path(
    "training/artifacts/v6/"
    "technical_setup_miner_v621"
)

Q_LEVELS = (
    0.80,
    0.90,
    0.95,
)

TP_NET = 29.5
SL_NET = -15.5
COST = 0.5

MIN_TRAIN = 200
MIN_VAL = 50


def pf(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    gains = x[
        x > 0
    ].sum()

    losses = -x[
        x < 0
    ].sum()

    if losses <= 0:
        return np.inf

    return float(
        gains / losses
    )


def stats(
    mask,
    race,
    pnl,
):
    n = int(
        mask.sum()
    )

    if n == 0:
        return {
            "n": 0,
            "tp": np.nan,
            "sl": np.nan,
            "to": np.nan,
            "win": np.nan,
            "mean": np.nan,
            "pf": np.nan,
        }

    r = race[
        mask
    ]

    x = pnl[
        mask
    ]

    return {
        "n":
            n,

        "tp":
            float(
                (r == 1).mean()
            ),

        "sl":
            float(
                (r == 0).mean()
            ),

        "to":
            float(
                (r == -1).mean()
            ),

        "win":
            float(
                (x > 0).mean()
            ),

        "mean":
            float(
                x.mean()
            ),

        "pf":
            pf(x),
    }


def add(
    row,
    prefix,
    d,
):
    for k, v in d.items():
        row[
            f"{prefix}_{k}"
        ] = v


def realized(
    df,
    side,
):
    race = df[
        f"{side}_race_tp30_sl15"
    ].to_numpy(
        np.int8
    )

    terminal = (
        df[
            f"{side}_terminal_bps"
        ].to_numpy(
            np.float32
        )
        - COST
    )

    pnl = np.full(
        len(df),
        np.nan,
        dtype=np.float32,
    )

    pnl[
        race == 1
    ] = TP_NET

    pnl[
        race == 0
    ] = SL_NET

    pnl[
        race == -1
    ] = terminal[
        race == -1
    ]

    return race, pnl


def load():
    x = np.load(
        SETUP_DIR
        / "technical_setup_features.npy",
        mmap_mode="r",
    )

    valid = np.load(
        SETUP_DIR
        / "technical_setup_valid.npy",
        mmap_mode="r",
    ).astype(
        bool
    )

    with open(
        SETUP_DIR
        / "feature_names.json"
    ) as f:
        names = json.load(
            f
        )

    with open(
        SETUP_DIR
        / "setup_names.json"
    ) as f:
        setups = json.load(
            f
        )

    if x.shape[1] != len(
        names
    ):
        raise RuntimeError(
            "feature name mismatch"
        )

    df = pd.read_parquet(
        TARGET
    )

    source = df[
        "source_row"
    ].to_numpy(
        np.int64
    )

    year = df[
        "year"
    ].to_numpy(
        np.int16
    )

    horizon = (
        df[
            "horizon_valid"
        ].to_numpy(
            np.int8
        )
        == 1
    )

    if (
        source.min() < 0
        or source.max()
        >= len(valid)
    ):
        raise RuntimeError(
            "source_row out of range"
        )

    col = {
        name: i
        for i, name
        in enumerate(
            names
        )
    }

    score = {
        name: np.asarray(
            x[
                source,
                col[name],
            ],
            dtype=np.float32,
        )
        for name in setups
    }

    split = {
        "train":
            year <= 2022,

        "val":
            (
                (year >= 2023)
                & (year <= 2024)
            ),

        "test2025":
            year == 2025,

        "diag2026":
            year == 2026,
    }

    sides = {}

    for side in (
        "long",
        "short",
    ):
        race, pnl = realized(
            df,
            side,
        )

        ok = (
            horizon
            & valid[
                source
            ]
            & np.isin(
                race,
                [-1, 0, 1],
            )
            & np.isfinite(
                pnl
            )
        )

        sides[
            side.upper()
        ] = {
            "names": [
                name
                for name in setups
                if name.endswith(
                    f"_{side}"
                )
            ],

            "race":
                race,

            "pnl":
                pnl,

            "valid":
                ok,
        }

    return (
        df,
        year,
        score,
        split,
        sides,
    )


def thresholds(
    score,
    split,
    sides,
):
    th = {}
    active = {}
    rows = []

    for side, info in sides.items():
        train = (
            info["valid"]
            & split["train"]
        )

        for q in Q_LEVELS:
            qi = int(
                q * 100
            )

            for name in info[
                "names"
            ]:
                t = float(
                    np.quantile(
                        score[name][
                            train
                        ],
                        q,
                    )
                )

                th[
                    (
                        side,
                        qi,
                        name,
                    )
                ] = t

                m = (
                    score[name]
                    >= t
                )

                if t <= 1e-8:
                    m = np.zeros(
                        len(m),
                        dtype=bool,
                    )

                active[
                    (
                        side,
                        qi,
                        name,
                    )
                ] = m

                rows.append(
                    {
                        "side":
                            side,

                        "quantile":
                            qi,

                        "setup":
                            name,

                        "threshold":
                            t,

                        "train_active":
                            int(
                                (
                                    m
                                    & train
                                ).sum()
                            ),
                    }
                )

    pd.DataFrame(
        rows
    ).to_csv(
        OUT
        / "setup_thresholds.csv",
        index=False,
    )

    return (
        th,
        active,
    )


def mine(
    split,
    sides,
    th,
    active,
):
    rows = []

    for side, info in sides.items():
        for q in Q_LEVELS:
            qi = int(
                q * 100
            )

            available = [
                name
                for name
                in info["names"]
                if th[
                    (
                        side,
                        qi,
                        name,
                    )
                ] > 1e-8
            ]

            print(
                f"{side} Q{qi}: "
                f"{len(available)} usable setups"
            )

            for k in (
                1,
                2,
                3,
            ):
                kept = 0

                for members in combinations(
                    available,
                    k,
                ):
                    m = info[
                        "valid"
                    ].copy()

                    for name in members:
                        m &= active[
                            (
                                side,
                                qi,
                                name,
                            )
                        ]

                    train_n = int(
                        (
                            m
                            & split[
                                "train"
                            ]
                        ).sum()
                    )

                    if train_n < 50:
                        continue

                    kept += 1

                    row = {
                        "side":
                            side,

                        "quantile":
                            qi,

                        "combo_size":
                            k,

                        "members":
                            " + ".join(
                                members
                            ),
                    }

                    for s in (
                        "train",
                        "val",
                        "test2025",
                        "diag2026",
                    ):
                        add(
                            row,
                            s,
                            stats(
                                m
                                & split[
                                    s
                                ],
                                info[
                                    "race"
                                ],
                                info[
                                    "pnl"
                                ],
                            ),
                        )

                    rows.append(
                        row
                    )

                print(
                    f"  size={k}: "
                    f"{kept:,}"
                )

    out = pd.DataFrame(
        rows
    )

    if out.empty:
        raise RuntimeError(
            "no combinations survived"
        )

    return out


def interaction_lift(
    df,
):
    singles = {
        (
            r.side,
            int(
                r['quantile']
            ),
            r.members,
        ): r
        for _, r
        in df[
            df.combo_size == 1
        ].iterrows()
    }

    for split in (
        "train",
        "val",
        "test2025",
    ):
        vals = []

        for _, r in df.iterrows():
            if int(
                r.combo_size
            ) == 1:
                vals.append(
                    0.0
                )
                continue

            members = (
                r.members
                .split(
                    " + "
                )
            )

            base = max(
                singles[
                    (
                        r.side,
                        int(
                            r['quantile']
                        ),
                        member,
                    )
                ][
                    f"{split}_mean"
                ]
                for member
                in members
            )

            vals.append(
                r[
                    f"{split}_mean"
                ]
                - base
            )

        df[
            f"{split}_lift_bps"
        ] = vals

    return df


def rank(
    df,
):
    df[
        "stable_mean"
    ] = np.minimum(
        df.train_mean,
        df.val_mean,
    )

    df[
        "gap"
    ] = np.abs(
        df.train_mean
        - df.val_mean
    )

    df[
        "stable_score"
    ] = (
        df.stable_mean
        - 0.25
        * df.gap
    )

    df[
        "stable_pf"
    ] = np.minimum(
        df.train_pf,
        df.val_pf,
    )

    q = df[
        (
            df.train_n
            >= MIN_TRAIN
        )
        & (
            df.val_n
            >= MIN_VAL
        )
    ].copy()

    return q.sort_values(
        [
            "stable_score",
            "val_mean",
        ],
        ascending=False,
    )


def yearly(
    top,
    year,
    sides,
    active,
):
    rows = []

    for _, r in top.iterrows():
        info = sides[
            r.side
        ]

        qi = int(
            r['quantile']
        )

        m = info[
            "valid"
        ].copy()

        for name in (
            r.members
            .split(
                " + "
            )
        ):
            m &= active[
                (
                    r.side,
                    qi,
                    name,
                )
            ]

        for y in sorted(
            np.unique(
                year
            )
        ):
            d = stats(
                m
                & (
                    year == y
                ),
                info[
                    "race"
                ],
                info[
                    "pnl"
                ],
            )

            rows.append(
                {
                    "side":
                        r.side,

                    "quantile":
                        qi,

                    "combo_size":
                        int(
                            r.combo_size
                        ),

                    "members":
                        r.members,

                    "stable_score":
                        r.stable_score,

                    "year":
                        int(y),

                    **d,
                }
            )

    pd.DataFrame(
        rows
    ).to_csv(
        OUT
        / "top50_yearly_stability.csv",
        index=False,
    )


def report(
    q,
):
    print()
    print(
        "=" * 120
    )

    print(
        "TOP 30 STABLE TECHNICAL SETUPS"
    )

    print(
        "SELECTION: 2016-2024 ONLY"
    )

    print(
        "=" * 120
    )

    for i, (
        _,
        r,
    ) in enumerate(
        q.head(
            30
        ).iterrows(),
        1,
    ):
        print(
            f"{i:02d} "
            f"{r.side:<5} "
            f"Q{int(r['quantile']):02d} "
            f"K={int(r.combo_size)} "
            f"TRn={int(r.train_n):>6,} "
            f"Vn={int(r.val_n):>5,} "
            f"TR={r.train_mean:+6.2f} "
            f"PF={r.train_pf:5.2f} "
            f"VAL={r.val_mean:+6.2f} "
            f"PF={r.val_pf:5.2f} "
            f"2025={r.test2025_mean:+6.2f} "
            f"PF={r.test2025_pf:5.2f} "
            f"LIFT={r.val_lift_bps:+6.2f}"
        )

        print(
            "    "
            + r.members
        )

    combos = q[
        q.combo_size >= 2
    ].sort_values(
        "val_lift_bps",
        ascending=False,
    )

    print()
    print(
        "=" * 120
    )

    print(
        "TOP 20 POSITIVE INTERACTIONS"
    )

    print(
        "=" * 120
    )

    for i, (
        _,
        r,
    ) in enumerate(
        combos.head(
            20
        ).iterrows(),
        1,
    ):
        print(
            f"{i:02d} "
            f"{r.side:<5} "
            f"Q{int(r['quantile']):02d} "
            f"K={int(r.combo_size)} "
            f"n={int(r.val_n):>5,} "
            f"VAL={r.val_mean:+6.2f} "
            f"PF={r.val_pf:5.2f} "
            f"LIFT={r.val_lift_bps:+6.2f} "
            f"2025={r.test2025_mean:+6.2f}"
        )

        print(
            "    "
            + r.members
        )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.2.1 "
        "HISTORICAL TECHNICAL SETUP MINER"
    )

    print(
        "=" * 120
    )

    (
        df,
        year,
        score,
        split,
        sides,
    ) = load()

    print(
        "Rows:",
        f"{len(df):,}",
    )

    for side, info in sides.items():
        for s in (
            "train",
            "val",
            "test2025",
            "diag2026",
        ):
            d = stats(
                info[
                    "valid"
                ]
                & split[
                    s
                ],
                info[
                    "race"
                ],
                info[
                    "pnl"
                ],
            )

            print(
                f"BASE {side:<5} "
                f"{s:<8} "
                f"n={d['n']:>7,} "
                f"TP={d['tp']:6.2%} "
                f"WIN={d['win']:6.2%} "
                f"mean={d['mean']:+7.3f} "
                f"PF={d['pf']:6.3f}"
            )

    th, active = thresholds(
        score,
        split,
        sides,
    )

    result = mine(
        split,
        sides,
        th,
        active,
    )

    result = interaction_lift(
        result
    )

    qualified = rank(
        result
    )

    result.to_csv(
        OUT
        / "all_setup_combinations.csv",
        index=False,
    )

    qualified.to_csv(
        OUT
        / "qualified_ranked_setups.csv",
        index=False,
    )

    yearly(
        qualified.head(
            50
        ),
        year,
        sides,
        active,
    )

    report(
        qualified
    )

    print()
    print(
        "Candidates:",
        f"{len(result):,}",
    )

    print(
        "Qualified:",
        f"{len(qualified):,}",
    )

    print(
        "Saved:",
        OUT,
    )

    print(
        "2025/2026 were not "
        "used for selection."
    )


if __name__ == "__main__":
    main()
