from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(
    "training/v6/data_lake/large_move_v60/"
    "large_move_targets_v60.parquet"
)

OUT_DIR = Path(
    "training/v6/data_lake/action_value_v61"
)

OUT = (
    OUT_DIR
    / "action_value_targets_v61.parquet"
)

TP = 30.0
SL = 15.0
COST = 0.5


def action_value(
    race,
    terminal,
):
    value = np.full(
        len(race),
        np.nan,
        dtype=np.float32,
    )

    tp = race == 1
    sl = race == 0
    timeout = race == -1

    value[tp] = (
        TP - COST
    )

    value[sl] = (
        -SL - COST
    )

    value[timeout] = (
        terminal[timeout]
        - COST
    )

    return value


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.1 ACTION-VALUE TARGET BUILDER"
    )
    print("=" * 100)

    df = pd.read_parquet(
        SRC
    )

    long_race = df[
        "long_race_tp30_sl15"
    ].to_numpy(
        np.int8
    )

    short_race = df[
        "short_race_tp30_sl15"
    ].to_numpy(
        np.int8
    )

    long_terminal = df[
        "long_terminal_bps"
    ].to_numpy(
        np.float32
    )

    short_terminal = df[
        "short_terminal_bps"
    ].to_numpy(
        np.float32
    )

    long_value = action_value(
        long_race,
        long_terminal,
    )

    short_value = action_value(
        short_race,
        short_terminal,
    )

    valid = (
        np.isfinite(long_value)
        & np.isfinite(short_value)
        & (
            df[
                "horizon_valid"
            ].to_numpy(
                np.int8
            )
            == 1
        )
    )

    # 0 = NO TRADE
    # 1 = LONG
    # 2 = SHORT
    best_action = np.zeros(
        len(df),
        dtype=np.int8,
    )

    choose_long = (
        valid
        & (long_value > 0)
        & (
            long_value
            >= short_value
        )
    )

    choose_short = (
        valid
        & (short_value > 0)
        & (
            short_value
            > long_value
        )
    )

    best_action[
        choose_long
    ] = 1

    best_action[
        choose_short
    ] = 2

    best_value = np.maximum.reduce(
        [
            np.zeros(
                len(df),
                dtype=np.float32,
            ),
            np.nan_to_num(
                long_value,
                nan=-1e9,
            ),
            np.nan_to_num(
                short_value,
                nan=-1e9,
            ),
        ]
    )

    value_gap = np.abs(
        long_value
        - short_value
    )

    out = df[
        [
            "source_row",
            "available_at",
            "m1_end_index",
            "year",
            "horizon_valid",
            "entry_bid",
            "entry_ask",
            "entry_mid",
            "spread_bps_v6",
        ]
    ].copy()

    out[
        "long_value_bps"
    ] = long_value

    out[
        "short_value_bps"
    ] = short_value

    out[
        "best_value_bps"
    ] = best_value

    out[
        "value_gap_bps"
    ] = value_gap.astype(
        np.float32
    )

    out[
        "best_action"
    ] = best_action

    out[
        "long_profitable"
    ] = (
        long_value > 0
    ).astype(
        np.int8
    )

    out[
        "short_profitable"
    ] = (
        short_value > 0
    ).astype(
        np.int8
    )

    out[
        "target_valid"
    ] = valid.astype(
        np.int8
    )

    out.to_parquet(
        OUT,
        index=False,
        compression="zstd",
    )

    print(
        "Rows:",
        f"{len(out):,}",
    )

    print(
        "Valid:",
        f"{valid.sum():,}",
        f"({valid.mean():.2%})",
    )

    print()
    print("BEST ACTION")
    print("-" * 100)

    for action, name in (
        (0, "NO_TRADE"),
        (1, "LONG"),
        (2, "SHORT"),
    ):
        mask = (
            valid
            & (
                best_action
                == action
            )
        )

        print(
            f"{name:<12}",
            f"{mask.sum():>8,}",
            f"{mask.sum() / valid.sum():>7.2%}",
        )

    print()
    print("VALUE DISTRIBUTION")
    print("-" * 100)

    for name, value in (
        ("LONG", long_value),
        ("SHORT", short_value),
        ("BEST", best_value),
    ):
        x = value[
            valid
        ]

        print(
            f"{name:<8}",
            f"mean={np.mean(x):+.3f}",
            f"median={np.median(x):+.3f}",
            f"p90={np.quantile(x, .90):+.3f}",
        )

    print()
    print("BY YEAR")
    print("-" * 100)

    temp = pd.DataFrame(
        {
            "year":
                out["year"],

            "valid":
                valid,

            "action":
                best_action,

            "long_value":
                long_value,

            "short_value":
                short_value,
        }
    )

    for year in sorted(
        temp[
            "year"
        ].unique()
    ):
        z = temp[
            (temp["year"] == year)
            & temp["valid"]
        ]

        print(
            year,
            f"n={len(z):,}",
            f"LONG={(z.action == 1).mean():.2%}",
            f"SHORT={(z.action == 2).mean():.2%}",
            f"NO={(z.action == 0).mean():.2%}",
            f"EV_L={z.long_value.mean():+.2f}",
            f"EV_S={z.short_value.mean():+.2f}",
        )

    print()
    print(
        "Saved:",
        OUT,
    )


if __name__ == "__main__":
    main()
