from pathlib import Path

import numpy as np
import pandas as pd

import training.v6.audits.audit_confluence_policy_v602 as v602


OUT = Path(
    "training/artifacts/v6/"
    "rank_calibrated_policy_v603"
)

COVERAGES = (
    0.05,
    0.02,
    0.01,
    0.005,
    0.002,
)

COST_BPS = 0.5


def percentile_against_reference(
    x,
    reference,
):
    ref = np.sort(
        np.asarray(
            reference,
            dtype=np.float64,
        )
    )

    return (
        np.searchsorted(
            ref,
            x,
            side="right",
        )
        / len(ref)
    )


def profit_factor(pnl):
    gain = pnl[
        pnl > 0
    ].sum()

    loss = -pnl[
        pnl < 0
    ].sum()

    if loss <= 0:
        return np.inf

    return gain / loss


def calibrated_scores(
    pred,
    reference_pred,
):
    result = {}

    for branch in (
        "historical",
        "technical",
        "fusion",
    ):
        result[branch] = {}

        for side, j in (
            ("long", 2),
            ("short", 3),
        ):
            result[
                branch
            ][side] = (
                percentile_against_reference(
                    pred[
                        branch
                    ][:, j],
                    reference_pred[
                        branch
                    ][:, j],
                )
            )

    return result


def build_policy_scores(cal):
    h_long = cal[
        "historical"
    ]["long"]

    h_short = cal[
        "historical"
    ]["short"]

    t_long = cal[
        "technical"
    ]["long"]

    t_short = cal[
        "technical"
    ]["short"]

    f_long = cal[
        "fusion"
    ]["long"]

    f_short = cal[
        "fusion"
    ]["short"]

    h_side = (
        h_long >= h_short
    )

    t_side = (
        t_long >= t_short
    )

    f_side = (
        f_long >= f_short
    )

    agreement = (
        (h_side == t_side)
        & (h_side == f_side)
    )

    long_consensus = np.minimum.reduce(
        [
            h_long,
            t_long,
            f_long,
        ]
    )

    short_consensus = np.minimum.reduce(
        [
            h_short,
            t_short,
            f_short,
        ]
    )

    predicted_long = (
        long_consensus
        >= short_consensus
    )

    chosen_score = np.where(
        predicted_long,
        long_consensus,
        short_consensus,
    )

    opposite_score = np.where(
        predicted_long,
        short_consensus,
        long_consensus,
    )

    margin = (
        chosen_score
        - opposite_score
    )

    return (
        predicted_long,
        chosen_score,
        margin,
        agreement,
    )


def evaluate(
    name,
    frame,
    race,
    pred,
    reference_pred,
):
    cal = calibrated_scores(
        pred,
        reference_pred,
    )

    (
        predicted_long,
        consensus,
        margin,
        agreement,
    ) = build_policy_scores(
        cal
    )

    chosen_race = np.where(
        predicted_long,
        race[:, 2],
        race[:, 3],
    )

    terminal = np.where(
        predicted_long,
        frame[
            "long_terminal_bps"
        ].to_numpy(
            np.float64
        ),
        frame[
            "short_terminal_bps"
        ].to_numpy(
            np.float64
        ),
    )

    valid = (
        (chosen_race != -2)
        & (chosen_race != -3)
    )

    candidate = (
        agreement
        & valid
    )

    # Require strength AND directional separation.
    score = (
        consensus
        + 0.25
        * np.maximum(
            margin,
            0.0,
        )
    )

    candidate_idx = np.flatnonzero(
        candidate
    )

    print()
    print(name)
    print("-" * 110)

    print(
        "3-BRAIN AGREEMENT:",
        f"{agreement.mean():.2%}",
    )

    print(
        "Agreement LONG share:",
        f"{predicted_long[agreement].mean():.2%}",
    )

    rows = []

    for coverage in COVERAGES:
        n = max(
            1,
            int(
                round(
                    len(frame)
                    * coverage
                )
            ),
        )

        n = min(
            n,
            len(candidate_idx),
        )

        local_scores = (
            score[
                candidate_idx
            ]
        )

        local = np.argpartition(
            local_scores,
            -n,
        )[-n:]

        idx = candidate_idx[
            local
        ]

        r = chosen_race[
            idx
        ]

        tp = (
            r == 1
        )

        sl = (
            r == 0
        )

        timeout = (
            r == -1
        )

        pnl = np.empty(
            len(idx),
            dtype=np.float64,
        )

        pnl[tp] = 30.0
        pnl[sl] = -15.0

        pnl[timeout] = (
            terminal[
                idx
            ][
                timeout
            ]
        )

        pnl -= COST_BPS

        pf = profit_factor(
            pnl
        )

        long_share = (
            predicted_long[
                idx
            ].mean()
        )

        row = {
            "split":
                name,

            "coverage":
                coverage,

            "signals":
                len(idx),

            "tp_rate":
                tp.mean(),

            "sl_rate":
                sl.mean(),

            "timeout_rate":
                timeout.mean(),

            "win_rate":
                (pnl > 0).mean(),

            "mean_bps":
                pnl.mean(),

            "median_bps":
                np.median(pnl),

            "profit_factor":
                pf,

            "long_share":
                long_share,

            "mean_consensus":
                consensus[
                    idx
                ].mean(),

            "mean_margin":
                margin[
                    idx
                ].mean(),
        }

        rows.append(
            row
        )

        print(
            f"{coverage:>5.1%}"
            f" n={len(idx):>5}"
            f" TP={tp.mean():>6.2%}"
            f" SL={sl.mean():>6.2%}"
            f" TO={timeout.mean():>6.2%}"
            f" WIN={(pnl > 0).mean():>6.2%}"
            f" mean={pnl.mean():>+7.3f}"
            f" PF={pf:>6.3f}"
            f" LONG={long_share:>6.2%}"
            f" C={consensus[idx].mean():>5.3f}"
            f" M={margin[idx].mean():>+6.3f}"
        )

    return rows


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        df,
        race,
        eligible,
        arrays,
        norms,
        model,
        device,
    ) = v602.load_all()

    years = df[
        "year"
    ].to_numpy(
        np.int16
    )

    print(
        "TEN V6.0.3 RANK-CALIBRATED "
        "DUAL-BRAIN POLICY"
    )
    print("=" * 110)

    print(
        "Score calibration reference: 2022"
    )

    print(
        "NO LABELS USED FOR CALIBRATION"
    )

    calibration_rows = (
        np.flatnonzero(
            eligible
            & (years == 2022)
        )
    )

    print(
        "Calibration anchors:",
        f"{len(calibration_rows):,}",
    )

    reference_pred = (
        v602.predict_split(
            calibration_rows,
            arrays,
            norms,
            model,
            device,
        )
    )

    splits = {
        "2023_2024":
            np.flatnonzero(
                eligible
                & (years >= 2023)
                & (years <= 2024)
            ),

        "2025":
            np.flatnonzero(
                eligible
                & (years == 2025)
            ),

        "2026":
            np.flatnonzero(
                eligible
                & (years == 2026)
            ),
    }

    all_rows = []

    for name, rows in splits.items():
        print()
        print("=" * 110)

        print(
            name,
            f"({len(rows):,} anchors)"
        )

        print("=" * 110)

        pred = v602.predict_split(
            rows,
            arrays,
            norms,
            model,
            device,
        )

        frame = (
            df.iloc[
                rows
            ]
            .reset_index(
                drop=True
            )
        )

        split_race = race[
            rows
        ]

        all_rows.extend(
            evaluate(
                name,
                frame,
                split_race,
                pred,
                reference_pred,
            )
        )

    result = pd.DataFrame(
        all_rows
    )

    result.to_csv(
        OUT
        / "rank_calibrated_policy_v603.csv",
        index=False,
    )

    print()
    print("=" * 110)

    print(
        "BREAKEVEN REFERENCE"
    )

    print(
        "Pure TP30/SL15 with 0.5bps cost:"
    )

    breakeven = (
        15.5
        / (
            29.5
            + 15.5
        )
    )

    print(
        "Approx TP-first break-even "
        "(ignoring timeout PnL):",
        f"{breakeven:.2%}",
    )

    print()
    print(
        "Saved:",
        OUT,
    )


if __name__ == "__main__":
    main()
