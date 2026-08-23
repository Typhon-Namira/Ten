from pathlib import Path

import numpy as np
import pandas as pd
import torch

import training.v6.models.train_dual_brain_v60 as v60


TARGET = Path(
    "training/v6/data_lake/large_move_v60/"
    "large_move_targets_v60.parquet"
)

TECH_DIR = Path(
    "training/v6/data_lake/technical_state_v60"
)

ART = Path(
    "training/artifacts/v6/dual_brain_v60"
)

OUT = Path(
    "training/artifacts/v6/confluence_policy_v602"
)

COVERAGES = (
    0.05,
    0.02,
    0.01,
    0.005,
    0.002,
)

COST_BPS = 0.5


def profit_factor(pnl):
    win = pnl[
        pnl > 0
    ].sum()

    loss = -pnl[
        pnl < 0
    ].sum()

    if loss <= 0:
        return np.inf

    return win / loss


def load_all():
    df = pd.read_parquet(
        TARGET
    )

    tech = np.load(
        TECH_DIR / "technical_features.npy",
        mmap_mode="r",
    )

    tech_valid = np.load(
        TECH_DIR / "technical_valid.npy",
        mmap_mode="r",
    )

    m1 = np.load(
        v60.M1_FEATURES,
        mmap_mode="r",
    )

    source = df[
        "source_row"
    ].to_numpy(
        np.int64
    )

    m1_end = df[
        "m1_end_index"
    ].to_numpy(
        np.int64
    )

    race = df[
        v60.HEAD_COLS
    ].to_numpy(
        np.int8
    )

    labels = (
        race == 1
    ).astype(
        np.float32
    )

    masks = (
        race >= -1
    ).astype(
        np.float32
    )

    regression = np.log1p(
        np.clip(
            df[
                v60.REG_COLS
            ].to_numpy(
                np.float32
            ),
            0.0,
            200.0,
        )
    ).astype(
        np.float32
    )

    eligible = (
        (
            df[
                "horizon_valid"
            ].to_numpy(
                np.int8
            )
            == 1
        )
        & (
            tech_valid[
                source
            ]
            == 1
        )
        & (
            m1_end
            >= v60.SEQ - 1
        )
    )

    norm = np.load(
        ART / "normalization_v60.npz"
    )

    arrays = {
        "m1": m1,
        "tech": tech,
        "source": source,
        "m1_end": m1_end,
        "labels": labels,
        "masks": masks,
        "regression": regression,
    }

    norms = {
        "m1_mean":
            norm["m1_mean"],

        "m1_std":
            norm["m1_std"],

        "tech_mean":
            norm["tech_mean"],

        "tech_std":
            norm["tech_std"],
    }

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = v60.DualBrain(
        n_tech=tech.shape[1]
    ).to(device)

    ckpt = torch.load(
        ART / "best_dual_brain_v60.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    return (
        df,
        race,
        eligible,
        arrays,
        norms,
        model,
        device,
    )


def predict_split(
    rows,
    arrays,
    norms,
    model,
    device,
):
    loader = v60.make_loader(
        rows,
        False,
        arrays,
        norms,
    )

    pred, _, _ = v60.evaluate(
        model,
        loader,
        device,
    )

    return pred


def evaluate_policy(
    name,
    frame,
    race,
    pred,
    confluence,
):
    # Primary TP30/SL15 heads.
    # index 2 = LONG
    # index 3 = SHORT

    pf_long = pred[
        "fusion"
    ][:, 2]

    pf_short = pred[
        "fusion"
    ][:, 3]

    ph_long = pred[
        "historical"
    ][:, 2]

    ph_short = pred[
        "historical"
    ][:, 3]

    pt_long = pred[
        "technical"
    ][:, 2]

    pt_short = pred[
        "technical"
    ][:, 3]

    fusion_long = (
        pf_long >= pf_short
    )

    hist_long = (
        ph_long >= ph_short
    )

    tech_long = (
        pt_long >= pt_short
    )

    agreement = (
        (fusion_long == hist_long)
        & (fusion_long == tech_long)
    )

    fusion_score = np.maximum(
        pf_long,
        pf_short,
    )

    # Conservative confluence score:
    # chosen direction must score strongly
    # in ALL three brains.
    chosen_hist = np.where(
        fusion_long,
        ph_long,
        ph_short,
    )

    chosen_tech = np.where(
        fusion_long,
        pt_long,
        pt_short,
    )

    chosen_fusion = np.where(
        fusion_long,
        pf_long,
        pf_short,
    )

    consensus_score = np.minimum.reduce(
        [
            chosen_hist,
            chosen_tech,
            chosen_fusion,
        ]
    )

    if confluence:
        candidate = agreement
        rank_score = consensus_score
    else:
        candidate = np.ones(
            len(frame),
            dtype=bool,
        )
        rank_score = fusion_score

    chosen_race = np.where(
        fusion_long,
        race[:, 2],
        race[:, 3],
    )

    terminal = np.where(
        fusion_long,
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

    # Ambiguous same-bar race is excluded
    # from predictive precision.
    valid = (
        chosen_race != -2
    ) & (
        chosen_race != -3
    )

    candidate &= valid

    print()
    print(name)
    print("-" * 110)

    print(
        "Brain agreement:",
        f"{agreement.mean():.2%}",
    )

    rows_out = []

    n_total = len(frame)

    candidate_idx = np.flatnonzero(
        candidate
    )

    for coverage in COVERAGES:
        n = max(
            1,
            int(
                round(
                    n_total
                    * coverage
                )
            ),
        )

        n = min(
            n,
            len(candidate_idx),
        )

        scores = rank_score[
            candidate_idx
        ]

        local = np.argpartition(
            scores,
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

        long_share = (
            fusion_long[
                idx
            ].mean()
        )

        row = {
            "split": name,
            "coverage": coverage,
            "signals": len(idx),
            "tp_rate": tp.mean(),
            "sl_rate": sl.mean(),
            "timeout_rate":
                timeout.mean(),
            "win_rate":
                (pnl > 0).mean(),
            "mean_bps":
                pnl.mean(),
            "median_bps":
                np.median(pnl),
            "profit_factor":
                profit_factor(pnl),
            "long_share":
                long_share,
            "agreement_gate":
                confluence,
        }

        rows_out.append(row)

        print(
            f"{coverage:>5.1%}"
            f" n={len(idx):>5}"
            f" TP={tp.mean():>6.2%}"
            f" SL={sl.mean():>6.2%}"
            f" TO={timeout.mean():>6.2%}"
            f" WIN={row['win_rate']:>6.2%}"
            f" mean={row['mean_bps']:>+7.3f}"
            f" PF={row['profit_factor']:>6.3f}"
            f" LONG={long_share:>6.2%}"
        )

    return rows_out


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
    ) = load_all()

    print(
        "TEN V6.0.2 DEPLOYMENT-STYLE "
        "CONFLUENCE AUDIT"
    )

    print("=" * 110)

    print(
        "TP=30bps | SL=15bps | "
        "timeout=30m executable exit | "
        "cost=0.5bps"
    )

    years = df[
        "year"
    ].to_numpy(
        np.int16
    )

    splits = {
        "VAL_2023_2024":
            np.flatnonzero(
                eligible
                & (years >= 2023)
                & (years <= 2024)
            ),

        "TEST_2025":
            np.flatnonzero(
                eligible
                & (years == 2025)
            ),

        "BENCH_2026":
            np.flatnonzero(
                eligible
                & (years == 2026)
            ),
    }

    all_results = []

    for split_name, rows in (
        splits.items()
    ):
        print()
        print("=" * 110)
        print(
            split_name,
            f"({len(rows):,} anchors)"
        )
        print("=" * 110)

        pred = predict_split(
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

        all_results.extend(
            evaluate_policy(
                split_name
                + "_FUSION",
                frame,
                split_race,
                pred,
                False,
            )
        )

        all_results.extend(
            evaluate_policy(
                split_name
                + "_CONFLUENCE",
                frame,
                split_race,
                pred,
                True,
            )
        )

    result = pd.DataFrame(
        all_results
    )

    result.to_csv(
        OUT
        / "confluence_policy_v602.csv",
        index=False,
    )

    print()
    print(
        "Saved:",
        OUT,
    )


if __name__ == "__main__":
    main()
