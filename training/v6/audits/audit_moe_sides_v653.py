import numpy as np
import torch

import training.v6.models.train_technical_moe_v652 as v


CKPT = (
    "training/artifacts/v6/"
    "technical_moe_v652/"
    "best_technical_moe_v652.pt"
)


COVERAGES = (
    5.0,
    2.0,
    1.0,
    0.5,
    0.2,
    0.1,
)


def percentile(
    reference,
    values,
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
            values,
            side="right",
        )
        / len(ref)
    )


def side_calibrate(
    train_score,
    score,
):
    out = np.zeros_like(
        score,
        dtype=np.float64,
    )

    for side in range(2):
        out[:, side] = percentile(
            train_score[:, side],
            score[:, side],
        )

    return out


def profit_factor(x):
    gain = x[
        x > 0
    ].sum()

    loss = -x[
        x < 0
    ].sum()

    if loss <= 0:
        return np.inf

    return float(
        gain / loss
    )


def tail_indices(
    score,
    coverage,
):
    n = max(
        1,
        round(
            len(score)
            * coverage
            / 100.0
        ),
    )

    return np.argpartition(
        score,
        -n,
    )[-n:]


def report_side(
    title,
    score,
    pnl,
    race,
):
    print()
    print(title)
    print("-" * 124)

    for coverage in COVERAGES:
        idx = tail_indices(
            score,
            coverage,
        )

        x = pnl[
            idx
        ]

        r = race[
            idx
        ]

        tp = (
            r == 1
        ).mean()

        sl = (
            r == 0
        ).mean()

        resolved = (
            tp + sl
        )

        if resolved > 0:
            tp_resolved = (
                tp
                / resolved
            )
        else:
            tp_resolved = np.nan

        print(
            f"{coverage:>4.1f}% "
            f"n={len(idx):>5} "
            f"TP={tp:>6.2%} "
            f"SL={sl:>6.2%} "
            f"TP|RES={tp_resolved:>6.2%} "
            f"WIN={(x > 0).mean():>6.2%} "
            f"mean={x.mean():>+7.3f} "
            f"PF={profit_factor(x):>6.3f}"
        )


def report_combined(
    title,
    calibrated,
    pred,
):
    long_side = (
        calibrated[:, 0]
        >= calibrated[:, 1]
    )

    rank = np.maximum(
        calibrated[:, 0],
        calibrated[:, 1],
    )

    pnl = np.where(
        long_side,
        pred[
            "pnl"
        ][:, 0],
        pred[
            "pnl"
        ][:, 1],
    )

    race = np.where(
        long_side,
        pred[
            "race_true"
        ][:, 0],
        pred[
            "race_true"
        ][:, 1],
    )

    print()
    print("=" * 124)
    print(title)
    print("=" * 124)

    for coverage in COVERAGES:
        idx = tail_indices(
            rank,
            coverage,
        )

        x = pnl[
            idx
        ]

        r = race[
            idx
        ]

        tp = (
            r == 1
        ).mean()

        sl = (
            r == 0
        ).mean()

        resolved = (
            tp + sl
        )

        tp_resolved = (
            tp / resolved
            if resolved > 0
            else np.nan
        )

        print(
            f"{coverage:>4.1f}% "
            f"n={len(idx):>5} "
            f"TP={tp:>6.2%} "
            f"SL={sl:>6.2%} "
            f"TP|RES={tp_resolved:>6.2%} "
            f"WIN={(x > 0).mean():>6.2%} "
            f"mean={x.mean():>+7.3f} "
            f"PF={profit_factor(x):>6.3f} "
            f"LONG={long_side[idx].mean():>6.2%}"
        )


def report_tail_gates(
    title,
    score,
    pred,
    expert_names,
    side,
):
    print()
    print(title)

    for coverage in (
        0.5,
        0.2,
    ):
        idx = tail_indices(
            score[:, side],
            coverage,
        )

        gate = pred[
            "gates"
        ][
            idx
        ].mean(
            axis=0
        )

        order = np.argsort(
            gate
        )[::-1]

        text = ", ".join(
            f"{expert_names[i]}={gate[i]:.3f}"
            for i in order[:6]
        )

        print(
            f"  {coverage:.1f}% -> "
            + text
        )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "TEN V6.5.3 "
        "SIDE-SPECIFIC CALIBRATION "
        "& TAIL AUDIT"
    )

    print("=" * 124)
    print("Device:", device)

    (
        arrays,
        split,
        groups,
        names,
        mean,
        std,
    ) = v.load_data()

    model = v.TechnicalMoE(
        groups
    ).to(
        device
    )

    checkpoint = torch.load(
        CKPT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model"
        ]
    )

    print(
        "Champion epoch:",
        checkpoint[
            "epoch"
        ],
    )

    def get(rows):
        dl = v.loader(
            rows,
            False,
            arrays,
            mean,
            std,
        )

        pred = v.predict(
            model,
            dl,
            device,
        )

        score = v.scores(
            pred
        )

        return (
            pred,
            score,
        )

    print(
        "Predicting TRAIN reference ..."
    )

    train_pred, train_score = get(
        split[
            "train"
        ]
    )

    print(
        "Predicting VAL ..."
    )

    val_pred, val_score = get(
        split[
            "val"
        ]
    )

    print(
        "Predicting 2025 ..."
    )

    test_pred, test_score = get(
        split[
            "test2025"
        ]
    )

    policies = (
        "utility",
        "ev",
        "p_race",
        "p_tp",
    )

    for (
        split_name,
        pred,
        score,
    ) in (
        (
            "VAL 2023-2024",
            val_pred,
            val_score,
        ),
        (
            "TEST 2025",
            test_pred,
            test_score,
        ),
    ):
        print()
        print("#" * 124)
        print(split_name)
        print("#" * 124)

        for policy in policies:
            calibrated = side_calibrate(
                train_score[
                    policy
                ],
                score[
                    policy
                ],
            )

            print()
            print(
                f">>> POLICY: "
                f"{policy.upper()}"
            )

            report_side(
                split_name
                + " | "
                + policy.upper()
                + " | LONG ONLY",
                calibrated[
                    :,
                    0
                ],
                pred[
                    "pnl"
                ][
                    :,
                    0
                ],
                pred[
                    "race_true"
                ][
                    :,
                    0
                ],
            )

            report_side(
                split_name
                + " | "
                + policy.upper()
                + " | SHORT ONLY",
                calibrated[
                    :,
                    1
                ],
                pred[
                    "pnl"
                ][
                    :,
                    1
                ],
                pred[
                    "race_true"
                ][
                    :,
                    1
                ],
            )

            report_combined(
                split_name
                + " | "
                + policy.upper()
                + " | SIDE-NEUTRAL",
                calibrated,
                pred,
            )

            if policy in (
                "utility",
                "ev",
            ):
                report_tail_gates(
                    split_name
                    + " | "
                    + policy.upper()
                    + " LONG GATES",
                    calibrated,
                    pred,
                    model.expert_names,
                    0,
                )

                report_tail_gates(
                    split_name
                    + " | "
                    + policy.upper()
                    + " SHORT GATES",
                    calibrated,
                    pred,
                    model.expert_names,
                    1,
                )

    print()
    print("=" * 124)

    print(
        "Resolved-race theoretical "
        "break-even TP share:"
    )

    break_even = (
        -v.SL_VALUE
        / (
            v.TP_VALUE
            - v.SL_VALUE
        )
    )

    print(
        f"{break_even:.2%}"
    )

    print(
        "2026 RESERVED: NOT EVALUATED"
    )


if __name__ == "__main__":
    main()
