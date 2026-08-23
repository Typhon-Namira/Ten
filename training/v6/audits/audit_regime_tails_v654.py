import numpy as np
import torch

import training.v6.models.train_technical_moe_v652 as v


CKPT = (
    "training/artifacts/v6/"
    "technical_moe_v652/"
    "best_technical_moe_v652.pt"
)

MIN_N = 60


def pf(x):
    gain = x[x > 0].sum()
    loss = -x[x < 0].sum()

    if loss <= 0:
        return np.inf

    return float(gain / loss)


def percentile(ref, x):
    ref = np.sort(
        np.asarray(
            ref,
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


def qcuts(ref):
    return np.quantile(
        ref,
        [
            0.33,
            0.67,
            0.85,
        ],
    )


def bin_regime(
    x,
    cuts,
):
    return np.digitize(
        x,
        cuts,
        right=False,
    )


def report_group(
    split_name,
    policy_name,
    side_name,
    score,
    pnl,
    race,
    regime_name,
    regime,
):
    order = np.argsort(
        score
    )

    tail_n = max(
        1,
        round(
            len(score)
            * 0.01
        ),
    )

    tail = order[
        -tail_n:
    ]

    print()
    print(
        f"{split_name} | "
        f"{policy_name} | "
        f"{side_name} | "
        f"{regime_name}"
    )

    print("-" * 118)

    for value in np.unique(
        regime
    ):
        idx = tail[
            regime[
                tail
            ]
            == value
        ]

        if len(idx) < MIN_N:
            continue

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

        tp_res = (
            tp / resolved
            if resolved > 0
            else np.nan
        )

        print(
            f"regime={int(value)} "
            f"n={len(idx):>5} "
            f"TP={tp:>6.2%} "
            f"SL={sl:>6.2%} "
            f"TP|RES={tp_res:>6.2%} "
            f"mean={x.mean():>+7.3f} "
            f"PF={pf(x):>6.3f}"
        )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "TEN V6.5.4 "
        "REGIME-CONDITIONED TAIL MINER"
    )

    print("=" * 118)
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

    ckpt = torch.load(
        CKPT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    print(
        "Champion epoch:",
        ckpt["epoch"],
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

        source = arrays[
            "source"
        ][rows]

        current = np.asarray(
            arrays[
                "features"
            ][source],
            dtype=np.float32,
        )

        current = (
            current - mean
        ) / std

        return (
            pred,
            score,
            current,
        )

    print(
        "Predicting TRAIN ..."
    )

    train_pred, train_score, train_x = get(
        split["train"]
    )

    print(
        "Predicting VAL ..."
    )

    val_pred, val_score, val_x = get(
        split["val"]
    )

    print(
        "Predicting 2025 ..."
    )

    test_pred, test_score, test_x = get(
        split["test2025"]
    )

    expert_names = (
        model.expert_names
    )

    expert_indices = {
        name:
            i
        for i, name
        in enumerate(
            expert_names
        )
    }

    # Router probability itself is
    # a useful learned regime description.
    regime_sources = {}

    for expert in (
        "volatility",
        "session",
        "trend_structure",
        "support_resistance",
        "breakout_retest",
        "liquidity",
        "momentum",
    ):
        if expert not in expert_indices:
            continue

        i = expert_indices[
            expert
        ]

        train_values = train_pred[
            "gates"
        ][:, i]

        cuts = qcuts(
            train_values
        )

        regime_sources[
            "gate_" + expert
        ] = (
            cuts,
            val_pred[
                "gates"
            ][:, i],
            test_pred[
                "gates"
            ][:, i],
        )

    # Dominant expert is categorical,
    # no quantile fit required.
    regime_sources[
        "dominant_expert"
    ] = (
        None,
        np.argmax(
            val_pred[
                "gates"
            ],
            axis=1,
        ),
        np.argmax(
            test_pred[
                "gates"
            ],
            axis=1,
        ),
    )

    for (
        split_name,
        pred,
        score,
        which,
    ) in (
        (
            "VAL 2023-2024",
            val_pred,
            val_score,
            1,
        ),
        (
            "TEST 2025",
            test_pred,
            test_score,
            2,
        ),
    ):
        print()
        print("#" * 118)
        print(split_name)
        print("#" * 118)

        for policy_name in (
            "p_tp",
            "p_race",
            "ev",
        ):
            train_policy = train_score[
                policy_name
            ]

            policy = score[
                policy_name
            ]

            calibrated = np.zeros_like(
                policy,
                dtype=np.float64,
            )

            for side in range(
                2
            ):
                calibrated[
                    :,
                    side
                ] = percentile(
                    train_policy[
                        :,
                        side
                    ],
                    policy[
                        :,
                        side
                    ],
                )

            for side, side_name in (
                (
                    0,
                    "LONG",
                ),
                (
                    1,
                    "SHORT",
                ),
            ):
                for (
                    regime_name,
                    data,
                ) in regime_sources.items():

                    cuts = data[0]
                    values = data[
                        which
                    ]

                    if cuts is None:
                        regime = values
                    else:
                        regime = bin_regime(
                            values,
                            cuts,
                        )

                    report_group(
                        split_name,
                        policy_name.upper(),
                        side_name,
                        calibrated[
                            :,
                            side
                        ],
                        pred[
                            "pnl"
                        ][
                            :,
                            side
                        ],
                        pred[
                            "race_true"
                        ][
                            :,
                            side
                        ],
                        regime_name,
                        regime,
                    )

    print()
    print("=" * 118)

    break_even = (
        -v.SL_VALUE
        / (
            v.TP_VALUE
            - v.SL_VALUE
        )
    )

    print(
        "Resolved race break-even:",
        f"{break_even:.2%}",
    )

    print(
        "2026 RESERVED: NOT EVALUATED"
    )


if __name__ == "__main__":
    main()
