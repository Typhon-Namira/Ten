from pathlib import Path

import numpy as np
import torch

import training.v6.models.train_advanced_technical_brain_v630 as v630


CKPT = Path(
    "training/artifacts/v6/"
    "advanced_technical_brain_v630/"
    "best_advanced_technical_brain_v630.pt"
)


def percentile(ref, values):
    ref = np.sort(
        np.asarray(
            ref,
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
    out = np.zeros(
        score.shape,
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


def report(
    title,
    side_score,
    pred,
    use_margin=False,
):
    long_side = (
        side_score[:, 0]
        >= side_score[:, 1]
    )

    maximum = np.maximum(
        side_score[:, 0],
        side_score[:, 1],
    )

    margin = np.abs(
        side_score[:, 0]
        - side_score[:, 1]
    )

    if use_margin:
        rank_score = (
            maximum
            + 0.25 * margin
        )
    else:
        rank_score = maximum

    pnl = np.where(
        long_side,
        pred["pnl"][:, 0],
        pred["pnl"][:, 1],
    )

    tp = np.where(
        long_side,
        pred["tp_true"][:, 0],
        pred["tp_true"][:, 1],
    )

    sl = np.isclose(
        pnl,
        -15.5,
        atol=1e-4,
    )

    print()
    print(title)
    print("-" * 120)

    for cov in (
        5.0,
        2.0,
        1.0,
        0.5,
        0.2,
        0.1,
    ):
        n = max(
            1,
            round(
                len(pnl)
                * cov
                / 100.0
            ),
        )

        idx = np.argpartition(
            rank_score,
            -n,
        )[-n:]

        x = pnl[idx]

        print(
            f"{cov:>4.1f}% "
            f"n={n:>5} "
            f"TP={tp[idx].mean():>6.2%} "
            f"SL={sl[idx].mean():>6.2%} "
            f"WIN={(x > 0).mean():>6.2%} "
            f"mean={x.mean():>+7.3f} "
            f"PF={profit_factor(x):>6.3f} "
            f"LONG={long_side[idx].mean():>6.2%} "
            f"rank={rank_score[idx].mean():.4f}"
        )


def extract_heads(pred):
    return {
        "TP":
            pred["tp"],

        "WIN":
            pred["win"],

        "QUALITY":
            pred["quality"],

        "ACTION":
            np.column_stack(
                [
                    pred[
                        "action"
                    ][:, 1],

                    pred[
                        "action"
                    ][:, 2],
                ]
            ),
    }


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "TEN V6.3.2 "
        "TECHNICAL HEAD ATTRIBUTION AUDIT"
    )
    print("=" * 120)

    print(
        "Device:",
        device,
    )

    (
        arrays,
        split,
        groups,
        feature_names,
        mean,
        std,
    ) = v630.load_data()

    model = (
        v630.AdvancedTechnicalBrain(
            len(feature_names),
            groups,
        )
        .to(device)
    )

    checkpoint = torch.load(
        CKPT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    print(
        "Champion epoch:",
        checkpoint["epoch"],
    )

    def predict_rows(rows):
        loader = v630.make_loader(
            rows,
            False,
            arrays,
            mean,
            std,
        )

        return v630.predict(
            model,
            loader,
            device,
        )

    print(
        "Predicting TRAIN reference ..."
    )

    train = predict_rows(
        split["train"]
    )

    print(
        "Predicting VAL ..."
    )

    val = predict_rows(
        split["val"]
    )

    print(
        "Predicting 2025 ..."
    )

    test = predict_rows(
        split["test2025"]
    )

    train_heads = extract_heads(
        train
    )

    for split_name, pred in (
        (
            "VAL 2023-2024",
            val,
        ),
        (
            "TEST 2025",
            test,
        ),
    ):
        heads = extract_heads(
            pred
        )

        print()
        print("=" * 120)
        print(split_name)
        print("=" * 120)

        for head_name in (
            "TP",
            "WIN",
            "QUALITY",
            "ACTION",
        ):
            calibrated = (
                side_calibrate(
                    train_heads[
                        head_name
                    ],
                    heads[
                        head_name
                    ],
                )
            )

            report(
                split_name
                + " | "
                + head_name
                + " ONLY",
                calibrated,
                pred,
                False,
            )

            if head_name == "TP":
                report(
                    split_name
                    + " | TP + SIDE MARGIN",
                    calibrated,
                    pred,
                    True,
                )

    print()
    print(
        "2026 NOT EVALUATED."
    )


if __name__ == "__main__":
    main()
