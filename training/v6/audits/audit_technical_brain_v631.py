from pathlib import Path

import numpy as np
import torch

import training.v6.models.train_advanced_technical_brain_v630 as v630


CKPT = Path(
    "training/artifacts/v6/"
    "advanced_technical_brain_v630/"
    "best_advanced_technical_brain_v630.pt"
)


def percentile(ref, x):
    ref = np.sort(
        np.asarray(ref, dtype=np.float64)
    )

    return (
        np.searchsorted(
            ref,
            x,
            side="right",
        )
        / len(ref)
    )


def pf(x):
    gain = x[x > 0].sum()
    loss = -x[x < 0].sum()

    if loss <= 0:
        return np.inf

    return float(gain / loss)


def calibrate(train, pred):
    n = len(pred["tp"])

    scores = np.zeros(
        (n, 2),
        dtype=np.float64,
    )

    components = {}

    for head, weight in (
        ("tp", 0.55),
        ("win", 0.20),
        ("quality", 0.15),
    ):
        p = np.zeros(
            (n, 2),
            dtype=np.float64,
        )

        for side in range(2):
            p[:, side] = percentile(
                train[head][:, side],
                pred[head][:, side],
            )

        components[head] = p
        scores += weight * p

    action_p = np.zeros(
        (n, 2),
        dtype=np.float64,
    )

    action_p[:, 0] = percentile(
        train["action"][:, 1],
        pred["action"][:, 1],
    )

    action_p[:, 1] = percentile(
        train["action"][:, 2],
        pred["action"][:, 2],
    )

    components["action"] = action_p

    scores += 0.10 * action_p

    return scores, components


def report_side(
    name,
    score,
    pnl,
    tp,
):
    print()
    print(name)
    print("-" * 112)

    for cov in (
        5.0,
        2.0,
        1.0,
        0.5,
        0.2,
    ):
        n = max(
            1,
            round(
                len(score)
                * cov
                / 100
            ),
        )

        idx = np.argpartition(
            score,
            -n,
        )[-n:]

        x = pnl[idx]

        print(
            f"{cov:>4.1f}% "
            f"n={n:>5} "
            f"TP={tp[idx].mean():>6.2%} "
            f"WIN={(x > 0).mean():>6.2%} "
            f"mean={x.mean():>+7.3f} "
            f"PF={pf(x):>6.3f}"
        )


def report_combined(
    name,
    score,
    pred,
):
    long_side = (
        score[:, 0]
        >= score[:, 1]
    )

    selected_score = np.maximum(
        score[:, 0],
        score[:, 1],
    )

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

    print()
    print("=" * 112)
    print(name)
    print("=" * 112)

    for cov in (
        5.0,
        2.0,
        1.0,
        0.5,
        0.2,
    ):
        n = max(
            1,
            round(
                len(pnl)
                * cov
                / 100
            ),
        )

        idx = np.argpartition(
            selected_score,
            -n,
        )[-n:]

        x = pnl[idx]

        print(
            f"{cov:>4.1f}% "
            f"n={n:>5} "
            f"TP={tp[idx].mean():>6.2%} "
            f"WIN={(x > 0).mean():>6.2%} "
            f"mean={x.mean():>+7.3f} "
            f"PF={pf(x):>6.3f} "
            f"LONG={long_side[idx].mean():>6.2%} "
            f"score={selected_score[idx].mean():.4f}"
        )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "TEN V6.3.1 "
        "SIDE-NEUTRAL TECHNICAL AUDIT"
    )
    print("=" * 112)
    print("Device:", device)

    (
        arrays,
        split,
        groups,
        feature_names,
        mean,
        std,
    ) = v630.load_data()

    model = v630.AdvancedTechnicalBrain(
        len(feature_names),
        groups,
    ).to(device)

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

    print("Predicting TRAIN calibration set...")
    train = get(
        split["train"]
    )

    print("Predicting VAL...")
    val = get(
        split["val"]
    )

    print("Predicting 2025...")
    test = get(
        split["test2025"]
    )

    for name, pred in (
        ("VAL 2023-2024", val),
        ("TEST 2025", test),
    ):
        score, parts = calibrate(
            train,
            pred,
        )

        report_side(
            name + " | LONG ONLY",
            score[:, 0],
            pred["pnl"][:, 0],
            pred["tp_true"][:, 0],
        )

        report_side(
            name + " | SHORT ONLY",
            score[:, 1],
            pred["pnl"][:, 1],
            pred["tp_true"][:, 1],
        )

        report_combined(
            name
            + " | SIDE-NEUTRAL",
            score,
            pred,
        )

    print()
    print(
        "2026 NOT EVALUATED."
    )


if __name__ == "__main__":
    main()
