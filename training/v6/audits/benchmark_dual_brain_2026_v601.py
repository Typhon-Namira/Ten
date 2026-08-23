from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

import training.v6.models.train_dual_brain_v60 as v60


CKPT = Path(
    "training/artifacts/v6/dual_brain_v60/"
    "best_dual_brain_v60.pt"
)

TARGET = Path(
    "training/v6/data_lake/large_move_v60/"
    "large_move_targets_v60.parquet"
)

TECH_DIR = Path(
    "training/v6/data_lake/technical_state_v60"
)

OUT = Path(
    "training/artifacts/v6/"
    "benchmark_2026_v601"
)


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "TEN V6.0 — FROZEN 2026 BENCHMARK"
    )
    print("=" * 105)

    print("Device:", device)

    df = pd.read_parquet(TARGET)

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
    ].to_numpy(np.int64)

    m1_end = df[
        "m1_end_index"
    ].to_numpy(np.int64)

    year = df[
        "year"
    ].to_numpy(np.int16)

    race = df[
        v60.HEAD_COLS
    ].to_numpy(np.int8)

    labels = (
        race == 1
    ).astype(np.float32)

    masks = (
        race >= -1
    ).astype(np.float32)

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
    ).astype(np.float32)

    eligible = (
        (
            df[
                "horizon_valid"
            ].to_numpy(np.int8)
            == 1
        )
        & (
            tech_valid[source]
            == 1
        )
        & (
            m1_end
            >= v60.SEQ - 1
        )
    )

    rows = np.flatnonzero(
        eligible
        & (year == 2026)
    )

    print(
        "2026 anchors:",
        f"{len(rows):,}",
    )

    print()
    print(
        "Base rates:"
    )

    print(
        "LONG30:",
        f"{labels[rows, 2].mean():.2%}",
    )

    print(
        "SHORT30:",
        f"{labels[rows, 3].mean():.2%}",
    )

    norm = np.load(
        CKPT.parent
        / "normalization_v60.npz"
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

    loader = v60.make_loader(
        rows,
        False,
        arrays,
        norms,
    )

    model = v60.DualBrain(
        n_tech=tech.shape[1]
    ).to(device)

    ckpt = torch.load(
        CKPT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    print()
    print(
        "Frozen epoch:",
        ckpt["epoch"],
    )

    print(
        "Frozen VAL AP:",
        f"{ckpt['val_primary_ap']:.4f}",
    )

    pred, y, mask = v60.evaluate(
        model,
        loader,
        device,
    )

    print()
    print("=" * 105)
    print("2026 HISTORICAL BRAIN")
    print("=" * 105)

    v60.print_metrics(
        "2026 — HISTORICAL",
        pred["historical"],
        y,
        mask,
    )

    print()
    print("=" * 105)
    print("2026 TECHNICAL BRAIN")
    print("=" * 105)

    v60.print_metrics(
        "2026 — TECHNICAL",
        pred["technical"],
        y,
        mask,
    )

    print()
    print("=" * 105)
    print("2026 FUSION")
    print("=" * 105)

    v60.print_metrics(
        "2026 — FUSION",
        pred["fusion"],
        y,
        mask,
    )

    # Save exact frozen predictions.
    save = pd.DataFrame(
        {
            "source_row":
                source[rows],

            "available_at":
                df.iloc[rows][
                    "available_at"
                ].to_numpy(),
        }
    )

    for j, name in enumerate(
        v60.HEAD_NAMES
    ):
        save[
            f"actual_{name}"
        ] = labels[
            rows,
            j,
        ]

        save[
            f"hist_{name}"
        ] = pred[
            "historical"
        ][:, j]

        save[
            f"tech_{name}"
        ] = pred[
            "technical"
        ][:, j]

        save[
            f"fusion_{name}"
        ] = pred[
            "fusion"
        ][:, j]

    save.to_parquet(
        OUT
        / "predictions_2026_v601.parquet",
        index=False,
        compression="zstd",
    )

    print()
    print(
        "Saved:",
        OUT,
    )


if __name__ == "__main__":
    main()
