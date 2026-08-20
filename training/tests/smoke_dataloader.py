from __future__ import annotations

import torch

from training.data.build_dataset import (
    CONTEXT_BARS,
    FEATURE_COLUMNS,
    HORIZON_BARS,
)
from training.data.datamodule import TenMarketDataModule


def describe(name: str, value: torch.Tensor) -> None:
    print(
        f"{name:20s}",
        "shape=",
        tuple(value.shape),
        "dtype=",
        value.dtype,
        "finite=",
        bool(torch.isfinite(value).all()),
    )


def main() -> None:
    dm = TenMarketDataModule(
        dataset_dir="training/artifacts/dataset_smoke",
        batch_size=32,
        num_workers=0,
    )

    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    print("TEN DataLoader smoke test")
    print("-------------------------")

    for key, value in batch.items():
        describe(key, value)

    assert batch["x"].shape == (
        32,
        CONTEXT_BARS,
        len(FEATURE_COLUMNS),
    )

    assert batch["future_path"].shape == (
        32,
        HORIZON_BARS,
    )

    assert batch["future_high"].shape == (32,)
    assert batch["future_low"].shape == (32,)
    assert batch["future_close"].shape == (32,)
    assert batch["time_to_high"].shape == (32,)
    assert batch["time_to_low"].shape == (32,)
    assert batch["future_volatility"].shape == (32,)

    normalized_mean = batch["x"].mean().item()
    normalized_std = batch["x"].std().item()

    print()
    print("batch normalized mean:", normalized_mean)
    print("batch normalized std :", normalized_std)

    print()
    print("train samples      :", len(dm.train_dataset))
    print("validation samples :", len(dm.validation_dataset))
    print("test samples       :", len(dm.test_dataset))

    print()
    print("OK: TEN DataLoader is model-ready.")


if __name__ == "__main__":
    main()
