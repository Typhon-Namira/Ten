from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from training.data.build_dataset import (
    CONTEXT_BARS,
    FEATURE_COLUMNS,
    HORIZON_BARS,
)


@dataclass(frozen=True)
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(
        cls,
        bars: pd.DataFrame,
        *,
        train_end: pd.Timestamp,
    ) -> "FeatureScaler":
        train = bars.loc[
            bars["timestamp"] <= train_end,
            list(FEATURE_COLUMNS),
        ]

        values = train.to_numpy(dtype=np.float32)
        values = values[np.isfinite(values).all(axis=1)]

        if len(values) == 0:
            raise ValueError("no valid training features available for scaler")

        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = values.std(axis=0, dtype=np.float64).astype(np.float32)

        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

        return cls(mean=mean, std=std)


class TenMarketDataset(Dataset):
    def __init__(
        self,
        *,
        dataset_dir: Path,
        split: str,
        scaler: FeatureScaler,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split: {split}")

        bars = pd.read_parquet(dataset_dir / "bars.parquet")
        samples = pd.read_parquet(dataset_dir / "samples.parquet")

        samples = samples.loc[samples["split"] == split].reset_index(drop=True)

        if samples.empty:
            raise ValueError(f"split has zero samples: {split}")

        self.samples = samples

        self.features = bars.loc[
            :,
            list(FEATURE_COLUMNS),
        ].to_numpy(dtype=np.float32)

        self.scaler_mean = scaler.mean.astype(np.float32)
        self.scaler_std = scaler.std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples.iloc[index]

        start = int(sample["context_start_index"])
        cutoff = int(sample["cutoff_index"])

        x = self.features[start : cutoff + 1]

        if x.shape != (CONTEXT_BARS, len(FEATURE_COLUMNS)):
            raise RuntimeError(
                f"bad context shape {x.shape}, expected "
                f"{(CONTEXT_BARS, len(FEATURE_COLUMNS))}"
            )

        if not np.isfinite(x).all():
            raise RuntimeError("non-finite value reached model context")

        x = (x - self.scaler_mean) / self.scaler_std

        path = np.asarray(
            sample["future_close_log_returns"],
            dtype=np.float32,
        )

        if path.shape != (HORIZON_BARS,):
            raise RuntimeError(
                f"bad future path shape {path.shape}, "
                f"expected {(HORIZON_BARS,)}"
            )

        # Bars 1..6 are converted to 0..1 for the timing heads.
        time_to_high = float(sample["time_to_high_bars"]) / HORIZON_BARS
        time_to_low = float(sample["time_to_low_bars"]) / HORIZON_BARS

        return {
            "x": torch.from_numpy(x.copy()),

            # Complete future close trajectory:
            # [t+5m, t+10m, ..., t+30m]
            "future_path": torch.from_numpy(path.copy()),

            # Future 30-minute geometry.
            "future_high": torch.tensor(
                float(sample["future_high_log_return"]),
                dtype=torch.float32,
            ),
            "future_low": torch.tensor(
                float(sample["future_low_log_return"]),
                dtype=torch.float32,
            ),
            "future_close": torch.tensor(
                float(sample["future_close_log_return"]),
                dtype=torch.float32,
            ),

            # When extrema occur within the six-bar horizon.
            "time_to_high": torch.tensor(
                time_to_high,
                dtype=torch.float32,
            ),
            "time_to_low": torch.tensor(
                time_to_low,
                dtype=torch.float32,
            ),

            # Realized uncertainty/volatility in the future horizon.
            "future_volatility": torch.tensor(
                float(sample["future_realized_volatility"]),
                dtype=torch.float32,
            ),
        }
