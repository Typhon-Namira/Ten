from __future__ import annotations

import json
from pathlib import Path

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader

from training.data.dataset import FeatureScaler, TenMarketDataset


class TenMarketDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        dataset_dir: str | Path,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        super().__init__()

        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.scaler: FeatureScaler | None = None

        self.train_dataset: TenMarketDataset | None = None
        self.validation_dataset: TenMarketDataset | None = None
        self.test_dataset: TenMarketDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        metadata = json.loads(
            (self.dataset_dir / "metadata.json").read_text(
                encoding="utf-8"
            )
        )

        bars = pd.read_parquet(
            self.dataset_dir / "bars.parquet"
        )

        train_end = pd.Timestamp(metadata["train_end"])

        self.scaler = FeatureScaler.fit(
            bars,
            train_end=train_end,
        )

        self.train_dataset = TenMarketDataset(
            dataset_dir=self.dataset_dir,
            split="train",
            scaler=self.scaler,
        )

        self.validation_dataset = TenMarketDataset(
            dataset_dir=self.dataset_dir,
            split="validation",
            scaler=self.scaler,
        )

        self.test_dataset = TenMarketDataset(
            dataset_dir=self.dataset_dir,
            split="test",
            scaler=self.scaler,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.validation_dataset is not None

        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
