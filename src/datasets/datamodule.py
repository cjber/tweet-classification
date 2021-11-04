from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from src.pl_data.csv_dataset import CSVDataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        num_workers: int,
        batch_size: int,
    ):
        super().__init__()
        self.data_dir = data_dir

        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage: Optional[str]):
        if stage == "fit" or stage is None:
            csv_data = CSVDataset(self.data_dir)
            data_len = len(csv_data)
            val_len = data_len // 10
            self.train_dataset, self.val_dataset = random_split(
                csv_data,
                [data_len - val_len, val_len],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.data_dir=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )
