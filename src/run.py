from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

from src.pl_data.datamodule import DataModule
from src.pl_module.classifier_model import FloodModel


def build_callbacks() -> list[Callback]:
    callbacks: list[Callback] = [
        LearningRateMonitor(
            logging_interval="step",
            log_momentum=False,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=True,
            min_delta=0.0,
            patience=3,
        ),
        ModelCheckpoint(
            filename="checkpoint",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True,
        ),
    ]
    return callbacks


def run() -> None:
    seed_everything(42, workers=True)

    datamodule: pl.LightningDataModule = DataModule(
        Path("data/train/labelled.csv"),
        num_workers=8,
        batch_size=64,
    )
    model: pl.LightningModule = FloodModel()
    callbacks: list[Callback] = build_callbacks()
    csv_logger = CSVLogger(save_dir="csv_logs")

    trainer = pl.Trainer(
        logger=[csv_logger],
        default_root_dir="ckpts",
        log_every_n_steps=10,
        callbacks=callbacks,
        deterministic=True,
        gpus=-1,
        precision=32,
        max_epochs=35,
        gradient_clip_val=0.5,
        auto_select_gpus=True,
    )

    trainer.tune(model=model, datamodule=datamodule)
    trainer.fit(model=model, datamodule=datamodule)
    csv_logger.save()


if __name__ == "__main__":
    run()
