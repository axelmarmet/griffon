from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger

import os

import wandb

from griffon.models.encoder.count import CounT
from griffon.dataset.count_datamodule import CounTDataModule

wandb.init(project="griffon")
wandb_logger = WandbLogger(project="griffon", save_dir=os.path.join("wandb", wandb.run.name))

LightningCLI(CounT, CounTDataModule, trainer_defaults={"logger":wandb_logger})