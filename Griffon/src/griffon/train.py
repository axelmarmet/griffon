from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger

import os

import wandb

from griffon.models.encoder.count import CounT
from griffon.dataset.count_datamodule import CounTDataModule

wandb.init(project="griffon")
wandb_logger = WandbLogger(project="griffon", save_dir=os.path.join("wandb", wandb.run.name))

class MyLightningCLI(LightningCLI):

    def before_fit(self):
        # print("hiii")
        # print("-----------------")
        # self.trainer.tune(self.model, datamodule=self.datamodule)
        # print("done")
        assert isinstance(self.trainer, Trainer)
        # should only give the train dataloader
        self.trainer.tune(self.model, datamodule=self.datamodule)
        return

MyLightningCLI(CounT, CounTDataModule, trainer_defaults={"logger":wandb_logger,
                                                         "auto_lr_find":True})

