import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger

import os

import wandb

from griffon.models.encoder.count import CounT
from griffon.dataset.count_datamodule import CounTDataModule
import logging
# import flash.image
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, DATAMODULE_REGISTRY

# MODEL_REGISTRY.register_classes(flash.image, pl.LightningModule)
# print(MODEL_REGISTRY)

# wandb.init(project="griffon")
# wandb_logger = WandbLogger(project="griffon", save_dir=os.path.join("wandb", wandb.run.name))



class MyLightningCLI(LightningCLI):

    def before_fit(self):
        assert isinstance(self.trainer, Trainer)

        if isinstance(self.trainer.logger, WandbLogger):
            self.trainer.logger._save_dir = os.path.join("wandb", self.trainer.logger.experiment.name)
            # because the wandb logger set logging to debug
            logging.getLogger("wandb").setLevel(logging.WARNING)
            logging.getLogger("requests").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)

        # should only give the train dataloader
        self.trainer.tune(self.model, datamodule=self.datamodule)
        return

if __name__ == "__main__":
    MyLightningCLI()

