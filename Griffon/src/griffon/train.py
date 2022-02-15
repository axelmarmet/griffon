import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.distributed import rank_zero_only

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

    @rank_zero_only
    def _config_logger(self):
        assert isinstance(self.trainer, Trainer)

        if isinstance(self.trainer.logger, WandbLogger):
            exp_name = self.trainer.logger.experiment.name
            print(type(exp_name))
            self.trainer.logger._save_dir = os.path.join("wandb", exp_name)
            # because the wandb logger set logging to debug
            logging.getLogger("wandb").setLevel(logging.WARNING)
            logging.getLogger("requests").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            self.trainer.logger.watch(self.model, log="all")

    def before_test(self):
        self._config_logger()

    def before_fit(self):
        self._config_logger()
        # should only give the train dataloader
        self.trainer.tune(self.model, datamodule=self.datamodule)
        return

if __name__ == "__main__":
    MyLightningCLI()

