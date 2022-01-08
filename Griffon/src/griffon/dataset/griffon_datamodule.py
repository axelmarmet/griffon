import os
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

from math import ceil

from griffon.dataset.griffon_dataset import GriffonDataset


@DATAMODULE_REGISTRY
class GriffonDataModule(pl.LightningDataModule):

    def __init__(self, data_root:str, batch_size:int, num_workers:int=4):
        super().__init__()
        self.griffon_root = os.path.join(data_root, "griffon")

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage:str):
        assert stage in ['fit', 'validate', 'test', 'predict']
        if stage == 'fit':
            self.train_split = GriffonDataset(self.griffon_root, "train")
            self.valid_split = GriffonDataset(self.griffon_root, "valid")
            self.steps_per_epoch = len(self.train_split)
        elif stage == 'validate':
            self.valid_split = GriffonDataset(self.griffon_root, "valid")
        elif stage == 'test':
            self.test_split = GriffonDataset(self.griffon_root, "test")

    def train_dataloader(self):
        return self.train_split.to_dataloader(self.batch_size, self.num_workers)

    def val_dataloader(self):
        return self.valid_split.to_dataloader(self.batch_size, self.num_workers)

    def test_dataloader(self):
        return self.test_split.to_dataloader(self.batch_size, self.num_workers)

    def teardown(self, stage: Optional[str] = None):
        pass
