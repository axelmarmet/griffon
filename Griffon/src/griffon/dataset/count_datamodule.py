import os
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

from math import ceil

from griffon.dataset.count_dataset import CounTDataset
from griffon.dataset.semantic_testcase_dataset import SemanticTestCaseDataset


@DATAMODULE_REGISTRY
class CounTDataModule(pl.LightningDataModule):

    def __init__(self, data_root:str, batch_size:int, num_workers:int=4):
        super().__init__()
        self.count_root = os.path.join(data_root, "CounT")
        self.semantic_root = os.path.join(data_root, "semantic_tests")

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage:str):
        assert stage in ['fit', 'validate', 'test', 'predict']
        if stage == 'fit':
            self.train_split = CounTDataset(self.count_root, "train")
            self.count_valid_split = CounTDataset(self.count_root, "valid")
            self.semantic_valid_split = SemanticTestCaseDataset(self.semantic_root)
            self.steps_per_epoch = ceil(len(self.train_split) / self.batch_size)
        elif stage == 'validate':
            self.count_valid_split = CounTDataset(self.count_root, "valid")
            self.semantic_valid_split = SemanticTestCaseDataset(self.semantic_root)
        elif stage == 'test':
            self.test_split = CounTDataset(self.count_root, "test")

    def train_dataloader(self):
        return self.train_split.to_dataloader(self.batch_size, self.num_workers)

    def val_dataloader(self):
        count_val_dataloader = self.count_valid_split.to_dataloader(self.batch_size, self.num_workers)
        semantic_val_dataloader = self.semantic_valid_split.to_dataloader()
        return [count_val_dataloader, semantic_val_dataloader]

    def test_dataloader(self):
        return self.test_split.to_dataloader(self.batch_size, self.num_workers)

    def teardown(self, stage: Optional[str] = None):
        pass
