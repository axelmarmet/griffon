from pytorch_lightning.utilities.cli import LightningCLI

from griffon.models.encoder.count import CounT
from griffon.dataset.count_dataset import CounTDataModule
from griffon.dataset.semantic_testcase_dataset import SemanticTestCaseDataset

LightningCLI(CounT, CounTDataModule)