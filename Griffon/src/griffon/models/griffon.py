import argparse
import os
import pickle
from numpy import sqrt

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import Dict, Any, Callable, NamedTuple
from torch import Tensor

import json
import math

from torchtext.vocab import Vocab
import wandb
from griffon.constants import TGT_IGNORE_INDEX
from griffon.dataset.count_dataset import CounTDataset
from griffon.dataset.semantic_testcase_dataset import SemanticTestCaseDataset, SemanticTestCases
from griffon.functional.focal_loss import focal_loss
from griffon.metrics import top_k_metric
from griffon.models.cosine_warmup_scheduler import CosineWarmupScheduler
from griffon.models.encoder.code_transformer import CodeTransformer

from griffon.coq_dataclasses import CTCoqOutput, CTCoqSample, CounTBatch, CounTBatchInput
from griffon.models.encoder.standard_transformer import Seq2SeqEncoder
from griffon.preprocessing.stage2.vocab import AbstractVocab
from griffon.utils import cleanup, set_seed, setup

def _get_activation_fn(activation:str)->Callable[[Tensor],Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "sigmoid":
        return F.sigmoid
    else:
        raise ValueError("unknown activation function")


class Griffon(pl.LightningModule):

    def __init__(self, config:Dict[str,Any], learning_rate:float=1.e-4):
        super(Griffon, self).__init__()
        ...
    def forward(self, inp:CTCoqSample)->CTCoqOutput:
        ...

    def training_step(self, batch:CounTBatch, batch_idx):
        ...

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ...

    def configure_optimizers(self):
        ...

    def optimizer_step(self, *args, **kwargs):
        ...