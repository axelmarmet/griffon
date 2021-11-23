from typing import List, Dict

import pickle

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from griffon.coq_dataclasses import Stage2Sample, Stage2Statement, Stage1Sample
from griffon.dataset.ct_coq_dataset import CTCoqDataset

dataset = CTCoqDataset("../data/small/stage2/train", "../models/vocab.pickle")
sample = dataset[0]

print(sample)