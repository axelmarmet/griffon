import os
import pickle

import torch
import torch.nn.functional as F

# for typing
from typing import Callable
from torch import Tensor
from torchtext.vocab import Vocab



def load_vocab(filename:str)->Vocab:
    assert os.path.exists(filename), f"Path {filename} does not exist"
    return pickle.load(open(filename, "rb"))