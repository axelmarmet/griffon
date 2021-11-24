import os
import pickle

import torch
import torch.nn.functional as F

# for typing
from typing import Callable
from torch import Tensor
from torchtext.vocab import Vocab


def get_activation_fn(activation:str)->Callable[[Tensor],Tensor]:
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

def load_vocab(filename:str)->Vocab:
    assert os.path.exists(filename), f"Path {filename} does not exist"
    return pickle.load(open(filename, "rb"))