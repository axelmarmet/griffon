import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

# for typing
from typing import Any, Callable, Dict
from torch import Tensor
from torchtext.vocab import Vocab



def load_vocab(filename:str)->Vocab:
    assert os.path.exists(filename), f"Path {filename} does not exist"
    return pickle.load(open(filename, "rb"))

def get_norm(config:Dict[str, Any]):
    assert config["type"] == "layer_norm"
    return nn.LayerNorm(config["d_model"], config["eps"])