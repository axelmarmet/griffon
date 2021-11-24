import torch
import torch.nn as nn

from typing import Dict, Any
from torch import Tensor

from griffon.models.utils import load_vocab

class CounT(nn.Module):
    def __init__(self, config:Dict[str,Any]):

        self.embedding_dim = config["embedding_dim"]
        self.num_subtokens = config["num_subtokens"]

        self.vocab = load_vocab(config["vocab"])

        self.token_encoder = nn.Linear(self.num_subtokens * self.embedding_dim, self.embedding_dim)
        self.token_decoder = nn.Linear(self.embedding_dim, self.num_subtokens * self.embedding_dim)

        self.ct_encoder =
