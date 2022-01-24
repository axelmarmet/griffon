import torch
import torch.nn as nn

from torch import Tensor

from griffon.constants import NUM_SUB_TOKENS

class Predictor(nn.Module):

    def __init__(self, subtoken_dim:int, vocab_len:int):
        super(Predictor, self).__init__()
        self.subtoken_dim = subtoken_dim
        self.vocab_len = vocab_len

        self.seq = nn.Sequential(
            nn.Linear(subtoken_dim, subtoken_dim),
            nn.ReLU(),
            nn.BatchNorm1d(subtoken_dim),
            nn.Linear(subtoken_dim, vocab_len),
        )

    def forward(self, subtokens:Tensor)->Tensor:

        shape = subtokens.shape
        assert list(shape[-2:]) == [NUM_SUB_TOKENS, self.subtoken_dim]

        # adapt the shape for batch norm that expects shape B x Features
        subtokens = subtokens.view(-1, self.subtoken_dim)
        preds = self.seq.forward(subtokens).view(shape[:-1] + (self.vocab_len,))

        return preds