import torch
import torch.nn as nn

from torch import Tensor

from griffon.constants import NUM_SUB_TOKENS

class TokenDecoder(nn.Module):

    def __init__(self, subtoken_dim:int, token_dim:int, activation_fn:nn.Module):
        super(TokenDecoder, self).__init__()
        self.subtoken_dim = subtoken_dim
        self.token_dim = token_dim

        self.token_decoder = nn.Sequential(
            nn.Linear(token_dim, NUM_SUB_TOKENS * subtoken_dim),
            nn.ReLU(),
            nn.Linear(NUM_SUB_TOKENS * subtoken_dim, NUM_SUB_TOKENS * subtoken_dim),
            activation_fn
        )

    def forward(self, subtokens:Tensor)->Tensor:

        shape = subtokens.shape
        assert shape[-1] == self.token_dim

        # split the final dimension into multiple dimensions for the subtokens
        return self.token_decoder(subtokens).view(shape[:-1] + (NUM_SUB_TOKENS, self.subtoken_dim))