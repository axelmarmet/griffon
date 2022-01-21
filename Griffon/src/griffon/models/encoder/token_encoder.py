import torch
import torch.nn as nn

from torch import Tensor

from griffon.constants import NUM_SUB_TOKENS

class TokenEncoder(nn.Module):

    def __init__(self, subtoken_dim:int, token_dim:int, activation_fn:nn.Module):
        super(TokenEncoder, self).__init__()
        self.subtoken_dim = subtoken_dim
        self.token_dim = token_dim

        self.token_encoder = nn.Sequential(
            nn.Linear(NUM_SUB_TOKENS * subtoken_dim, NUM_SUB_TOKENS * subtoken_dim),
            nn.ReLU(),
            nn.Linear(NUM_SUB_TOKENS * subtoken_dim, token_dim),
            activation_fn
        )

    def forward(self, subtokens:Tensor)->Tensor:

        shape = subtokens.shape
        assert list(shape[-2:]) == [NUM_SUB_TOKENS, self.subtoken_dim]

        # flatten the subtokens dimensions into one contiguous dimension
        subtokens = subtokens.view(shape[:-2] + (NUM_SUB_TOKENS * self.subtoken_dim,))
        return self.token_encoder(subtokens)