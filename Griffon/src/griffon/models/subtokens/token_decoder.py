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
            nn.BatchNorm1d(NUM_SUB_TOKENS * subtoken_dim),
            nn.Linear(NUM_SUB_TOKENS * subtoken_dim, NUM_SUB_TOKENS * subtoken_dim),
            activation_fn
        )
        self.final_batch_norm = nn.BatchNorm1d(subtoken_dim)

    def forward(self, subtokens:Tensor)->Tensor:

        shape = subtokens.shape
        assert shape[-1] == self.token_dim

        # adapt the shape for batch norm that expects shape B x Features
        subtokens = subtokens.reshape(-1, self.token_dim)

        # split the final dimension into multiple dimensions for the subtokens
        subtoken_embeddings = self.token_decoder(subtokens).view((-1, self.subtoken_dim))
        return self.final_batch_norm(subtoken_embeddings).view(shape[:-1] + (NUM_SUB_TOKENS, self.subtoken_dim))
