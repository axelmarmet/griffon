import torch
import torch.nn.functional as F
from torch import nn

class SimpleMHA(nn.Module):

    def __init__(self, embed_dim:int, n_head:int):
        super(SimpleMHA, self).__init__()
        self.register_buffer("dummy_values", torch.zeros((1,1,embed_dim)))
        self.mha = nn.MultiheadAttention(embed_dim, n_head)

    def forward(self,
                query:torch.Tensor,
                key:torch.Tensor)->torch.Tensor:

        S, B = key.shape[:2]
        value = self.dummy_values.expand(S, B, -1) # type: ignore

        return self.mha.forward(query, key, value)[1] #type:ignore

    def unbatched_forward(self,
                query:torch.Tensor,
                key:torch.Tensor)->torch.Tensor:

        S = key.shape[0]

        return self.mha.forward(query=query.unsqueeze(1),
                                key=key.unsqueeze(1),
                                value=self.dummy_values.expand(S, -1, -1))[1].squeeze(0) #type:ignore


