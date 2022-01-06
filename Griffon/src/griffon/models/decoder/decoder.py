from typing import Optional
import copy

import torch
from torch import nn
from torch.nn import MultiheadAttention, LayerNorm
from torch.nn.modules.sparse import Embedding

import torch.nn.functional as F

from typing import Dict, Any
from torch import Tensor

from griffon.constants import NUM_SUB_TOKENS
from griffon.models.decoder.simple_multihead_attention import SimpleMHA
from griffon.models.encodings.positional_encoding import PositionalEncoding

class DecoderLayer(nn.Module):

    def __init__(self, d_model:int, n_head:int, dim_feedforward:int, dropout:float, layer_norm_eps:float=1e-5):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, n_head, dropout=dropout)

        self.title_attn = SimpleMHA(d_model, n_head)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, memory_key_padding_mask: Tensor, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required). shape `num_statement x max_num_tokens x d_model`
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        S, NUM_STATEMENT, E_ = memory.shape
        L, B, E = tgt.shape
        assert B == 1, 'batching not supported for size larger than 1'
        assert E == E_

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # each statement has for first token a title (for example IH or Goal)
        # we compute an attention map across statements using this token
        titles = memory[:,0].unsqueeze(1) # shape `num_statements x 1 x e_dim`
        statement_attention_dist = self.title_attn.forward(query=tgt, key=titles).permute(1,2,0) # shape `tgt_len x num_statements x 1`

        # we flatten the memory so that is has batch size 1
        memory = memory.view(S*NUM_STATEMENT, 1, E)
        memory_key_padding_mask = memory_key_padding_mask.view(1,S*NUM_STATEMENT)

        statement_outputs = self.multihead_attn.forward(query=tgt, key=memory, value=memory, attn_mask=memory_mask,
                                           key_padding_mask=memory_key_padding_mask)[0] # shape `tgt_len x num_statements x e_dim`
        tgt2 = (statement_attention_dist * statement_outputs).sum(dim=1, keepdim=True)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Decoder(nn.Module):

    def __init__(self, decoder_layer:DecoderLayer, n_layers:int, dropout:float):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer)
         for _ in range(n_layers)])

        self.positional_encoding = PositionalEncoding(decoder_layer.d_model, dropout)

    def forward(self,
                memory: Tensor,
                tgt: Tensor,
                tgt_mask: Tensor,
                memory_key_padding_mask: Tensor) -> torch.Tensor:
        """
        Args:
            - memory: :math:`(S, N, E)`
            - tgt: :math:`(T, 1, E)`
            - tgt_mask: :math:`(T, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

        Returns:
            - output: :math:`(T, 1, E)`
        """
        output = self.positional_encoding(tgt)

        for mod in self.layers: #type: DecoderLayer
            output = mod.forward(output, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)

        return output
