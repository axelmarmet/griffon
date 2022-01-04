from typing import Optional

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

    def __init__(self, d_model:int, n_head, dim_feedforward, dropout, layer_norm_eps=1e-5):
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

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
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
        L, B, E = tgt.shape
        assert B == 1, 'batching not supported for size larger than 1'


        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # each statement has for first token a title (for example IH or Goal)
        # we compute an attention map across statements using this token
        titles = memory[:,0].unsqueeze(1) # shape `num_statements x 1 x e_dim`
        statement_attention_dist = self.title_attn.forward(query=tgt, key=titles).permute(1,2,0) # shape `tgt_len x num_statements x 1`

        # pretty ugly hack, normally we don't use batch, but we use the batch dim
        # to process all statements at once
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

    def __init__(self, embedding: Embedding, config: Dict[str, Any]):
        super(Decoder, self).__init__()

        self.output_subtokens_per_token = NUM_SUB_TOKENS

        self.d_model = config["d_model"]
        self.n_layers = config["n_layers"]

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model = config["d_model"],
                n_head = config["n_head"],
                dim_feedforward = config["dim_feedforward"],
                dropout = config["dropout"],
                layer_norm_eps= config["layer_norm_eps"]
            )
         for _ in range(self.n_layers)])


        self.token_embedding = embedding
        self.positional_encoding = PositionalEncoding(config["d_model"], config["decoder_dropout"])

    def forward(self,
                memory: Tensor,
                target: Tensor,
                memory_key_padding_mask: Optional[Tensor]) -> torch.Tensor:

        src_emb = self.token_embedding(target)
        src_emb = self.positional_encoding(src_emb)

        output = src_emb

        for mod in self.layers: #type: DecoderLayer
            output = mod.forward(output, memory, memory_key_padding_mask=memory_key_padding_mask)

        return output
