from typing import List, Optional
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

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, memory_key_padding_mask: Tensor,
                tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required). shape `L x B x E`
            memory: the sequence from the last layer of the encoder (required). shape `B x NUM_STATEMENTS x S x E`
            tgt_mask: the mask for the tgt sequence (optional). shape `L x L`
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional). shape `B x L`
            memory_key_padding_mask: the mask for the memory keys per batch (optional). shape `B x NUM_STATEMENTS x S`

        Return
            Tensor: shape `B x L x E`
        """
        B, NUM_STATEMENTS, S, E = memory.shape
        L = tgt.shape[0]

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # each statement has for first token a title (for example IH or Goal)
        # we compute an attention map across statements using this token
        titles = memory[:,:,0,:].permute(1,0,2)
        titles_padding = memory_key_padding_mask[:,:,0]

        statement_attention_dist = self.title_attn.forward(query=tgt, key=titles, key_padding_mask=titles_padding) # shape B x L x NUM_STATEMENT
        # massage the shape into something that will broadcast for the reduction
        statement_attention_dist = statement_attention_dist.permute(1,0,2).unsqueeze(2)

        statement_outputs : List[Tensor] = []
        for i in range(NUM_STATEMENTS):
            ith_statement_tokens = memory[:,i,:,:].permute(1,0,2)
            ith_statement_padding = memory_key_padding_mask[:,i,:]

            fully_padded_mask = ith_statement_padding[:,0].unsqueeze(0).expand((L ,-1))

            # hack so that the MHA does not get any NaN
            first_token_mask = torch.ones_like(ith_statement_padding)
            first_token_mask[:,0] = False
            final_padding_mask = torch.logical_and(first_token_mask, ith_statement_padding)

            statement_output = self.multihead_attn.forward(query=tgt, key=ith_statement_tokens, value=ith_statement_tokens, attn_mask=None,
                                           key_padding_mask=final_padding_mask)[0]

            statement_output[fully_padded_mask] = 0

            statement_outputs.append(statement_output)

        output = torch.stack(statement_outputs, dim=-1) # shape : L x B x E x NUM_STATEMENTS
        tgt2 = (statement_attention_dist * output).sum(dim=-1) # shape : L x B x E

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
                tgt_key_padding_mask: Tensor,
                memory_key_padding_mask: Tensor) -> torch.Tensor:
        """
        Args:
            - memory: :math:`B x NUM_STATEMENTS x S x E`
            - tgt: :math:`B x L x E`
            - tgt_mask: :math:`L x L`.
            - tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).  `B x L`
            - memory_key_padding_mask: :math:`B x NUM_STATEMENTS x S`

        Returns:
            - output: :math:`(B, L, E)`
        """
        output = self.positional_encoding(tgt)

        # transpose the input to fit the L x B x E shape
        output = output.permute(1,0,2)


        for mod in self.layers: #type: DecoderLayer
            output = mod.forward(output, memory,
                                 tgt_mask=tgt_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask)

        # transpose the output back to fit the B x L x E shape
        output = output.permute(1,0,2)

        return output
