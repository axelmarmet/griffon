from typing import Any, Dict, Optional
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder
import math

from griffon.models.utils import get_norm
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 200):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :]) #type:ignore

# Seq2Seq Network
class Seq2SeqEncoder(nn.Module):
    def __init__(self, config:Dict[str,Any]):

        super(Seq2SeqEncoder, self).__init__()

        d_model = config["encoder_layer"]["d_model"]
        nhead = config["encoder_layer"]["nhead"]
        num_layers = config["num_layers"]
        dim_feedforward = config["encoder_layer"]["dim_feedforward"]
        dropout = 0.1

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        encoder_norm = get_norm(config["norm"])

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.positional_encoding = PositionalEncoding(
            d_model, dropout=dropout)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor]=None,
                src_padding_mask: Optional[Tensor]=None):

        src_emb = self.positional_encoding(src)
        return self.encoder.forward(src_emb, src_mask, src_padding_mask)