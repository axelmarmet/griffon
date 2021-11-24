import torch
from torch import nn
from math import pi


class TransformerPositionalEncoding(nn.Module):

    def __init__(self, d_model, pos_emb_base_pow=10000, **kwargs):
        super(TransformerPositionalEncoding, self).__init__()
        self.pos_emb_base_pow = pos_emb_base_pow
        self.d_model = d_model

    def forward(self, distance_bins):
        device = distance_bins.device
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float, device=device)
        inv_freq = 1 / torch.pow(self.pos_emb_base_pow, (freq_seq / self.d_model))

        batch_size, num_bins = distance_bins.shape
        dists_flat = distance_bins.reshape(-1)

        sinusoid_inp = torch.einsum('i,d->id', dists_flat, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb.reshape([batch_size, num_bins, self.d_model])

        return pos_emb