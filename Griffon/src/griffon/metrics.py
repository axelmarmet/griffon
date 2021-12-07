import torch
from torch import Tensor
from griffon.constants import TGT_IGNORE_INDEX


def top_k_metric(preds:Tensor, tgt:Tensor, k:int=3):

    number_sub_tokens, len_vocab  = preds.shape
    number_sub_tokens_, = tgt.shape

    assert number_sub_tokens == number_sub_tokens_

    mask = tgt != TGT_IGNORE_INDEX

    preds = preds[mask]
    tgt = tgt[mask].unsqueeze(-1)

    assert 0 <= tgt.min().item() and tgt.max().item() < len_vocab

    _, indices = torch.topk(preds, k)
    return torch.sum(indices == tgt, dim=-1, dtype=torch.bool).float()
