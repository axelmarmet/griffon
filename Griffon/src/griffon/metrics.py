from typing import Optional
import torch
from math import log2

from torch import Tensor
import torch.nn.functional as F
from griffon.constants import TGT_IGNORE_INDEX


def top_k_metric(preds:Tensor, tgt:Tensor, pad_idx:Optional[int] = None, k:int=3):
    number_sub_tokens, len_vocab  = preds.shape
    number_sub_tokens_, = tgt.shape

    assert number_sub_tokens == number_sub_tokens_

    mask = tgt != TGT_IGNORE_INDEX
    if pad_idx is not None:
        mask = torch.logical_and(mask, tgt != pad_idx)

    preds = preds[mask]
    tgt = tgt[mask].unsqueeze(-1)

    assert 0 <= tgt.min().item() and tgt.max().item() < len_vocab

    _, indices = torch.topk(preds, k)
    return torch.sum(indices == tgt, dim=-1, dtype=torch.bool).float()

def relative_entropy(preds:Tensor):
    """Compute the relative entropy of predictions

    Args:
        preds (Tensor): the predictions, shape `num_prediction x vocab len`
    """

    len_vocab = preds.shape[-1]

    max_entropy = - (len_vocab * (1 / len_vocab) * log2(1 /len_vocab))
    probs = F.softmax(preds, dim=-1)
    current_entropy = -(probs * probs.log2()).sum(dim=-1)

    return current_entropy / max_entropy
