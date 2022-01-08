import torch
import torch.nn.functional as F

def focal_loss(inp, tgt, gamma, ignore_index:int = -100):
    CE_loss = F.cross_entropy(inp, tgt, ignore_index=ignore_index)
    pt = torch.exp(-CE_loss)
    F_loss = (1 - pt) ** gamma * CE_loss
    return F_loss.mean()

def focal_loss_from_log_probs(inp, tgt, gamma, ignore_index:int = -100):
    CE_loss = F.nll_loss(inp, tgt, ignore_index=ignore_index)
    pt = torch.exp(-CE_loss)
    F_loss = (1 - pt) ** gamma * CE_loss
    return F_loss.mean()