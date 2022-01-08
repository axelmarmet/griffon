from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

from griffon.models.decoder.simple_multihead_attention import SimpleMHA

class PointerNetwork(nn.Module):

    def __init__(self, subtoken_dim:int, num_heads:int):
        super(PointerNetwork, self).__init__()

        self.subtoken_dim = subtoken_dim
        self.linear = nn.Linear(subtoken_dim, 1)
        self.mha = SimpleMHA(subtoken_dim, num_heads)

    def forward(self,
                logits:torch.Tensor,
                extended_vocab_ids:torch.Tensor,
                src_subtokens:torch.Tensor,
                src_padding:torch.Tensor,
                tgt_subtokens:torch.Tensor,
                len_vocab: int,
                max_len_extended_vocab: int)->torch.Tensor:
        """

        Args:
            logits (torch.Tensor): shape `batch, tgt_subtokens, vocab_len`
            extended_vocab_ids (torch.Tensor): shape `batch, src_subtokens`
            src_subtokens (torch.Tensor): shape `batch, src_subtokens, subtoken_dim`
            src_padding (torch.Tensor): shape `batch, src_subtokens, subtoken_dim`
            tgt_subtokens (torch.Tensor): shape `batch, tgt_subtokens, subtoken_dim`
            len_vocab (int): the length of the fixed vocab
            max_len_extended_vocab (int): the max length of the fixed vocab
        """

        # shape validation
        assert logits.shape[0] == tgt_subtokens.shape[0]

        assert logits.shape[1] == tgt_subtokens.shape[1]
        assert logits.shape[2] == len_vocab
        assert src_subtokens.shape[1] == extended_vocab_ids.shape[1]
        assert src_subtokens.shape[2] == tgt_subtokens.shape[2]

        DEVICE = logits.device

        B = logits.shape[0]
        SRC_LEN = src_subtokens.shape[1]
        TGT_LEN = tgt_subtokens.shape[1]

        # note that copy prob is NOT a log prob
        copy_prob = torch.sigmoid(self.linear(tgt_subtokens))

        # standard transpose to fit the usual S x B x E format
        tgt_subtokens = tgt_subtokens.permute(1,0,2)
        src_subtokens = src_subtokens.permute(1,0,2)

        attention_distribution = self.mha.forward(query=tgt_subtokens,
                                                  key=src_subtokens,
                                                  key_padding_mask=src_padding)

        #######################################
        # shadiest part of the whole codebase #
        #######################################

        # Sum up probabilities of the same tokens
        M = torch.zeros((B, max_len_extended_vocab + len_vocab, SRC_LEN), device=DEVICE)
        M[
            torch.arange(B).unsqueeze(-1).expand(B, SRC_LEN).reshape(-1),
            extended_vocab_ids.view(-1),
            torch.arange(SRC_LEN).repeat(B)] = 1
        M = M.permute(0,2,1)
        attention = torch.bmm(attention_distribution, M).squeeze()

        attention[attention == 0] = torch.finfo(torch.float).eps
        pointer_log_probs= attention.log()
        # Avoid having -inf in attention scores as they produce NaNs during backward pass
        # pointer_log_probs[pointer_log_probs == -float('inf')] = torch.finfo(torch.float).min

        if torch.isnan(pointer_log_probs).any():
            print("NaN in final pointer attention!", pointer_log_probs)

        # Probability distribution of the decoder over original vocabulary
        decoder_log_probs = F.log_softmax(logits, dim=-1)
        # Decoder cannot predict extended vocabulary tokens. Thus, these have 0 probability
        decoder_log_probs = F.pad(decoder_log_probs, [0, max_len_extended_vocab], value=-float('inf'))

        # Combine decoder probability distribution with pointer attention distribution in log space
        p = torch.stack([decoder_log_probs + (1 - copy_prob).log(),
                         pointer_log_probs + copy_prob.log()])
        p[p == -float('inf')] = torch.finfo(torch.float).min
        log_probs = torch.logsumexp(p, dim=0)

        assert not torch.isnan(log_probs).any()

        return log_probs

