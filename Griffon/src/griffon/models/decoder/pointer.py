from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

from griffon.models.decoder.simple_multihead_attention import SimpleMHA

class PointerNetwork(nn.Module):

    def __init__(self, subtoken_dim:int, num_heads:int, len_vocab:int):
        super(PointerNetwork, self).__init__()

        self.subtoken_dim = subtoken_dim
        self.len_vocab = len_vocab
        self.linear = nn.Linear(subtoken_dim, 1)
        self.mha = SimpleMHA(subtoken_dim, num_heads)

    def forward(self,
                logits:torch.Tensor,
                extended_vocab_ids:torch.Tensor,
                src_subtokens:torch.Tensor,
                tgt_subtokens:torch.Tensor,
                len_extended_vocab: int)->torch.Tensor:
        """

        Args:
            logits (torch.Tensor): shape `tgt_subtokens, vocab_len`
            extended_vocab_ids (torch.Tensor): shape `src_subtokens`
            src_subtokens (torch.Tensor): shape `src_subtokens, subtoken_dim`
            tgt_subtokens (torch.Tensor): shape `tgt_subtokens, subtoken_dim`
            len_extended_vocab (int): the length of the extended vocab
        """
        SRC_LEN = src_subtokens.shape[0]
        TGT_LEN = tgt_subtokens.shape[0]

        # note that copy prob is NOT a log prob
        copy_prob = torch.sigmoid(self.linear(tgt_subtokens))

        attention_distribution = self.mha.unbatched_forward(query=tgt_subtokens,
                                                            key=src_subtokens)

        # Sum up probabilities of the same tokens
        M = torch.zeros((SRC_LEN,len_extended_vocab))
        M[torch.arange(SRC_LEN), extended_vocab_ids] = 1

        attention = attention_distribution @ M
        pointer_log_probs= attention.log()

        # Probability distribution of the decoder over original vocabulary
        decoder_log_probs = F.log_softmax(logits, dim=1)
        # Decoder cannot predict extended vocabulary tokens. Thus, these have 0 probability
        decoder_log_probs = F.pad(decoder_log_probs, [0, len_extended_vocab, self.len_vocab], value=-float('inf'))

        # Combine decoder probability distribution with pointer attention distribution in log space
        p = torch.stack([decoder_log_probs + (1 - copy_prob).log(),
                         pointer_log_probs + copy_prob.log()])
        log_probs = torch.logsumexp(p, dim=0)

        assert not torch.isnan(log_probs).any()

        return log_probs

