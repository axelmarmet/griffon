import math
from typing import Optional

import torch
from torch import nn
from torch.nn import TransformerDecoder, MultiheadAttention, LayerNorm
from torch.nn.init import xavier_uniform_
from torch.nn.modules.sparse import Embedding
from torch.nn.modules.transformer import TransformerDecoderLayer

from typing import Dict, Any
from torch import Tensor

from code_transformer.configuration.transformer_lm_decoder import TransformerDecoderConfig
from code_transformer.modeling.code_transformer.code_transformer import _get_activation_fn
from code_transformer.modeling.code_transformer.lm import CodeTransformerOutput

from griffon.constants import NUM_SUB_TOKENS
from griffon.models.decoder.pointer import PointerNetwork
from griffon.models.utils import _get_activation_fn

from code_transformer.utils.data import batch_index_select


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, position):
        x = x + self.pe[position, :]
        return self.dropout(x)


class Decoder(nn.Module):

    def __init__(self, embedding: Embedding, config: Dict[str, Any]):
        super(Decoder, self).__init__()

        self.sos_id = config["sos_id"]
        self.unk_id = config["unk_id"]
        self.pad_id = config["pad_id"]

        self.output_subtokens_per_token = NUM_SUB_TOKENS

        self.vocab_size = config["vocab_size"]

        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.n_layers = config["n_layers"]

        self.output_nonlinearity = _get_activation_fn(
            config["output_nonlinearity"])

        # we intentionally don't ignore pad index for now
        # could use label smoothing loss
        self.loss_fct = nn.CrossEntropyLoss()

        decoder_layer = TransformerDecoderLayer(
            self.d_model,
            self.n_heads,
            config["decoder_dim_feedforward"],
            config["decoder_dropout"],
            _get_activation_fn(config["decoder_activation"]))

        self.transformer_decoder = TransformerDecoder(
            decoder_layer, self.n_layers)

        self.positional_encoding = PositionalEncoding(
            self.d_model, config["decoder_dropout"])

        self.pointer_network = PointerNetwork(
            self.d_model,
            NUM_SUB_TOKENS,
            self.n_heads
        )

        self.token_embedding = embedding

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self,
                memory: Tensor,
                labels: Tensor,
                attention_pad_mask: Tensor,
                extended_vocabulary_ids=None,
                pointer_pad_mask: Optional[torch.Tensor] = None,
                **model_input) -> CodeTransformerOutput:
        """
        :param memory: torch.Tensor [Batch, Statements, Src_Seq, E_Dim]
        :param labels: torch.Tensor [Batch, Tgt_Seq, sub_tokens]
        :param attention_pad_mask: torch.Tensor [Batch, Tgt_Seq]
        :param extended_vocabulary_ids: torch.Tensor [B, subtoken_seq_len]
            Defines a sequence of subtokens for every sample. Can be seen as a flattened version of the tokens input
            with UNKNOWN_TOKENs replaced by incremental artificial vocabulary IDs that are only valid for this sample.
            Needed for the pointer mechanism
        :param pointer_pad_mask: torch.Tensor [B, S, num_subtokens]
            A mask that specifies padding down to subtoken level. Needed for the pointer mechanism as we need to point
            to distinct subtokens. 1 indicates that the respective subtoken is NOT a PAD token
        :param model_input:
            additional inputs that are passed on to the encoder.
            The TransformerDecoder expects that model_input has at least the following entries:
                - pad_mask: [B, S], where 1 indicates that the position is a PAD token
                - attention_mask: [B, S, S], where a 1 at [:,i,j] indicates that position i may not attend position j
        :return:
        """

        assert memory.shape[0] == 1, "Batch size of 1 is the only size supported for now"

        device = next(self.parameters()).device
        pointer_gates = None
        pointer_attentions = None
        pointer_attention_distributions = None

        B = memory.shape[0]
        STATEMENTS = memory.shape[1]
        SRC_SEQ = memory.shape[2]

        TGT_SEQ = labels.shape[1]

        V = self.vocab_size
        D = self.d_model

        # Initially start decoding with a sequence containing only one <s> token per sample
        # Input tokens should have B x T x D
        # every batch decoding starts with the same initial input
        initial_input = torch.tensor([[self.sos_id]], device=device)

        decoder_input = self.token_embedding(initial_input).expand((B, -1, -1))
        decoder_input = self.positional_encoding.forward(decoder_input, 0)

        pointer_input_subtokens = memory

        self.pointer_network.init_batch(pointer_input_subtokens, pointer_pad_mask, extended_vocabulary_ids,
                                        self.vocab_size)

        logits = torch.zeros(
            (TGT_SEQ,
             self.output_subtokens_per_token,
             B,
             self.pointer_network.len_extended_vocab),
            device=device)

        pointer_gates = torch.zeros((self.output_subtokens_per_token, B))
        pointer_attentions = torch.zeros(
            (self.output_subtokens_per_token, B, self.pointer_network.len_extended_vocab))
        pointer_attention_distributions = torch.zeros(
            (self.output_subtokens_per_token, B,
             extended_vocabulary_ids.shape[1])
        )

        # pad_mask has 1s for all regular (non-pad) tokens
        # attention_mask has 1s for all illegal tokens that may not be attended (such as function name and CLS token)
        pad_mask = model_input['pad_mask'].bool()

        # REMOVED
        # for now all tokens can attend to all tokens

        for token_idx in range(labels.shape[1]):
            for subtoken_idx in range(self.output_subtokens_per_token):

                pointer_query = decoder_input.select(1, -1)

                # REMOVED
                # can use self attention on the query

                self.pointer_network.calculate_pointer_attention(pointer_query)

                decoder_output = self.transformer_decoder.forward(decoder_input.transpose(0, 1),
                                                                  memory.transpose(
                                                                      0, 1),
                                                                  memory_key_padding_mask=attention_pad_mask)

                decoder_output = self.output_nonlinearity(decoder_output)

                # B x V
                subtoken_logits = decoder_output.select(
                    0, -1) @ self.token_embedding.weight.T

                subtoken_logits = self.pointer_network.combine_probabilites(
                    subtoken_logits)
                pointer_gates[subtoken_idx] = self.pointer_network.pointer_gate.squeeze(
                    -1)
                pointer_attentions[subtoken_idx] = self.pointer_network.pointer_attention
                pointer_attention_distributions[subtoken_idx] = self.pointer_network.pointer_attention_distribution

                logits[subtoken_idx] = subtoken_logits

                # Calculate next decoder_input
                # Use previous label as next input
                next_input = labels[:, :, subtoken_idx]  # B x 1

                next_input = self.pointer_network.get_next_input(
                    next_input, self.unk_id)

                next_input_embedding = self.token_embedding(next_input)
                next_input_embedding = self.positional_encoding.forward(
                    next_input_embedding, subtoken_idx + 1)
                next_input = torch.cat(
                    [decoder_input, next_input_embedding], 1)
                decoder_input = next_input

        loss = self.loss_fct(logits.transpose(
            0, 1).reshape(-1, logits.size(-1)), labels.view(-1))

        logits = logits.transpose(0, 1).unsqueeze(
            1)  # B x 1 x output_subtokens x V
        logits = logits.reshape(B // n_predict, n_predict,
                                logits.shape[2], logits.shape[3])
        outputs = CodeTransformerOutput(loss=loss,
                                        logits=logits,
                                        attentions=transformer_output.attentions,
                                        pointer_gates=pointer_gates,
                                        pointer_attentions=pointer_attentions,
                                        pointer_attention_distributions=pointer_attention_distributions)

        return outputs
