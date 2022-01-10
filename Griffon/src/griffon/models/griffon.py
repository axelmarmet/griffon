import argparse
import os
import pickle
from numpy import sqrt

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import Dict, Any, Callable, List, NamedTuple
from torch import Tensor

import json
import math

from torchtext.vocab import Vocab
import wandb
from griffon.constants import NUM_SUB_TOKENS, TGT_IGNORE_INDEX, UNK_TOKEN
from griffon.dataset.count_dataset import CounTDataset
from griffon.dataset.semantic_testcase_dataset import SemanticTestCaseDataset, SemanticTestCases
from griffon.functional.focal_loss import focal_loss, focal_loss_from_log_probs
from griffon.metrics import top_k_metric
from griffon.models.cosine_warmup_scheduler import CosineWarmupScheduler
from griffon.models.decoder.decoder import Decoder
from griffon.models.decoder.pointer import PointerNetwork
from griffon.models.encoder.code_transformer import CodeTransformer

from griffon.coq_dataclasses import CTCoqOutput, GriffonBatch, GriffonSample, CounTBatch, CounTBatchInput
from griffon.models.encoder.count import CounT
from griffon.models.encoder.standard_transformer import Seq2SeqEncoder
from griffon.preprocessing.stage2.vocab import AbstractVocab
from griffon.utils import cleanup, set_seed, setup

class Griffon(pl.LightningModule):

    def __init__(self, count_ckpt:str, decoder:Decoder, pointer:PointerNetwork, optimizer_config:Dict[str,Any], learning_rate:float=1.e-4):
        super(Griffon, self).__init__()

        self.save_hyperparameters(optimizer_config)
        self.learning_rate = learning_rate

        self.decoder = decoder

        # we extract some modules from the pretraining task CounT
        count:CounT = CounT.load_from_checkpoint(count_ckpt)

        # sanity check
        assert self.decoder.layers[0].d_model == count.d_model

        self.vocab = count.vocab
        self.unk_id = self.vocab[UNK_TOKEN]
        self.vocab_len = len(self.vocab)

        self.subtoken_embedding_dim = count.subtoken_embedding_dim
        self.token_embedding_dim = count.d_model

        self.embedding = count.embedding

        self.token_encoder = count.token_encoder
        self.token_decoder = count.token_decoder

        self.ct_encoder = count.encoder
        self.predictor = count.predictor

        del count

        self.pointer = pointer

    def get_tgt_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def forward(self, inp:GriffonBatch)->Tensor:

        ################################################################
        # we are overcautious/inneficient and reshape frequently to make
        # sure no mistake slips by
        ################################################################

        NUM_DISTANCES = inp.distances_indices.shape[2]
        B, NUM_STATEMENTS, S = inp.statements.shape[:3]
        L = inp.lemmas.shape[1]
        E = self.token_embedding_dim
        S_E = self.subtoken_embedding_dim

        def process_statement_subtokens(statements:Tensor)->Tensor:
            """Takes statements subtoken ids and return statement tokens

            Args:
                statements (Tensor): shape `B x NUM_STATEMENTS x S x SUBTOKENS`

            Returns:
                Tensor: shape `B x NUM_STATEMENTS x S x E
            """

            assert list(statements.shape) == [B, NUM_STATEMENTS, S, NUM_SUB_TOKENS]

            statement_subtokens_embeddings:Tensor = \
                self.embedding.forward(statements).view(B, NUM_STATEMENTS, S, self.subtoken_embedding_dim * NUM_SUB_TOKENS) # type: ignore
            return self.token_encoder(statement_subtokens_embeddings)

        def process_lemma_subtokens(lemmas:Tensor)->Tensor:
            """Takes lemma subtoken ids and return statement tokens

            Args:
                lemmas (Tensor): shape `B x L x SUBTOKENS`

            Returns:
                Tensor: shape `B x L-1 x E
            """
            assert list(lemmas.shape) == [B, L, NUM_SUB_TOKENS]

            # we discard the <eos> token from the input
            lemmas = lemmas[:,:-1]

            # giving the embedding of copied subtoken would probably be better
            lemmas[lemmas >= self.vocab_len] = self.unk_id
            # we don't care about the -1 vocabs, we just don't want an OOB exception in the embedding
            lemmas[lemmas == TGT_IGNORE_INDEX] = self.unk_id

            lemma_subtokens_embeddings:Tensor = \
                self.embedding.forward(lemmas).view(B, L-1, self.subtoken_embedding_dim * NUM_SUB_TOKENS) # type: ignore
            return self.token_encoder(lemma_subtokens_embeddings)

        def encode_statement_tokens(statement_tokens:Tensor,
                                    statement_token_padding:Tensor,
                                    distances_indices:Tensor,
                                    distances_bins:Tensor)->Tensor:
            """passe the statement tokens through the CT encoder

            Args:
                statement_tokens (Tensor): shape `B x NUM_STATEMENTS x S x E`
                statement_token_padding (Tensor): shape `B x NUM_STATEMENTS x S`
                distances_indices (Tensor): shape `B x NUM_STATEMENTS x DISTANCES x S x S`
                distances_bins (Tensor): shape `B x NUM_STATEMENTS x DISTANCES x BINS
            Returns:
                Tensor: shape `B x NUM_STATEMENTS x S x E`
            """

            # shape validation
            assert list(statement_tokens.shape) == [B, NUM_STATEMENTS, S, E]
            assert list(statement_token_padding.shape) == [B, NUM_STATEMENTS, S]
            assert list(distances_indices.shape) == [B, NUM_STATEMENTS, 4, S, S]
            assert list(distances_bins.shape[:2]) == [B, NUM_STATEMENTS]

            # we flatten B * NUM_STATEMENT into a bigger batch dimension
            # we don't create a bigger sequence because that would require to
            # change the size of the distance indices
            statement_token_embeddings = statement_tokens.reshape(
                B * NUM_STATEMENTS, S, self.token_embedding_dim)
            statement_token_padding = statement_token_padding.reshape(
                B * NUM_STATEMENTS, S
            )
            relative_distances = distances_indices.reshape(B * NUM_STATEMENTS, NUM_DISTANCES, S, S), \
                                distances_bins.reshape(B * NUM_STATEMENTS, NUM_DISTANCES, -1)

            # we feed those flattened tensor to the encoder
            statement_token_embeddings = self.ct_encoder.forward(
                statement_token_embeddings,
                src_key_padding_mask=statement_token_padding,
                relative_distances=relative_distances).permute(1,0,2)

            # sanity check
            assert statement_token_embeddings.shape[0] == B * NUM_STATEMENTS
            assert statement_token_embeddings.shape[1] == S
            assert statement_token_embeddings.shape[2] == self.token_embedding_dim

            return statement_token_embeddings.reshape(B, NUM_STATEMENTS, S, self.token_embedding_dim)

        def extract_statement_subtokens(statement_tokens:Tensor)->Tensor:
            """ Go from tokens to subtokens

            Args:
                statement_tokens (Tensor): shape `B x NUM_STATEMENTS x S x E`

            Returns:
                Tensor: shape `B x NUM_STATEMENTS x S x NUM_SUBTOKENS x SUBTOKEN_DIM`
            """
            # shape validation
            assert list(statement_tokens.shape) == [B, NUM_STATEMENTS, S, E]

            return self.token_decoder(statement_tokens).view(B, NUM_STATEMENTS, S, NUM_SUB_TOKENS, self.subtoken_embedding_dim)

        def extract_lemma_subtokens(lemma_tokens:Tensor)->Tensor:
            """ Go from tokens to subtokens

            Args:
                lemma_tokens (Tensor): shape `B x L-1 x E`

            Returns:
                Tensor: shape `B x L-1 x NUM_SUBTOKENS x SUBTOKEN_DIM`
            """
            # shape validation
            assert list(lemma_tokens.shape) == [B, L-1, E]

            return self.token_decoder(lemma_tokens).view(B, L-1, NUM_SUB_TOKENS, self.subtoken_embedding_dim)

        def decode_lemma_tokens(statement_tokens:Tensor,
                        statement_padding:Tensor,
                        lemma_tokens:Tensor,
                        lemma_padding:Tensor)->Tensor:
            """Use the decoder

            Args:
                statement_tokens (Tensor): shape B x NUM_STATEMENTS x S x E
                statement_padding (Tensor): shape B x NUM_STATEMENTS x S
                lemma_tokens (Tensor): shape B x L-1 x E
                lemma_padding (Tensor): shape B X L

            Returns:
                Tensor: B x L-1 x E
            """

            assert list(statement_tokens.shape) == [B, NUM_STATEMENTS, S, E]
            assert list(statement_padding.shape) == [B, NUM_STATEMENTS, S]
            assert list(lemma_tokens.shape) == [B, L-1, E]
            assert list(lemma_padding.shape) == [B, L]

            # we adjust the padding
            lemma_padding = lemma_padding[:,:-1]

            return self.decoder.forward(memory=statement_tokens,
                                                            tgt=lemma_tokens,
                                                            tgt_mask=self.get_tgt_mask(L-1),
                                                            tgt_key_padding_mask=lemma_padding,
                                                            memory_key_padding_mask=statement_padding)

        def get_predictions(lemma_subtokens:Tensor,
                            statement_subtokens:Tensor,
                            extended_vocab_ids:Tensor,
                            statement_padding:Tensor,
                            extended_vocabularies:List[Dict[str, int]])->Tensor:
            """Use the predictor and pointer to get log probs
               over the extended vocabularies

            Args:
                lemma_subtokens (Tensor): shape `B x L-1 x NUM_SUB_TOKENS x S_E`
                statement_subtokens (Tensor): shape `B x NUM_STATEMENTS x S x NUM_SUB_TOKENS x S_E`
                extended_vocab_ids (Tensor): shape `B x NUM_STATEMENTS x S x NUM_SUB_TOKENS`
                statement_padding (Tensor): shape `B x NUM_STATEMENTS x S`
                extended_vocabularies (List[Dict[int, str]]):

            Returns:
                Tensor: shape `B x L-1 x LEN_EXTENDED_VOCAB`
            """

            assert list(lemma_subtokens.shape) == [B, L-1, NUM_SUB_TOKENS, S_E]
            assert list(statement_subtokens.shape) == [B, NUM_STATEMENTS, S, NUM_SUB_TOKENS, S_E]
            assert list(extended_vocab_ids.shape) == [B, NUM_STATEMENTS, S, NUM_SUB_TOKENS]
            assert list(statement_padding.shape) == [B, NUM_STATEMENTS, S]
            assert len(extended_vocabularies) == B

            # we brutally expand the statement padding mask so that it is at the subtoken granularity
            statement_padding = statement_padding.unsqueeze(-1).expand(-1, -1, -1, NUM_SUB_TOKENS)

            # we flatten the tensors to fit the `batch x total_num_tokens` shape expected by the pointer network
            lemma_subtokens = lemma_subtokens.reshape(B, (L-1) * NUM_SUB_TOKENS, S_E)
            extended_vocab_ids = extended_vocab_ids.view(B, NUM_STATEMENTS * S * NUM_SUB_TOKENS)
            statement_subtokens = statement_subtokens.reshape(B, NUM_STATEMENTS * S * NUM_SUB_TOKENS, S_E)
            statement_padding = statement_padding.reshape(B, NUM_STATEMENTS * S * NUM_SUB_TOKENS)

            # we first get the predictions without using the pointer network
            logits:Tensor = self.predictor(lemma_subtokens)

            # we combine the predictions with the one from the pointer network
            log_probs = self.pointer.forward(
                logits = logits,
                extended_vocab_ids = extended_vocab_ids,
                src_subtokens = statement_subtokens,
                src_padding = statement_padding,
                tgt_subtokens = lemma_subtokens,
                len_vocab = self.vocab_len,
                max_len_extended_vocab=max(len(extended_vocab) for extended_vocab in inp.extended_vocabularies))

            return log_probs

        statement_tokens = process_statement_subtokens(inp.statements)
        statement_tokens = encode_statement_tokens(statement_tokens,
                                                   inp.statement_token_padding,
                                                   inp.distances_indices,
                                                   inp.distances_bins)

        lemma_tokens = process_lemma_subtokens(inp.lemmas)
        lemma_tokens = decode_lemma_tokens(statement_tokens,
                                           inp.statement_token_padding,
                                           lemma_tokens,
                                           inp.lemma_token_padding)

        statement_subtokens = extract_statement_subtokens(statement_tokens)
        lemma_subtokens = extract_lemma_subtokens(lemma_tokens)

        log_probs = get_predictions(lemma_subtokens,
                                    statement_subtokens,
                                    inp.extended_vocabulary_ids,
                                    inp.statement_token_padding,
                                    inp.extended_vocabularies)
        return log_probs

    def training_step(self, batch:GriffonBatch, batch_idx):
        B = batch.lemmas.shape[0]

        preds = self.forward(batch)
        len_vocab = preds.shape[-1]
        # we shift the targets by 1 since <sos> should predict the first actual token
        # and the last actual token should predict <eos>
        tgts = batch.lemmas[:,1:]
        loss = focal_loss_from_log_probs(
            preds.view(-1, len_vocab),
            tgts.reshape(-1),
            gamma=self.hparams["loss_fn"]["gamma"],
            ignore_index=TGT_IGNORE_INDEX
        )
        self.log("training_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch:GriffonBatch, batch_idx, dataloader_idx=0):
        B = batch.lemmas.shape[0]

        preds = self.forward(batch)
        len_vocab = preds.shape[-1]

        tgts = batch.lemmas[:,1:].view(B, -1)
        res = top_k_metric(preds.reshape(-1, len_vocab), tgts.reshape(-1), 1)
        self.log("validation_accuracy", res, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        assert isinstance(self.trainer, Trainer)
        num_batches = self.trainer.datamodule.steps_per_epoch / self.trainer.accumulate_grad_batches #type: ignore
        num_steps = math.ceil(num_batches * self.trainer.max_epochs)
        print(f"Will run for {num_steps} steps")

        pretrained_params, new_params = [], []
        for name, param in self.named_parameters():
            if name.startswith("decoder") or name.startswith("pointer"):
                new_params.append(param)
            else:
                pretrained_params.append(param)

        optimizer = optim.Adam([
            {"params": new_params},
            {"params": pretrained_params, 'lr': self.learning_rate / 100 }
        ], lr=self.learning_rate) #type: ignore

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.hparams["warmup_steps"],
            max_iters=num_steps #type:ignore
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()