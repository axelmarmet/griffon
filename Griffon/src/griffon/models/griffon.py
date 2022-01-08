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

from typing import Dict, Any, Callable, NamedTuple
from torch import Tensor

import json
import math

from torchtext.vocab import Vocab
import wandb
from griffon.constants import NUM_SUB_TOKENS, TGT_IGNORE_INDEX, UNK_TOKEN
from griffon.dataset.count_dataset import CounTDataset
from griffon.dataset.semantic_testcase_dataset import SemanticTestCaseDataset, SemanticTestCases
from griffon.functional.focal_loss import focal_loss
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
        return mask

    def forward(self, inp:GriffonBatch)->Tensor:
        NUM_DISTANCES = inp.distances_indices.shape[0]
        B, NUM_STATEMENTS, S = inp.statements.shape[:3]
        T = inp.lemmas.shape[1]

        # we process the statements first
        statement_subtokens_embeddings:Tensor = \
            self.embedding.forward(inp.sequences).view(B, NUM_STATEMENTS, S, self.subtoken_embedding_dim * NUM_SUB_TOKENS) # type: ignore

        statement_token_embeddings = self.token_encoder(statement_subtokens_embeddings)

        # we flatten B * NUM_STATEMENT into a bigger batch dimension
        statement_token_embeddings = statement_token_embeddings.reshape(
            B * NUM_STATEMENTS, S, self.token_embedding_dim)
        statement_token_padding = inp.token_padding_mask.reshape(
            B * NUM_STATEMENTS, S
        )
        relative_distances = inp.distances_indices.reshape(B * NUM_STATEMENTS, NUM_DISTANCES, S, S), \
                             inp.distances_bins.reshape(B * NUM_STATEMENTS, NUM_DISTANCES, -1)

        # we feed those flattened tensor to the encoder
        statement_token_embeddings = self.ct_encoder.forward(
            statement_token_embeddings,
            src_key_padding_mask=statement_token_padding,
            relative_distances=relative_distances)

        # we flatten the statement token embeddings for the decoder, predictor and pointer
        flat_stmt_token_embeddings = statement_token_embeddings[~inp.token_padding_mask.T]

        statement_subtokens_embeddings = \
            self.token_decoder(flat_stmt_token_embeddings).reshape(B * NUM_STATEMENTS,  S, NUM_SUB_TOKENS, self.subtoken_embedding_dim)
        statement_subtokens_embeddings = statement_subtokens_embeddings.view(B, NUM_STATEMENTS * S * NUM_SUB_TOKENS, self.subtoken_embedding_dim)

        # we discard the <eos> token from the input
        lemma_subtokens = inp.lemmas[:,:-1]
        # we transpose the batch dimension with the sequence dimension in the lemmas
        lemma_subtokens = lemma_subtokens.permute(1,0,2)

        # giving the embedding of copied subtoken would probably be better
        lemma_subtokens[lemma_subtokens >= self.vocab_len] = self.unk_id

        lemma_subtokens_embeddings:Tensor = self.embedding.forward(lemma_subtokens).view(T-1, B, self.subtoken_embedding_dim * NUM_SUB_TOKENS) # type: ignore
        lemma_tokens_embeddings = self.token_encoder(lemma_subtokens_embeddings)

        output_tokens_embeddings = self.decoder.forward(memory=statement_token_embeddings,
                                                        tgt=lemma_tokens_embeddings,
                                                        tgt_mask=self.get_tgt_mask(T-1),
                                                        memory_key_padding_mask=statement_token_padding)

        output_subtokens_embeddings:Tensor = self.token_decoder(output_tokens_embeddings).reshape(T-1, B, NUM_SUB_TOKENS, self.subtoken_embedding_dim)
        # we flatten the output subtokens
        # output_subtokens_embeddings = output_subtokens_embeddings.reshape(-1, self.subtoken_embedding_dim)

        # we get the predictions without using the predictor network
        raw_preds:Tensor = self.predictor(output_subtokens_embeddings)

        # right now

        # statement_token_padding : B * NUM_STATEMENTS, S

        # logits                  : T-1, B, NUM_SUB_TOKENS, VOCAB_LEN
        # extended_vocabulary_ids : B, NUM_STATEMENTS, S, NUM_SUB_TOKENS
        # src_subtokens           : B, NUM_STATEMENTS * S * NUM_SUB_TOKENS, self.subtoken_embedding_dim
        # tgt_subtokens           : T-1, B, NUM_SUB_TOKENS, self.subtoken_embedding_dim
        # len_vocab : easy
        # len_extended_vocab : easy

        # we combine the predictions with the one from the pointer network
        log_probs = self.pointer.forward(
            logits = raw_preds,
            extended_vocab_ids = inp.extended_vocabulary_ids[~inp.token_padding_mask].view(-1),
            src_subtokens = statement_subtokens_embeddings,
            tgt_subtokens = output_subtokens_embeddings,
            len_vocab = self.vocab_len,
            len_extended_vocab = len(inp.extended_vocabulary))

        return log_probs

    def training_step(self, batch:GriffonBatch, batch_idx):
        B = batch.lemmas.shape[0]

        preds = self.forward(batch)
        # we shift the targets by 1 since <sos> should predict the first actual token
        # and the last actual token should predict <eos>
        tgts = batch.lemmas[:,1:].view(B, -1)
        loss = F.cross_entropy(preds, tgts, ignore_index=TGT_IGNORE_INDEX)
        self.log("training_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch:GriffonSample, batch_idx, dataloader_idx=0):
        preds = self.forward(batch)

        tgts = batch.lemma[1:].view(-1)
        res = top_k_metric(preds, tgts, 1)
        self.log("validation accuracy", res, on_step=False, on_epoch=True)

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