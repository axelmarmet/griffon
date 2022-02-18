import pickle
from numpy import sqrt

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import Dict, Any
from torch import Tensor

import math

from griffon.constants import PAD_TOKEN, TGT_IGNORE_INDEX
from griffon.dataset.semantic_testcase_dataset import SemanticTestCases
from griffon.functional.focal_loss import focal_loss
from griffon.metrics import relative_entropy, top_k_metric
from griffon.models.cosine_warmup_scheduler import CosineWarmupScheduler
from griffon.models.subtokens.predictor import Predictor
from griffon.models.subtokens.token_decoder import TokenDecoder
from griffon.models.encoder.code_transformer import CodeTransformer

from griffon.coq_dataclasses import CounTBatch, CounTBatchInput
from griffon.models.subtokens.token_encoder import TokenEncoder
from griffon.preprocessing.stage2.vocab import AbstractVocab

def _get_activation_fn(activation:str)->nn.Module:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError("unknown activation function")

@MODEL_REGISTRY
class CounT(pl.LightningModule):

    def __init__(self, config:Dict[str,Any], learning_rate:float=1.e-4):
        super(CounT, self).__init__()
        self.save_hyperparameters(config)
        self.learning_rate = learning_rate

        architecture_config = config["architecture"]

        # propagate the d_model config param
        architecture_config["code_transformer"]["encoder_layer"]["d_model"] = architecture_config["token_embedding_dim"]
        architecture_config["code_transformer"]["norm"]["d_model"] = architecture_config["token_embedding_dim"]

        self.subtoken_embedding_dim = architecture_config["subtoken_embedding_dim"]
        self.num_subtokens = architecture_config["num_subtokens"]
        self.d_model = architecture_config["token_embedding_dim"]

        self.vocab:AbstractVocab = pickle.load(open(architecture_config["vocab_file"], "rb"))
        self.pad_idx = self.vocab[PAD_TOKEN]

        self.scale_token_embeddings = architecture_config["scale_token_embeddings"]

        if architecture_config["pretrained_embeddings_path"]:
            embedding_tensor:Tensor = pickle.load(
                open(architecture_config["pretrained_embeddings_path"], "rb"))

            assert embedding_tensor.shape[0] == len(self.vocab)
            assert embedding_tensor.shape[1] == self.subtoken_embedding_dim
            self.embedding = nn.Embedding.from_pretrained(embedding_tensor) # type: nn.Embedding
        else:
            self.embedding = nn.Embedding(len(self.vocab), self.subtoken_embedding_dim)

        self.token_encoder = TokenEncoder(
            self.subtoken_embedding_dim,
            self.d_model,
            _get_activation_fn(architecture_config["activation_fn"])
        )

        self.token_decoder = TokenDecoder(
            self.subtoken_embedding_dim,
            self.d_model,
            _get_activation_fn(architecture_config["activation_fn"])
        )

        self.encoder = CodeTransformer(architecture_config["code_transformer"])

        self.predictor = Predictor(self.subtoken_embedding_dim, len(self.vocab))


    def forward(self, inp:CounTBatchInput)->Tensor:

        B, S = inp.input_ids.shape[:2]

        subtokens_embeddings = self.embedding.forward(inp.input_ids)
        token_embeddings = self.token_encoder(subtokens_embeddings)

        if self.scale_token_embeddings:
            token_embeddings *= sqrt(self.d_model)

        relative_distances = inp.distance_indices, inp.distance_bins

        token_embeddings = self.encoder.forward(
            token_embeddings,
            src_key_padding_mask=inp.input_padding_mask,
            relative_distances=relative_distances)
        token_embeddings = token_embeddings.transpose(1,0)

        assert token_embeddings.shape[0] == B
        assert token_embeddings.shape[1] == S
        assert token_embeddings.shape[2] == self.d_model

        subtokens_embeddings = self.token_decoder(token_embeddings)
        predictions = self.predictor(subtokens_embeddings)

        return predictions

    def training_step(self, batch:CounTBatch, batch_idx):
        inp:CounTBatchInput
        tgt_ids:Tensor
        inp, tgt_ids = batch.as_tuple()
        preds = self.forward(inp)

        len_vocab = preds.shape[-1]
        if self.hparams["optimizer"]["loss_fn"]["type"] == "f_loss":
            gamma = (self.current_epoch / self.trainer.max_epochs) * \
                    self.hparams["optimizer"]["loss_fn"]["gamma"]
            loss = focal_loss(
                preds.view(-1, len_vocab),
                tgt_ids.view(-1),
                gamma=gamma,
                ignore_index=TGT_IGNORE_INDEX
            )
        else:
            loss = F.cross_entropy(
                preds.view(-1, len_vocab),
                tgt_ids.view(-1),
                ignore_index=TGT_IGNORE_INDEX
            )
        assert not torch.any(torch.isnan(loss))
        self.log("training loss", loss, on_step=False, on_epoch=True)
        return loss

    def get_mlm_metrics(self, predictions:Tensor, tgt_ids:Tensor, name_prefix:str):

        metrics = {}
        metrics[f"{name_prefix}_top_3_no_padding"] = \
            top_k_metric(predictions.reshape(-1, len(self.vocab)), tgt_ids.reshape(-1), pad_idx=self.vocab[PAD_TOKEN], k=3).mean()
        metrics[f"{name_prefix}_accuracy_no_padding"] = \
            top_k_metric(predictions.reshape(-1, len(self.vocab)), tgt_ids.reshape(-1), pad_idx=self.vocab[PAD_TOKEN], k=1).mean()
        metrics[f"{name_prefix}_top_3_with_padding"] = \
            top_k_metric(predictions.reshape(-1, len(self.vocab)), tgt_ids.reshape(-1), k=3).mean()
        metrics[f"{name_prefix}_accuracy_with_padding"] = \
            top_k_metric(predictions.reshape(-1, len(self.vocab)), tgt_ids.reshape(-1), k=1).mean()

        mask = tgt_ids != TGT_IGNORE_INDEX

        preds = predictions[mask].reshape(-1, len(self.vocab))

        metrics[f"{name_prefix}_padding_prob"] = \
            torch.mean(F.softmax(preds, dim=-1)[:,self.pad_idx])

        metrics[f"{name_prefix}_relative_entropy_with_padding"] = \
            relative_entropy(preds).mean()
        mask = torch.logical_and(mask, tgt_ids != self.vocab[PAD_TOKEN])
        metrics[f"{name_prefix}_relative_entropy_no_padding"] = \
            relative_entropy(preds).mean()

        return metrics

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        def semantic_validation_step(test_cases:SemanticTestCases):
            inp, tgt_ids = test_cases.batch.as_tuple()
            predictions:Tensor = self.forward(inp)

            itos = self.vocab.get_itos()

            verbose_columns = ["Statement Name", "Masked Statement", "Expected", "Predicted"]
            verbose_data = []

            summary_columns = ["Statement Name", "Ratio"]
            summary_data = []

            for pred, targets, name, orig_s, masked_s in zip(predictions,
                                                            tgt_ids, test_cases.names,
                                                            test_cases.original_sentences,
                                                            test_cases.masked_sentences):
                mask = targets != TGT_IGNORE_INDEX
                pred = torch.argmax(pred[mask], dim=-1)
                targets = targets[mask]
                assert pred.shape == targets.shape

                correctly_classified = torch.count_nonzero(pred == targets)

                summary_data.append([name, f"{correctly_classified}/{pred.shape[0]}"])
                for i in range(pred.shape[0]):
                    verbose_data.append([name, masked_s, itos[targets[i]], itos[pred[i]]])


            res = top_k_metric(predictions.reshape(-1, len(self.vocab)), tgt_ids.reshape(-1), k=1).mean()

            self.log("semantic_test_cases", res, on_step=False, on_epoch=True, add_dataloader_idx=False)
            if isinstance(self.logger, WandbLogger):
                self.logger.log_text(key="verbose semantic tests",
                                        columns = verbose_columns,
                                        data = verbose_data)
                self.logger.log_text(key="summary semantic tests",
                                        columns = summary_columns,
                                        data = summary_data)

        def mlm_validation_step(batch:CounTBatch):
            inp, tgt_ids = batch.as_tuple()

            predictions = self.forward(inp)
            metrics = self.get_mlm_metrics(predictions, tgt_ids, "validation")
            for name, value in metrics.items():
                self.log(name, value, on_step=False, on_epoch=True, add_dataloader_idx=False)

        assert dataloader_idx < 2, f"Unexpected index {dataloader_idx}"

        if dataloader_idx == 0:
            assert isinstance(batch, CounTBatch)
            mlm_validation_step(batch)
        else:
            assert isinstance(batch, SemanticTestCases)
            semantic_validation_step(batch)

    def test_step(self, batch, batch_idx, dataloader_idx=0):

        def semantic_test_step(test_cases:SemanticTestCases):
            inp, tgt_ids = test_cases.batch.as_tuple()
            predictions:Tensor = self.forward(inp)

            itos = self.vocab.get_itos()

            verbose_columns = ["Statement Name", "Masked Statement", "Expected", "Predicted"]
            verbose_data = []

            summary_columns = ["Statement Name", "Ratio"]
            summary_data = []

            for pred, targets, name, orig_s, masked_s in zip(predictions,
                                                            tgt_ids, test_cases.names,
                                                            test_cases.original_sentences,
                                                            test_cases.masked_sentences):
                mask = targets != TGT_IGNORE_INDEX
                pred = torch.argmax(pred[mask], dim=-1)
                targets = targets[mask]
                assert pred.shape == targets.shape

                correctly_classified = torch.count_nonzero(pred == targets)

                summary_data.append([name, f"{correctly_classified}/{pred.shape[0]}"])
                for i in range(pred.shape[0]):
                    verbose_data.append([name, masked_s, itos[targets[i]], itos[pred[i]]])


            res = top_k_metric(predictions.reshape(-1, len(self.vocab)), tgt_ids.reshape(-1), k=1).mean()
            self.log("accuracy semantic test cases", res, on_step=False, on_epoch=True, add_dataloader_idx=False)

            res = top_k_metric(predictions.reshape(-1, len(self.vocab)), tgt_ids.reshape(-1), k=3).mean()
            self.log("top 3 semantic test cases", res, on_step=False, on_epoch=True, add_dataloader_idx=False)

            if isinstance(self.logger, WandbLogger):
                self.logger.log_text(key="test verbose semantic tests",
                                        columns = verbose_columns,
                                        data = verbose_data)
                self.logger.log_text(key="test summary semantic tests",
                                        columns = summary_columns,
                                        data = summary_data)

        def mlm_test_step(batch:CounTBatch):
            inp, tgt_ids = batch.as_tuple()

            predictions = self.forward(inp)
            metrics = self.get_mlm_metrics(predictions, tgt_ids, "test")
            for name, value in metrics.items():
                self.log(name, value, on_step=False, on_epoch=True, add_dataloader_idx=False)

        assert dataloader_idx < 2, f"Unexpected index {dataloader_idx}"

        if dataloader_idx == 0:
            assert isinstance(batch, CounTBatch)
            mlm_test_step(batch)
        else:
            assert isinstance(batch, SemanticTestCases)
            semantic_test_step(batch)

    def configure_optimizers(self):
        assert isinstance(self.trainer, Trainer)
        num_batches = self.trainer.datamodule.steps_per_epoch / self.trainer.accumulate_grad_batches #type: ignore
        num_steps = math.ceil(num_batches * self.trainer.max_epochs)
        print(f"Will run for {num_steps} steps")

        all_params_but_embedding = (param for name, param in self.named_parameters() if not name.startswith("embedding"))

        optimizer = optim.Adam([
            {"params": all_params_but_embedding},
            {"params": self.embedding.parameters(), 'lr': (self.learning_rate or 1e-3) /10 }
        ], lr=self.learning_rate or 1e-3) #type: ignore

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.hparams["optimizer"]["warmup_steps"],
            max_iters=num_steps #type:ignore
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()
        self.log("learning rate factor",
            self.lr_scheduler.get_lr_factor(epoch=self.lr_scheduler.last_epoch),  #type: ignore
            on_step=True)