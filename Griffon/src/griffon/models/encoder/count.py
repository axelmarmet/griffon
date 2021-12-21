import argparse
import os
import pickle
from numpy import sqrt

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import Dict, Any, Callable, NamedTuple
from torch import Tensor

import json

from torchtext.vocab import Vocab
import wandb
from griffon.constants import TGT_IGNORE_INDEX
from griffon.dataset.count_dataset import CounTDataset
from griffon.dataset.semantic_testcase_dataset import SemanticTestCaseDataset, SemanticTestCases
from griffon.metrics import top_k_metric
from griffon.models.cosine_warmup_scheduler import CosineWarmupScheduler
from griffon.models.encoder.code_transformer import CodeTransformer

from griffon.coq_dataclasses import CounTBatch, CounTInput
from griffon.models.encoder.standard_transformer import Seq2SeqEncoder
from griffon.models.encoder.train import train
from griffon.preprocessing.stage2.vocab import AbstractVocab
from griffon.utils import cleanup, set_seed, setup

def _get_activation_fn(activation:str)->Callable[[Tensor],Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "sigmoid":
        return F.sigmoid
    else:
        raise ValueError("unknown activation function")


class CounT(pl.LightningModule):

    def __init__(self, config:Dict[str,Any], learning_rate):
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

        self.scale_token_embeddings = architecture_config["scale_token_embeddings"]

        self.embedding = nn.Embedding(len(self.vocab), self.subtoken_embedding_dim)

        self.token_encoder = nn.Linear(self.num_subtokens * self.subtoken_embedding_dim, self.d_model)
        self.token_decoder = nn.Linear(self.d_model, self.num_subtokens * self.subtoken_embedding_dim)
        self.activation_fn = _get_activation_fn(architecture_config["activation_fn"])

        self.encoder = CodeTransformer(architecture_config["code_transformer"])

    def forward(self, inp:CounTInput)->Tensor:

        B, S = inp.input_ids.shape[:2]

        subtokens_embeddings = self.embedding.forward(inp.input_ids).view(B, S, -1)
        token_embeddings = self.token_encoder(subtokens_embeddings)
        token_embeddings = self.activation_fn(token_embeddings)

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

        subtokens_embeddings = self.token_decoder(token_embeddings).reshape(B, S, self.num_subtokens, self.subtoken_embedding_dim)
        subtokens_embeddings = self.activation_fn(subtokens_embeddings)

        return subtokens_embeddings

    def training_step(self, batch:CounTBatch, batch_idx):
        inp:CounTInput
        tgt_ids:Tensor
        inp, tgt_ids = batch.as_tuple()
        preds = self.forward(inp) @ self.embedding.weight.T

        len_vocab = preds.shape[-1]
        log_probs = F.log_softmax(preds, -1).view(-1, len_vocab)
        loss = F.nll_loss(log_probs, tgt_ids.view(-1), ignore_index=TGT_IGNORE_INDEX)

        self.log("training loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        def semantic_validation_step(test_cases:SemanticTestCases):
            inp, tgt_ids = test_cases.batch.as_tuple()
            predictions:Tensor = self.forward(inp) @ self.embedding.weight.T

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

            self.log("semantic test cases", res, on_step=False, on_epoch=True, add_dataloader_idx=False)
            assert isinstance(self.logger, WandbLogger)
            self.logger.log_text(key="verbose semantic tests",
                                 columns = verbose_columns,
                                 data = verbose_data)
            self.logger.log_text(key="summary semantic tests",
                                 columns = summary_columns,
                                 data = summary_data)

        def mlm_validation_step(batch:CounTBatch):
            K = 3

            inp, tgt_ids = batch.as_tuple()

            predictions = self.forward(inp) @ self.embedding.weight.T
            res = top_k_metric(predictions.reshape(-1, len(self.vocab)), tgt_ids.reshape(-1), k=K).mean()
            self.log(f"validation top {K}", res, on_step=False, on_epoch=True, add_dataloader_idx=False)

        assert dataloader_idx < 2, f"Unexpected index {dataloader_idx}"

        if dataloader_idx == 0:
            assert isinstance(batch, CounTBatch)
            mlm_validation_step(batch)
        else:
            assert isinstance(batch, SemanticTestCases)
            semantic_validation_step(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate) #type: ignore

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.hparams["optimizer"]["warmup_steps"],
            max_iters=self.hparams["optimizer"]["max_iters"] #type:ignore
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()

def run(args:argparse.Namespace, rank:int, world_size:int):

    if args.distributed:
        setup(rank, world_size)

    assert os.path.exists(args.config), f"file {args.config} does not exist"

    args.world_size = world_size
    args.rank = rank
    args.is_main = rank == 0
    if args.distributed:
        torch.cuda.device(args.rank)
        args.device = rank

    config = json.load(open(args.config, "r"))
    set_seed(config["seed"])

    # get the dataset
    datasets = {}
    datasets["train"] = CounTDataset(args.count_root, "train")
    datasets["valid"] = CounTDataset(args.count_root, "valid")
    datasets["semantic_test"] = SemanticTestCaseDataset(args.semantic_tests_root)

    model = CounT(config["architecture"])
    model = model.to(args.device)

    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.rank],
            output_device=args.rank
        )

    best_model = train(model, datasets, config, args)

    if args.is_main:
        filename = os.path.join(args.save_dir, "best_model")
        torch.save(best_model.state_dict(), filename)

    if args.distributed:
        cleanup()


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="""
            Train CounT
        """
    )
    arg_parser.add_argument(
        "--data_root", type=str, default="data", help="The root directory of the dataset"
    )
    arg_parser.add_argument(
        "--config", type=str, required=True, help="The config file that contains all hyperparameters"
    )
    arg_parser.add_argument(
        "--gpus", type=int, default=0, help="The number of gpus"
    )
    arg_parser.add_argument(
        "--save_dir", required=True
    )
    arg_parser.add_argument('--use_wandb', dest='use_wandb', action='store_true')

    args = arg_parser.parse_args()
    setattr(args, "count_root", os.path.join(args.data_root, "CounT"))
    setattr(args, "semantic_tests_root", os.path.join(args.data_root, "semantic_tests"))


    if args.distributed:
        # check how many GPUs are available
        size = torch.cuda.device_count()

        # spawn that many processes
        processes = []
        mp.set_start_method("spawn")
        for rank in range(size):
            p = mp.Process(target=run, args=(args, rank, size))
            p.start()
            processes.append(p)

        # wait for all processes to be done to finish
        for p in processes:
            p.join()
    else:
        run(args, 0, 1)


