import argparse
import os
from numpy import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import Dict, Any, Callable, NamedTuple
from torch import Tensor

import json

from torchtext.vocab import Vocab
import wandb
from griffon.dataset.count_dataset import CounTDataset
from griffon.dataset.semantic_testcase_dataset import SemanticTestCaseDataset
from griffon.models.encoder.code_transformer import CodeTransformer

from griffon.coq_dataclasses import CounTInput
from griffon.models.encoder.standard_transformer import Seq2SeqEncoder
from griffon.models.encoder.train import train
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

class CounTInput(NamedTuple):
    input_ids          : Tensor # shape `batch x max number tokens x number_subtokens`
    distance_indices   : Tensor # shape `batch x number distances x max number tokens x max number_tokens`
    distance_bins      : Tensor # shape `batch x number distances x number bins`
    input_padding_mask : Tensor  # shape `batch x max_number_tokens`

class CounT(nn.Module):

    def __init__(self, vocab:Vocab, config:Dict[str,Any]):
        super(CounT, self).__init__()

        self.subtoken_embedding_dim = config["subtoken_embedding_dim"]
        self.num_subtokens = config["num_subtokens"]
        self.d_model = config["token_embedding_dim"]

        self.vocab = vocab

        self.scale_token_embeddings = config["scale_token_embeddings"]

        self.embedding = nn.Embedding(len(vocab), self.subtoken_embedding_dim)

        self.token_encoder = nn.Linear(self.num_subtokens * self.subtoken_embedding_dim, self.d_model)
        self.token_decoder = nn.Linear(self.d_model, self.num_subtokens * self.subtoken_embedding_dim)
        self.activation_fn = _get_activation_fn(config["activation_fn"])

        transformer_config = config["transformer"]
        assert transformer_config["type"] in ["code", "standard"]
        if transformer_config["type"] == "code":
            self.encoder = CodeTransformer(transformer_config)
        else:
            self.encoder = Seq2SeqEncoder(transformer_config)


    def forward(self, inp:CounTInput, predict:bool=False)->Tensor:

        B, S = inp.input_ids.shape[:2]

        subtokens_embeddings = self.embedding.forward(inp.input_ids).view(B, S, -1)
        token_embeddings = self.token_encoder(subtokens_embeddings)
        token_embeddings = self.activation_fn(token_embeddings)

        if self.scale_token_embeddings:
            token_embeddings *= sqrt(self.d_model)

        relative_distances = inp.distance_indices, inp.distance_bins

        if isinstance(self.encoder, Seq2SeqEncoder):
            token_embeddings = self.encoder.forward(
                token_embeddings,
                src_padding_mask=inp.input_padding_mask
            )
        else:
            assert isinstance(self.encoder, CodeTransformer)
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

        if predict:
            return subtokens_embeddings @ self.embedding.weight.T
        else:
            return subtokens_embeddings

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

    model = CounT(datasets["train"].vocab, config["architecture"])
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
        "--device", type=str, default="cpu", help="The device on which to run the training (default : cpu)"
    )
    arg_parser.add_argument('--distributed', dest='distributed', action='store_true', help="""
        use distributed training, if set then device must not be specified
    """)
    arg_parser.add_argument(
        "--save_dir", required=True
    )
    arg_parser.add_argument('--use_wandb', dest='use_wandb', action='store_true')

    args = arg_parser.parse_args()
    setattr(args, "count_root", os.path.join(args.data_root, "CounT"))
    setattr(args, "semantic_tests_root", os.path.join(args.data_root, "semantic_tests"))

    assert not (args.distributed and args.device != "cpu"), "flag --distributed cannot be set at the same time that a device is given"

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


