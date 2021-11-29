import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import Dict, Any, Callable, NamedTuple
from torch import Tensor

import json

from torchtext.vocab import Vocab
from griffon.dataset.count_dataset import CounTDataset
from griffon.models.encoder.code_transformer import CodeTransformer

from griffon.coq_dataclasses import CounTInput
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

        self.embedding_dim = config["embedding_dim"]
        self.num_subtokens = config["num_subtokens"]

        self.vocab = vocab

        self.embedding = nn.Embedding(len(vocab), self.embedding_dim)

        self.token_encoder = nn.Linear(self.num_subtokens * self.embedding_dim, self.embedding_dim)
        self.token_decoder = nn.Linear(self.embedding_dim, self.num_subtokens * self.embedding_dim)
        self.activation_fn = _get_activation_fn(config["activation_fn"])

        self.ct_encoder = CodeTransformer(config["code_transformer"])

    def forward(self, inp:CounTInput, predict:bool=False)->Tensor:

        B, S = inp.input_ids.shape[:2]

        subtokens_embeddings = self.embedding.forward(inp.input_ids).view(B, S, -1)
        token_embeddings = self.token_encoder(subtokens_embeddings)
        token_embeddings = self.activation_fn(token_embeddings)

        relative_distances = (inp.distance_indices.transpose(0,1), \
                              inp.distance_bins.permute(1,2,0))

        token_embeddings = self.ct_encoder.forward(
            token_embeddings,
            src_key_padding_mask=inp.input_padding_mask,
            relative_distances=relative_distances)

        subtokens_embeddings = self.token_decoder(token_embeddings).reshape(B, S, self.num_subtokens, self.embedding_dim)
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
    datasets["train"] = CounTDataset(args.data_root, "train")
    datasets["test"] = CounTDataset(args.data_root, "test")
    datasets["valid"] = CounTDataset(args.data_root, "valid")

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
        torch.save(best_model, "my_best_model")

    if args.distributed:
        cleanup()


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="""
            Train CounT
        """
    )
    arg_parser.add_argument(
        "--data_root", type=str, default="data/CounT", help="The root directory of the dataset"
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
    arg_parser.set_defaults(feature=True)
    args = arg_parser.parse_args()

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


