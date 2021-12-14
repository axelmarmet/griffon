from argparse import Namespace
import copy
import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F

from typing import Dict, Any
from torch import Tensor

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data.dataset import Dataset
from griffon.constants import TGT_IGNORE_INDEX
from griffon.coq_dataclasses import CounTBatch

from griffon.dataset.count_dataset import CounTDataset
from griffon.dataset.semantic_testcase_dataset import SemanticTestCaseDataset, SemanticTestCases
from griffon.metrics import top_k_metric
from griffon.models.scheduled_optimizer import ScheduledOptim

from tqdm import tqdm

import numpy as np

import wandb

def train(model,
          datasets:Dict[str, Dataset],
          config:Dict[str,Any],
          args:Namespace):

    should_log = not (args.distributed and (not args.is_main))
    if should_log and args.use_wandb:
        wandb.init(
            project="griffon",
            entity="axelmarmet",
            config=config
        )
        wandb.watch(model, log='all')

    training_config = config["training"]

    assert training_config["simulated_batch_size"] % training_config["batch_size"] == 0
    steps_before_opt = training_config["simulated_batch_size"] // training_config["batch_size"]

    # get the dataloaders
    train_split:CounTDataset = datasets["train"] #type: ignore
    val_split:CounTDataset = datasets["valid"] #type: ignore
    semantic_tests:SemanticTestCaseDataset = datasets["semantic_test"] # type: ignore

    train_sampler = DistributedSampler(train_split) if args.distributed else None
    val_sampler = DistributedSampler(val_split, shuffle=False) if args.distributed else None

    pad_idx = train_split.pad_id
    ignore_pad_idx = training_config["ignore_pad_idx"]


    # we divide the batch size by the world_size so that the total
    # batch size does not vary with the number of GPUs used
    assert training_config["batch_size"] % args.world_size == 0, \
        f"batch size ({training_config['batch_size']}) is not cleanly divided by " \
        f"number of gpus ({args.world_size})"
    batch_size = training_config["batch_size"] // args.world_size

    train_dataloader = train_split.to_dataloader(batch_size, train_sampler)
    val_dataloader   = val_split.to_dataloader(batch_size, val_sampler)
    sem_test_dataloader = semantic_tests.to_dataloader()

    # get the optimizer
    opt = ScheduledOptim(
        optim.Adam(model.parameters()),
        training_config["lr_mult"],
        config["architecture"]["token_embedding_dim"],
        training_config["warmup_steps"]
    )

    # get the loss
    criterion = nn.NLLLoss(ignore_index=TGT_IGNORE_INDEX).to(args.device)

    epochs = training_config["epochs"]

    best_model = model
    val_max = -np.inf
    for epoch in range(epochs):

        if args.distributed:
            # necessary for shuffling to work with the distributed sampler
            train_sampler.set_epoch(epoch) # type:ignore
            dist.barrier()

        total_loss = torch.zeros((1), device=args.device)
        model.train()

        for i, (inp, tgt_ids) in enumerate(tqdm(train_dataloader, disable=not should_log)):
            inp.to(args.device)
            tgt_ids = tgt_ids.to(args.device)

            assert torch.all(tgt_ids != pad_idx)

            pred = model.forward(inp, predict=True)

            len_vocab = pred.shape[-1]

            log_probs = F.log_softmax(pred, -1).view(-1, len_vocab)
            loss = criterion(log_probs, tgt_ids.view(-1)) / steps_before_opt

            total_loss += loss
            loss.backward()

            if i % steps_before_opt == steps_before_opt-1:
                opt.step_and_update_lr()
                opt.zero_grad()


        # get mean loss and not sum of mean batch loss
        total_loss /= epochs
        if args.distributed:
            dist.reduce(total_loss, 0, dist.ReduceOp.SUM)
        total_loss /= args.world_size

        val_accs = test(val_dataloader, model, args, verbose=should_log, ignore_pad_idx=ignore_pad_idx, pad_idx=pad_idx)
        sem_accs = semantic_test(sem_test_dataloader, model, args, verbose=should_log)

        metrics = dict(val_accs, **sem_accs)

        if val_max < metrics['semantic tests precision'] and should_log:
            val_max = metrics['semantic tests precision']
            best_model = copy.deepcopy(model)

        if should_log:
            print(f"Epoch {epoch+1}:")


            path = os.path.join(args.save_dir, f"model_{epoch+1}.pkl")
            torch.save(model.state_dict(), path)

            if args.use_wandb:
                wandb.log({
                    "training loss": total_loss,
                    "metrics":metrics
                })

    return best_model

@torch.no_grad()
def semantic_test(dataloader:DataLoader[SemanticTestCases], model, args:Namespace, verbose:bool):

    model.eval()

    results = []
    for test_case in tqdm(dataloader, disable=not verbose):

        inp, tgt_ids = test_case.batch

        inp.to(args.device)
        tgt_ids = tgt_ids.to(args.device)

        predictions:Tensor = model.forward(inp, predict=True).detach()

        if verbose:
            for pred, targets, name, orig_s, masked_s in zip(predictions,
                                                            tgt_ids, test_case.names,
                                                            test_case.original_sentences,
                                                            test_case.masked_sentences):
                mask = targets != TGT_IGNORE_INDEX
                pred = torch.argmax(pred[mask], dim=-1)
                targets = targets[mask]
                assert pred.shape == targets.shape

                correctly_classified = torch.count_nonzero(pred == targets)

                print(f"{name} [{correctly_classified}/{pred.shape[0]}]")
                print(f"original : {orig_s}")
                print(f"masked   : {masked_s}")
                print("------------------")
                for i in range(pred.shape[0]):
                    print(f"expected : {model.vocab[targets[i]]}")
                    print(f"got : {model.vocab[pred[i]]}")
                print("------------------")

        mask = tgt_ids != TGT_IGNORE_INDEX
        res = top_k_metric(predictions.reshape(-1, len(model.vocab)), tgt_ids.reshape(-1), k=1)

        results.append(res)

    return {"semantic tests precision":  torch.cat(results).mean()}

@torch.no_grad()
def test(dataloader:DataLoader[CounTBatch], model, args:Namespace, verbose:bool, ignore_pad_idx:bool=False, pad_idx = -1):

    k = 3
    assert not (ignore_pad_idx and pad_idx == -1), \
        "give a correct value to pad_idx if you want to ignore it"

    model.eval()

    results = []
    for inp, tgt_ids in tqdm(dataloader, disable=not verbose):

        inp.to(args.device)
        tgt_ids = tgt_ids.to(args.device)

        pred:Tensor = model.forward(inp, predict=True).detach()
        vocab_len = pred.shape[-1]

        res = top_k_metric(pred.reshape(-1, vocab_len), tgt_ids.reshape(-1), k=k)

        results.append(res)

    mean = torch.cat(results).mean()

    if args.distributed:
            dist.reduce(mean, 0, dist.ReduceOp.SUM)
            mean = mean / args.world_size

    return {f"top {k} accuracy":mean}