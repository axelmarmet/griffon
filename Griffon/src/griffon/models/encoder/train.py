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
from griffon.constants import TGT_IGNORE_INDEX
from griffon.coq_dataclasses import CounTBatch

from griffon.dataset.count_dataset import CounTDataset
from griffon.metrics import top_k_metric
from griffon.models.scheduled_optimizer import ScheduledOptim

from tqdm import tqdm

import numpy as np

import wandb

def train(model, datasets:Dict[str, CounTDataset], config:Dict[str,Any], args:Namespace):

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
    train_split, val_split, test_split = datasets["train"], datasets["valid"], datasets["test"]

    train_sampler = DistributedSampler(train_split) if args.distributed else None
    test_sampler = DistributedSampler(test_split, shuffle=False) if args.distributed else None
    val_sampler = DistributedSampler(val_split, shuffle=False) if args.distributed else None

    pad_idx = train_split.pad_id
    ignore_pad_idx = training_config["ignore_pad_idx"]

    dataloaders:Dict[str, DataLoader[CounTBatch]] = {}

    # we divide the batch size by the world_size so that the total
    # batch size does not vary with the number of GPUs used
    assert training_config["batch_size"] % args.world_size == 0, \
        f"batch size ({training_config['batch_size']}) is not cleanly divided by " \
        f"number of gpus ({args.world_size})"
    batch_size = training_config["batch_size"] // args.world_size

    dataloaders['train'] = train_split.to_dataloader(batch_size, train_sampler)
    dataloaders['val'] =   val_split.to_dataloader(batch_size, val_sampler)
    dataloaders['test'] =  test_split.to_dataloader(batch_size, test_sampler)

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

        for i, (inp, tgt_ids) in enumerate(tqdm(dataloaders['train'], disable=not should_log)):
            inp.to(args.device)
            tgt_ids = tgt_ids.to(args.device)

            if ignore_pad_idx:
                tgt_ids[tgt_ids == pad_idx] = TGT_IGNORE_INDEX


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

        accs = test(dataloaders, model, args, verbose=should_log, ignore_pad_idx=ignore_pad_idx, pad_idx=pad_idx)

        if val_max < accs['val'] and should_log:
            val_max = accs['val']
            best_model = copy.deepcopy(model)

        if should_log:
            print("Epoch {}: Validation: {:.4f}, Loss: {:.4f}".format(
                epoch + 1, accs['val'], total_loss.item()))

            path = os.path.join(args.save_dir, f"model_{epoch+1}.pkl")
            torch.save(model.state_dict(), path)

            if args.use_wandb:
                wandb.log({
                    "training loss": total_loss,
                    "validation top 3 accuracy": accs['val'],
                })

    final_accs = test(dataloaders, best_model, args, should_log, ignore_pad_idx, pad_idx)
    if should_log:
        print("FINAL MODEL: Validation: {:.4f}".format(final_accs['val']))

    return best_model

def test(dataloaders, model, args:Namespace, verbose:bool, ignore_pad_idx:bool=False, pad_idx = -1):

    assert not (ignore_pad_idx and pad_idx == -1), \
        "give a correct value to pad_idx if you want to ignore it"

    model.eval()

    accs = {}
    for dataset in dataloaders:
        if dataset == "train" or dataset == "test":
            continue

        results = []
        for i, (inp, tgt_ids) in enumerate(tqdm(dataloaders[dataset], disable=not verbose)):

            inp.to(args.device)
            tgt_ids = tgt_ids.to(args.device)

            pred:Tensor = model.forward(inp, predict=True).detach()
            vocab_len = pred.shape[-1]

            res = top_k_metric(pred.reshape(-1, vocab_len), tgt_ids.reshape(-1), k=3)

            results.append(res)


        accs[dataset] = results = torch.cat(results).mean()

    if args.distributed:
        for value in accs.values():
            dist.reduce(value, 0, dist.ReduceOp.SUM)

    accs = {key:val.item() / args.world_size for key, val in accs.items()}

    return accs