from argparse import Namespace
import copy
import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F

from typing import Dict, Any

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DistributedSampler
from griffon.constants import TGT_IGNORE_INDEX
from griffon.coq_dataclasses import CounTBatch

from griffon.dataset.count_dataset import CounTDataset
from griffon.models.scheduled_optimizer import ScheduledOptim

from tqdm import tqdm

import numpy as np
from sklearn.metrics import top_k_accuracy_score

import wandb

def train(model, datasets:Dict[str, CounTDataset], config:Dict[str,Any], args:Namespace):

    should_log = not (args.distributed and (not args.is_main))
    # if should_log:
    #     wandb.init(
    #         project="griffon",
    #         entity="axelmarmet",
    #         config=config
    #     )

    training_config = config["training"]

    # get the dataloaders
    train_split, val_split, test_split = datasets["train"], datasets["valid"], datasets["test"]
    sampler = DistributedSampler(train_split) if args.distributed else None
    dataloaders:Dict[str, DataLoader[CounTBatch]] = {}

    # we divide the batch size by the world_size so that the total
    # batch size does not vary with the number of GPUs used
    assert training_config["batch_size"] % args.world_size == 0, \
        f"batch size ({training_config['batch_size']}) is not cleanly divided by " \
        f"number of gpus ({args.world_size})"
    batch_size = training_config["batch_size"] // args.world_size

    dataloaders['train'] = train_split.to_dataloader(batch_size, sampler)
    dataloaders['val'] =   val_split.to_dataloader(batch_size)
    dataloaders['test'] =  test_split.to_dataloader(batch_size)

    # get the optimizer
    opt = ScheduledOptim(
        optim.Adam(model.parameters()),
        training_config["lr_mult"],
        config["architecture"]["embedding_dim"],
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
            sampler.set_epoch(epoch) # type:ignore
            dist.barrier()

        total_loss = torch.zeros((1), device=args.device)
        model.train()

        for inp, tgt_ids in tqdm(dataloaders['train'], disable=not should_log):
            inp.to(args.device)
            tgt_ids = tgt_ids.to(args.device)

            opt.zero_grad()
            pred = model.forward(inp, predict=True)

            len_vocab = pred.shape[-1]

            log_probs = F.log_softmax(pred).view(-1, len_vocab)
            loss = criterion(log_probs, tgt_ids.view(-1))

            total_loss += loss
            loss.backward()
            opt.step_and_update_lr()

        # get mean loss and not sum of mean batch loss
        total_loss /= epochs
        dist.reduce(total_loss, 0, dist.ReduceOp.SUM)
        total_loss /= args.world_size

        accs = test(dataloaders, model, args.device, verbose=should_log)
        if val_max < accs['val'] and should_log:
            val_max = accs['val']
            best_model = copy.deepcopy(model)

        if should_log:
            print("Epoch {}: Validation: {:.4f}. Test: {:.4f}, Loss: {:.4f}".format(
                epoch + 1, accs['val'], accs['test'], total_loss.item()))
            # wandb.log({
            #     "training loss": total_loss,
            #     "validation accuracy": accs['val'],
            #     "test accuracy": accs['test']
            # })

    final_accs = test(dataloaders, best_model, args.device, should_log)
    if should_log:
        print("FINAL MODEL: Validation: {:.4f}. Test: {:.4f}".format(final_accs['val'], final_accs['test']))

    return best_model

def test(dataloaders, model, device, verbose):
    model.eval()

    accs = {}
    for dataset in dataloaders:
        if dataset == "train":
            continue

        labels = []
        predictions = []
        for inp, tgt_ids in tqdm(dataloaders[dataset], disable=not verbose):


            inp.to(device)
            pred = model.forward(inp, predict=True)
            predictions.append(pred.round().cpu().detach().numpy())
            labels.append(tgt_ids.cpu().numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accs[dataset] = top_k_accuracy_score(labels, predictions, k=3)
    return accs
