# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
import torch.nn.functional as F

from random import random, randint

from typing import Dict, Tuple, List
from torch import Tensor

import pickle
import os
import glob

from torchtext.vocab import Vocab

from griffon.utils import pad_list

import numpy as np

from griffon.coq_dataclasses import *
from griffon.constants import *
from griffon.dataset.count_dataset import CounTDataset

import random

random.seed(0)


# %%
ds = CounTDataset("../data/CounT/train", "../models/vocab.pickle")


# %%
print(ds.vocab.get_itos()[:8])

test = pickle.load(open("../data/small/stage2/train/proof00000002.pickle", "rb"))
print(test.hypotheses[1].tokens)

test_2 = next(iter(ds))
print(test_2.input_ids)


# %%
random.seed(0)
pad_idx = ds.vocab[PAD_TOKEN]
print(pad_idx)

def print_sample(vocab:Vocab, sample:CounTSample):
    itos = ds.vocab.get_itos()

    def pretty_print(ids:Tensor):
        tokens = ids.tolist()
        print(" ".join(["_".join([itos[subtoken] for subtoken in token if subtoken != pad_idx]) for token in tokens]))

    print("sample")
#    print("super original token")
#    print(sample.input_ids.tolist())
    original_tokens = torch.clone(sample.input_ids)
    if torch.count_nonzero(sample.target_mask):
        original_tokens[sample.target_mask] = sample.target_ids

    print("original")
    pretty_print(original_tokens)
    print("input")
    pretty_print(sample.input_ids)
    print("-----------------")

for i in range(10):
    print(i)
    print_sample(ds.vocab, ds[i])
    if i == 10:
        break


# %%
from torch.utils.data.dataloader import DataLoader

def print_batch(vocab:Vocab, inp:CounTInput, tgt:CounTTarget):
    itos = ds.vocab.get_itos()

    def pretty_print(ids:Tensor):
        tokens = ids.tolist()
        print(" ".join(["_".join([itos[subtoken] for subtoken in token if subtoken != pad_idx]) for token in tokens]))

    for b in range(inp.input_ids.shape[0]):
        print("sample")

        inp_select_mask = torch.logical_not(inp.input_padding_mask[b])
        tgt_select_mask = torch.logical_not(tgt.target_padding_mask[b])

        original_tokens = torch.clone(inp.input_ids[b][inp_select_mask])
        if torch.count_nonzero(tgt.target_mask[b][inp_select_mask]):
            tokens_to_replace = original_tokens[tgt.target_mask[b][inp_select_mask]]
            new_tokens = tgt.target_ids[b][tgt_select_mask]

            original_tokens[tgt.target_mask[b][inp_select_mask]] = tgt.target_ids[b][tgt_select_mask]

        print("original")
        pretty_print(original_tokens)
        print("input")
        pretty_print(inp.input_ids[b][inp_select_mask])
        print("-----------------")

dl = DataLoader(ds, 10, collate_fn=ds.collate_fn)
for inp, tgt in dl:
    print(type(inp))
    print_batch(ds.vocab, inp, tgt)
    break


