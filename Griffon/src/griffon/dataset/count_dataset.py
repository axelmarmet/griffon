import os
from glob import glob
import pickle
from typing import Tuple

import numpy as np

import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from random import randint, random

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

from griffon.coq_dataclasses import *
from griffon.utils import pad_mask, pad_list

from griffon.constants import NUM_SUB_TOKENS, MASK_TOKEN, PAD_TOKEN, TGT_IGNORE_INDEX

class CounTDataset(Dataset[CounTBatch]):
    """
    Unites common functionalities used across different datasets such as applying the token mapping to the
    distance matrices and collating the matrices from multiple samples into one big tensor.
    """

    def __init__(self, root_path:str, split:str):

        assert os.path.exists(root_path), f"Path {root_path} does not exist"
        assert split in ["train", "test", "valid"], f"Split {split} is not supported"

        sample_dir = os.path.join(root_path, split)
        assert os.path.exists(sample_dir), f"Sample directory {sample_dir} does not exist"

        self.files = sorted(glob(os.path.join(sample_dir, "*.pickle")))
        self.vocab = pickle.load(open(os.path.join(root_path, "vocab.pkl"), "rb"))

        self.pad_id = self.vocab[PAD_TOKEN]

    def to_dataloader(self, batch_size:int, num_workers:int):
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=self.collate_fn, # type: ignore
            pin_memory=True,
            num_workers=num_workers)

    def __getitem__(self, index:int)->CounTSample:
        sample:CounTSample = pickle.load(open(self.files[index], "rb"))
        return sample

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def collate_fn(samples: List[CounTSample])->CounTBatch:

        # we need to go through every tensor and pad it before stacking them along
        # the batching digm

        max_input_length = max([sample.input_ids.shape[0] for sample in samples])

        # only tensor where no padding is needed
        distance_bins_list = [sample.distance_bins for sample in samples]

        input_ids_list = []
        distance_indices_list = []
        target_ids_list = []

        input_lengths = []

        for sample in samples:

            input_length = sample.input_ids.shape[0]
            input_lengths.append(input_length)
            pad_length = max_input_length - input_length

            input_ids_list.append(
                F.pad(sample.input_ids, [0, 0, 0, pad_length], value=0)
            )
            distance_indices_list.append(
                F.pad(sample.distance_indices, [0, pad_length, 0, pad_length], value=0)
            )
            target_ids_list.append(
                F.pad(sample.target_ids, [0, 0, 0, pad_length], value=TGT_IGNORE_INDEX)
            )


        input_ids = torch.stack(input_ids_list)
        distance_indices = torch.stack(distance_indices_list)
        distance_bins = torch.stack(distance_bins_list)
        target_ids = torch.stack(target_ids_list)

        # we need to generate the padding mask this way, because we need to
        # to distinguish between a token padding and a subtoken padding
        input_padding_mask = torch.logical_not(pad_mask(torch.tensor(input_lengths), max_input_length).bool())

        input_batch = CounTBatchInput(
            input_ids = input_ids,
            distance_indices = distance_indices,
            distance_bins = distance_bins,
            input_padding_mask = input_padding_mask,
        )
        return CounTBatch(input_batch, target_ids)