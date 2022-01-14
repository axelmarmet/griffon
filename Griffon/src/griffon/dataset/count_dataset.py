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

    def __init__(self, root_path:str, split:str,
                 mask_prob:float = 0.15,
                 random_token_prob: float = 0.1,
                 leave_unmasked_prob: float = 0.1):

        assert os.path.exists(root_path), f"Path {root_path} does not exist"
        assert split in ["train", "test", "valid"], f"Split {split} is not supported"

        sample_dir = os.path.join(root_path, split)
        assert os.path.exists(sample_dir), f"Sample directory {sample_dir} does not exist"

        self.files = sorted(glob(os.path.join(sample_dir, "*.pickle")))
        self.vocab = pickle.load(open(os.path.join(root_path, "vocab.pkl"), "rb"))

        self.mask_prob = mask_prob
        self.random_token_prob = random_token_prob
        self.leave_unmasked_prob = leave_unmasked_prob

        self.pad_id = self.vocab[PAD_TOKEN]
        self.mask_id = self.vocab[MASK_TOKEN]

    def to_dataloader(self, batch_size:int, num_workers:int):
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=self.collate_fn, # type: ignore
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True)

    def __getitem__(self, index:int)->CounTSample:
        sample:CounTSample = pickle.load(open(self.files[index], "rb"))
        return sample

    def mask_item(self, statement:Stage2Statement)->CounTSample:
        """Heavily inspired from fairseq implementation

        Args:
            statement (Stage2Statement): [description]

        Returns:
            CounTSample: [description]
        """

        sz = len(statement.tokens)

        assert (
            all(self.mask_id not in token.subtokens for token in statement.tokens)
        ), f"Dataset contains mask_idx ({self.mask_id}), this is not expected!"

        # decide elements to mask
        mask = np.full(sz, False)
        num_mask = int(
            # add a random number for probabilistic rounding
            self.mask_prob * sz + np.random.rand()
        )

        mask_idc = np.random.choice(sz, num_mask, replace=False)

        try:
            mask[mask_idc] = True
        except:  # something wrong
            print(
                "Assigning mask indexes {} to mask {} failed!".format(
                    mask_idc, mask
                )
            )
            raise

        new_item = np.full(len(mask), self.pad_id)
        new_item[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
        return torch.from_numpy(new_item)


        # decide unmasking and random replacement
        rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
        if rand_or_unmask_prob > 0.0:
            rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
            if self.random_token_prob == 0.0:
                unmask = rand_or_unmask
                rand_mask = None
            elif self.leave_unmasked_prob == 0.0:
                unmask = None
                rand_mask = rand_or_unmask
            else:
                unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                decision = np.random.rand(sz) < unmask_prob
                unmask = rand_or_unmask & decision
                rand_mask = rand_or_unmask & (~decision)
        else:
            unmask = rand_mask = None

        if unmask is not None:
            mask = mask ^ unmask

        new_item = np.copy(item)
        new_item[mask] = self.mask_id
        if rand_mask is not None:
            num_rand = rand_mask.sum()
            if num_rand > 0:
                if self.mask_whole_words is not None:
                    rand_mask = np.repeat(rand_mask, word_lens)
                    num_rand = rand_mask.sum()

                new_item[rand_mask] = np.random.choice(
                    len(self.vocab),
                    num_rand,
                    p=self.weights,
                )

        return torch.from_numpy(new_item)

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