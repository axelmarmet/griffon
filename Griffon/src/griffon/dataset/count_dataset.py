import os
from glob import glob
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset

from griffon.coq_dataclasses import *
from griffon.dataset.bucket_sampler import BucketSampler
from griffon.utils import pad_mask

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

        self.weights = np.full(len(self.vocab), 1/len(self.vocab))

        # check that a random subset is sorted
        self.probabilistic_check()

    def probabilistic_check(self):
        indices = np.random.choice(len(self), 100, replace=False)
        for i, index_i in enumerate(indices):
            for index_j in indices[i+1:]:
                sample_i:PreCounTSample = pickle.load(open(self.files[index_i], "rb"))
                sample_j:PreCounTSample = pickle.load(open(self.files[index_j], "rb"))
                if index_i < index_j:
                    assert sample_i.input_ids.shape[0] < sample_j.input_ids.shape[0], \
                        f"file {self.files[index_i]} should have less tokens than" \
                        f"file {self.files[index_j]}, did you sort the dataset?"
                else:
                    assert sample_i.input_ids.shape[0] > sample_j.input_ids.shape[0], \
                        f"file {self.files[index_i]} should have more tokens than" \
                        f"file {self.files[index_j]}, did you sort the dataset?"


    def to_dataloader(self, batch_size:int, num_workers:int):
        inner_sampler = DistributedSampler(self, shuffle=True)
        bucket_sampler = BucketSampler(inner_sampler, batch_size)
        return DataLoader(
            self,
            batch_sampler=bucket_sampler,
            collate_fn=self.collate_fn, # type: ignore
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True)

    def __getitem__(self, index:int)->MaskedCounTSample:
        sample:PreCounTSample = pickle.load(open(self.files[index], "rb"))
        return self.mask_item(sample)

    def mask_item(self, sample:PreCounTSample)->MaskedCounTSample:
        """Heavily inspired from fairseq implementation

        Args:
            statement (PreCounTSample): [description]

        Returns:
            MaskedCounTSample: [description]
        """

        sz = sample.input_ids.shape[0]

        assert (
            (sample.input_ids != self.mask_id).all()
        ), f"Sample contains mask_idx ({self.mask_id}), this is not expected!"

        # decide elements to mask
        mask = np.full(sz, False)
        num_mask = int(
            # add a random number for probabilistic rounding
            self.mask_prob * sz + np.random.rand()
        )
        mask_idc = np.random.choice(sz, num_mask, replace=False)
        mask[mask_idc] = True

        target_ids = torch.empty_like(sample.input_ids).fill_(TGT_IGNORE_INDEX)
        target_ids[torch.from_numpy(mask)] = sample.input_ids[torch.from_numpy(mask)]

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

        new_input_ids = sample.input_ids
        new_input_ids[torch.from_numpy(mask)] = self.mask_id

        if rand_mask is not None:
            num_rand = rand_mask.sum()
            if num_rand > 0:
                num_rand_subtokens = np.random.geometric(0.4, (num_rand,1))
                # clamp so that we don't go further than possible
                num_rand_subtokens[num_rand_subtokens > NUM_SUB_TOKENS] = NUM_SUB_TOKENS

                random_subtokens = np.random.choice(
                    len(self.vocab),
                    (num_rand, NUM_SUB_TOKENS),
                    p=self.weights,
                )
                random_subtokens[np.arange(NUM_SUB_TOKENS)[np.newaxis] >= num_rand_subtokens] = self.pad_id
                new_input_ids[torch.from_numpy(rand_mask)] = torch.from_numpy(random_subtokens)

        return MaskedCounTSample(
            input_ids=new_input_ids,
            distance_indices=sample.distance_indices,
            distance_bins=sample.distance_bins,
            target_ids=target_ids
        )

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def collate_fn(samples: List[MaskedCounTSample])->CounTBatch:

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