import os
from glob import glob
import pickle

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
import torch.nn.functional as F

from random import randint, random

from griffon.coq_dataclasses import *
from griffon.utils import pad_mask, pad_list

from griffon.constants import NUM_SUB_TOKENS, MASK_TOKEN, PAD_TOKEN

class CounTDataset(IterableDataset):
    """
    Unites common functionalities used across different datasets such as applying the token mapping to the
    distance matrices and collating the matrices from multiple samples into one big tensor.
    """

    def __init__(self, root_path:str, vocab_path:str):

        assert os.path.exists(root_path), f"Path {root_path} does not exist"
        assert os.path.exists(vocab_path), f"Path {vocab_path} does not exist"

        self.files = sorted(glob(os.path.join(root_path, "*.pickle")))
        self.vocab = pickle.load(open(vocab_path, "rb"))

        self.pad_id = self.vocab[PAD_TOKEN]

    def to_dataloader(self):
        # TODO read more into typing sometimes...
        return DataLoader(self, collate_fn=self.collate_fn) # type: ignore

    def __next__(self):
        statement = next(self.statement_generator)
        return self.transform_statement(statement)

    def __iter__(self):

        def get_statement_generator():
            for file in self.files:
                sample:Stage2Sample = pickle.load(open(file, "rb"))
                for hypothesis in sample.hypotheses:
                    yield hypothesis
                yield sample.goal

        self.statement_generator = get_statement_generator()
        return self

    def transform_statement(self, statement: Stage2Statement)->CounTSample:

        def create_random_token():
            number_sub_tokens = min(np.random.geometric(0.4), NUM_SUB_TOKENS)

            itos = self.vocab.get_itos()
            subtokens = [itos[randint(0, len(self.vocab)-1)] for _ in range(number_sub_tokens)]
            return Stage2Token(self.vocab(subtokens), subtokens)

        mask  = []
        target_ids = []

        for i, token in enumerate(statement.tokens):
            prob = random()
            if prob > 0.15:
                mask.append(0)
                continue

            mask.append(1)
            target_ids.append(token.subtokens)

            # now we check if we want to
            # 1) mask the token (80% chance)
            # 2) replace the token by a random word (10% chance)
            # 3) leave the token intact

            prob /= 0.15
            if prob < 0.8:
                statement.tokens[i] = Stage2Token([self.vocab[MASK_TOKEN]] * NUM_SUB_TOKENS,
                                        [MASK_TOKEN]             * NUM_SUB_TOKENS)
            elif prob < 0.9:
                statement.tokens[i] = create_random_token()

        target_mask = torch.tensor(mask).bool()
        target_ids = torch.tensor(
            [pad_list(subtoken_ids, NUM_SUB_TOKENS, self.pad_id) for subtoken_ids in target_ids]
        )
        input_ids = torch.tensor(
            [pad_list(token.subtokens, NUM_SUB_TOKENS, self.pad_id) for token in
             statement.tokens]
        )

        distance_indices = torch.stack([distance[0] for distance in statement.distances])
        distance_bins = torch.stack([distance[1] for distance in statement.distances])

        return CounTSample(
            input_ids = input_ids,
            distance_indices = distance_indices,
            distance_bins = distance_bins,
            target_ids = target_ids,
            target_mask = target_mask
        )

    def collate_fn(self, samples: List[CounTSample])->CounTBatch:

        # we need to go through every tensor and pad it before stacking them along
        # the batching digm

        max_input_length = max([sample.input_ids.shape[0] for sample in samples])
        max_target_length = max([sample.target_ids.shape[0] for sample in samples])

        # only tensor where no padding is needed
        distance_bins_list = [sample.distance_bins for sample in samples]

        input_ids_list = []
        distance_indices_list = []
        target_ids_list = []
        target_mask_list = []

        input_lengths = []
        target_lengths = []

        for sample in samples:

            # first we pad everything depending on number of tokens
            input_length = sample.input_ids.shape[0]
            input_lengths.append(input_length)
            pad_length = max_input_length - input_length

            input_ids_list.append(
                F.pad(sample.input_ids, [0, 0, 0, pad_length], value=0)
            )
            target_mask_list.append(
                F.pad(sample.target_mask, [0, 0, 0, pad_length], value=0)
            )
            distance_indices_list.append(
                F.pad(sample.distance_indices, [0, pad_length, 0, pad_length], value=0)
            )

            # now we pad tensors that depend on number of *selected* tokens
            target_length = sample.input_ids.shape[0]
            target_lengths.append(target_length)
            pad_length = max_target_length - target_length

            target_ids_list.append(
                F.pad(sample.target_ids, [0, 0, 0, pad_length], value = 0)
            )

        input_ids = torch.stack(input_ids_list)
        distance_indices = torch.stack(distance_indices_list)
        distance_bins = torch.stack(distance_bins_list)
        target_ids = torch.stack(target_ids_list)
        target_mask = torch.stack(target_mask_list)

        input_padding_mask = pad_mask(torch.tensor(input_lengths), max_input_length)
        target_padding_mask = pad_mask(torch.tensor(target_lengths), max_target_length)

        return CounTBatch(
            input_ids = input_ids,
            distance_indices = distance_indices,
            distance_bins = distance_bins,
            target_ids = target_ids,
            target_mask = target_mask,
            input_padding_mask = input_padding_mask,
            target_padding_mask = target_padding_mask
        )


