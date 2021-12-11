import os
from glob import glob
import pickle
from typing import Tuple

import json

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from random import randint, random

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

from griffon.coq_dataclasses import *
from griffon.dataset.count_dataset import CounTDataset
from griffon.preprocessing.pipeline.stage2.create_semantic_test_stage2 import verify_masks
from griffon.utils import pad_mask, pad_list

from griffon.constants import NUM_SUB_TOKENS, MASK_TOKEN, PAD_TOKEN, TGT_IGNORE_INDEX

@dataclass
class SemanticTestCases:
    filename           : str
    batch              : CounTBatch
    names              : List[str]
    original_sentences : List[str]
    masked_sentences   : List[str]

    def to(self, *args):
        self.batch.to(*args)

class SemanticTestCaseDataset(Dataset):

    def __init__(self, sample_paths:str, vocab_path:str, mask_file:str):

        assert os.path.exists(sample_paths), f"Path {sample_paths} does not exist"
        assert os.path.exists(mask_file), f"Path {mask_file} does not exist"

        self.files = sorted(glob(os.path.join(sample_paths, "*.pkl")))
        self.masks = json.load(open(mask_file, "r"))
        self.vocab = pickle.load(open(os.path.join(vocab_path, "vocab.pickle"), "rb"))
        self.pad_id = self.vocab[PAD_TOKEN]
        self.mask_id = self.vocab[MASK_TOKEN]

    def transform_to_count_sample(self, statement:Stage2Statement, masked_statement:List[str])->CounTSample:

        target_ids = []

        assert len(statement.tokens) == len(masked_statement)

        for i, (token, masked_token) in enumerate(zip(statement.tokens, masked_statement)):

            if masked_token == MASK_TOKEN:
                target_ids.append(token.subtokens)
                statement.tokens[i] = Stage2Token([self.mask_id] * NUM_SUB_TOKENS,
                                                  [MASK_TOKEN]   * NUM_SUB_TOKENS)
            else:
                target_ids.append([TGT_IGNORE_INDEX]*NUM_SUB_TOKENS)

        target_ids = torch.tensor(
            [pad_list(subtoken_ids, NUM_SUB_TOKENS, TGT_IGNORE_INDEX) for subtoken_ids in target_ids]
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
        )

    def to_dataloader(self, batch_size:int, sampler:Optional[Sampler]=None):
        return DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=self.collate_fn, # type: ignore
            pin_memory=True,
            num_workers=4)

    def __getitem__(self, index:int)->SemanticTestCases:

        filename = self.files[index].split(os.path.sep)[-1].split(".")[0]

        masked_statements = self.masks[filename]
        statements:List[Stage2Statement] = pickle.load(open(self.files[index], "rb"))

        count_samples:List[CounTSample] = []
        sentence_names = []
        original_sentences = []
        masked_sentences = []

        for statement in statements:
            original_sentence = str(statement)
            masked_sentence = masked_statements[statement.name]
            count_samples.append(self.transform_to_count_sample(statement, masked_sentence))
            sentence_names.append(statement.name)
            original_sentences.append(original_sentence)
            masked_sentences.append(masked_sentence)

        count_batch = CounTDataset.collate_fn([sample for sample in count_samples])
        return SemanticTestCases(
            filename,
            batch=count_batch,
            names=sentence_names,
            original_sentences=original_sentences,
            masked_sentences=masked_sentences
        )

    def __len__(self) -> int:
        return len(self.files)

