import os
import pickle

from glob import glob
import torch

from typing import Dict, List, Tuple
from torch import Tensor

import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from griffon.constants import EOS_TOKEN, MAX_NUM_TOKEN, NUM_SUB_TOKENS, PAD_TOKEN, SOS_TOKEN, TGT_IGNORE_INDEX, UNK_TOKEN
from griffon.coq_dataclasses import GriffonBatch, GriffonLemma, GriffonSample, GriffonStatement, Stage1Token, Stage2Sample, Stage2Statement, Stage2Token
from griffon.utils import pad_list, pad_mask

class GriffonDataset(Dataset):

    def __init__(self, root_path:str, split:str):

        assert os.path.exists(root_path), f"Path {root_path} does not exist"
        assert split in ["train", "test", "valid"], f"Split {split} is not supported"

        sample_dir = os.path.join(root_path, split)
        assert os.path.exists(sample_dir), f"Sample directory {sample_dir} does not exist"

        self.files = sorted(glob(os.path.join(sample_dir, "*.pickle")))
        self.vocab = pickle.load(open(os.path.join(root_path, "vocab.pkl"), "rb"))

        self.pad_id:int = self.vocab[PAD_TOKEN]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index)->GriffonSample:
        sample:GriffonSample = pickle.load(open(self.files[index], "rb"))
        return sample

    def simple_collate_fn(self, samples:List[GriffonSample])->GriffonSample:
        assert len(samples) == 1
        return samples[0]

    def to_dataloader(self, batch_size:int, num_workers:int):
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=self.collate_fn, # type: ignore
            pin_memory=True,
            num_workers=num_workers)

    def collate_fn(self, samples:List[GriffonSample])->GriffonBatch:

        max_number_statements = max((sample.sequences.shape[0] for sample in samples))
        max_number_tokens_statements =  max((sample.sequences.shape[1] for sample in samples))

        max_number_tokens_lemma = max((sample.lemma.shape[0] for sample in samples))

        batch_statements : List[Tensor] = []
        batch_extended_vocabulary_ids : List[Tensor] = []

        batch_distances_indices : List[Tensor] = []
        batch_distances_bins : List[Tensor] = []

        batch_statement_token_paddings : List[Tensor] = []

        batch_lemmas : List[Tensor] = []
        batch_lemma_token_paddings : List[Tensor] = []

        for sample in samples:
            statement_padding_amount = max_number_statements - sample.sequences.shape[0]
            statement_token_padding_amount = max_number_tokens_statements - sample.sequences.shape[1]
            lemma_padding_amount = max_number_tokens_lemma - sample.lemma.shape[0]

            # we pad the statements
            statements = F.pad(sample.sequences,
                               [0, 0, 0, statement_token_padding_amount, 0, statement_padding_amount],
                               value=self.pad_id)
            batch_statements.append(statements)

            # we pad the vocabulary ids of the statements
            extended_vocabulary_ids = F.pad(sample.extended_vocabulary_ids,
                                            [0, 0, 0, statement_token_padding_amount, 0, statement_padding_amount],
                                            value=self.pad_id)
            batch_extended_vocabulary_ids.append(extended_vocabulary_ids)

            # we pad the distances indices
            distances_indices = F.pad(sample.distances_indices,
                                      [
                                        0, statement_token_padding_amount,
                                        0, statement_token_padding_amount,
                                        0, 0,
                                        0, statement_padding_amount],
                                      value=self.pad_id)
            batch_distances_indices.append(distances_indices)

            # we pad the distances bins
            distances_bins = F.pad(sample.distances_bins,
                                   [
                                       0, 0,
                                       0, 0,
                                       0, statement_padding_amount],
                                   value=0)
            batch_distances_bins.append(distances_bins)

            # we update the padding mask
            statement_token_padding = F.pad(sample.token_padding_mask,
                                            [
                                                0, statement_token_padding_amount,
                                                0, statement_padding_amount
                                            ],
                                            value=True)
            batch_statement_token_paddings.append(statement_token_padding)

            # we create the lemma padding mask
            lemma_mask = torch.ones((max_number_tokens_lemma)).bool()
            lemma_mask[torch.arange(sample.lemma.shape[0])] = False
            batch_lemma_token_paddings.append(lemma_mask)

            # we pad the lemmas
            lemma = F.pad(sample.lemma, [0, 0, 0, lemma_padding_amount], value=TGT_IGNORE_INDEX)
            batch_lemmas.append(lemma)


        statements:Tensor = torch.stack(batch_statements)
        extended_vocabulary_ids = torch.stack(batch_extended_vocabulary_ids)
        distances_indices = torch.stack(batch_distances_indices)
        distances_bins = torch.stack(batch_distances_bins)
        statement_token_padding = torch.stack(batch_statement_token_paddings)
        lemmas = torch.stack(batch_lemmas)
        lemma_token_padding = torch.stack(batch_lemma_token_paddings)
        extended_vocabularies = [sample.extended_vocabulary for sample in samples]

        batch = GriffonBatch(
            statements=statements,
            extended_vocabulary_ids=extended_vocabulary_ids,
            distances_indices=distances_indices,
            distances_bins=distances_bins,
            lemmas=lemmas,
            statement_token_padding = statement_token_padding,
            lemma_token_padding=lemma_token_padding,
            extended_vocabularies = extended_vocabularies
        )

        batch.validate()

        return batch
