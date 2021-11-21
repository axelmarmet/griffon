import os
import pickle

from glob import glob
import torch

from typing import Dict, List, Tuple
from torch import Tensor

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from griffon.constants import NUM_SUB_TOKENS, SUB_TOKEN_PAD, UNK_TOKEN
from griffon.coq_dataclasses import CTCoqStatement, Stage1Token, Stage2Sample, Stage2Statement, Stage2Token
from griffon.utils import pad_list


class CTCoqDataset(Dataset):
    """
    Unites common functionalities used across different datasets such as applying the token mapping to the
    distance matrices and collating the matrices from multiple samples into one big tensor.
    """

    def __init__(self, root_path:str, vocab_path:str):

        assert os.path.exists(root_path), f"Path {root_path} does not exist"
        assert os.path.exists(vocab_path), f"Path {vocab_path} does not exist"

        self.files = glob(os.path.join(root_path, "*.pickle"))
        self.vocab = pickle.load(open(vocab_path, "rb"))

        self.unk_id = self.vocab[UNK_TOKEN]
        self.sub_token_pad_id = self.vocab[SUB_TOKEN_PAD]

        self.num_sub_tokens = NUM_SUB_TOKENS


    def to_dataloader(self):
        return DataLoader(self, collate_fn=self.collate_fn)

    def __next__(self):
        sample = next(self.dataset)
        return self.transform_sample(sample)

    def transform_statement(self, statement: Stage2Statement,
                            extended_vocab:Dict[str,int])->CTCoqStatement:

        sequence = torch.tensor(
            [pad_list(token.sub_tokens, self.num_sub_tokens, self.sub_token_pad_id) for token in
             statement.tokens])

        extended_vocabulary_ids = []
        # Generating extended vocabulary for pointer network. The extended vocabulary essentially mimics an infinite
        # vocabulary size for this sample
        len_vocab = len(self.word_vocab)
        for idx_token, token in enumerate(statement.tokens):
            for idx_subtoken, subtoken in enumerate(token.sub_tokens):
                if subtoken == self.unk_id:
                    original_subtoken = token.original_subtokens[idx_subtoken]

                    if original_subtoken in extended_vocab:
                        extended_id = extended_vocab[original_subtoken]
                    else:
                        extended_id = len_vocab + len(extended_vocab)
                        extended_vocab[original_subtoken] = extended_id
                    extended_vocabulary_ids.append(extended_id)
                else:
                    extended_vocabulary_ids.append(subtoken)

        pointer_pad_mask = sequence != self.sub_token_pad_id

        return CTCoqStatement(tokens=sequence,
                              extended_vocabulary_ids=extended_vocabulary_ids,
                              pointer_pad_mask=pointer_pad_mask,
                              distances=statement.distances,
                              token_to_node=statement.token_to_node)

    def transform_lemma(self, lemma : List[Stage2Token], extended_vocab:Dict[str,int])->Tensor:

        def transform_lemma_token(token : Stage1Token)->Stage2Token:
            new_subtokens = []
            for subtoken in token.subtokens:
                if subtoken in self.vocab:
                    new_subtokens.append(self.vocab[subtoken])
                else:
                    if subtoken in extended_vocab:
                        new_subtokens.append(extended_vocab[subtoken])
                    else:
                        new_subtokens.append(self.unk_id)
            return Stage2Token(new_subtokens, token.subtokens)

        return torch.tensor(
            [pad_list(
                transform_lemma_token(token),
                self.num_sub_tokens,
                self.sub_token_pad_id) for token in lemma])


    def transform_sample(self, sample: Stage2Sample):
        """
        Transforms a sample into torch tensors, applies sub token padding (which is independent of the other samples in
        a batch) and applies the token mapping onto the distance matrices (which makes them much bigger)
        """
        extended_vocab = {}

        transformed_hypotheses = [self.transform_statement(
                                    hypothesis,
                                    extended_vocab)
                                  for hypothesis in sample.hypotheses]

        transformed_goal = self.transform_statement(goal,)

