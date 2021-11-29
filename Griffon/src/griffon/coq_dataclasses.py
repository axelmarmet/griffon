"""
    various dataclasses
"""
from collections import namedtuple
from dataclasses import dataclass

from typing import List, Dict, NamedTuple, Optional, Union

from CoqGym.gallina import GallinaTermParser
from CoqGym.utils import SexpCache

import networkx as nx

from lark.tree import Tree
from torch import Tensor


@dataclass
class Stage1Token:
    subtokens : List[str]

@dataclass
class Stage2Token:
    subtokens : List[int]
    original_subtokens : List[str]

class Distance(NamedTuple):
    distances: Tensor
    bins: Tensor
    name: str

@dataclass
class Stage1Statement:
    name :  str
    tokens : List[Stage1Token]
    vocabularized_tokens : Optional[List[Stage2Token]]
    ast: Dict[int, List[int]]
    token_to_node : List[int]

    def __str__(self):
        return self.name + " : " + " ".join(["_".join([str(subtoken) for subtoken in token.subtokens]) for token in self.tokens])

@dataclass
class Stage2Statement:
    name :  str
    tokens : List[Stage2Token]
    distances : List[Distance]

@dataclass
class Stage1Sample:
    hypotheses : List[Stage1Statement]
    goal : Stage1Statement
    lemma_used : List[Stage1Token]

@dataclass
class Stage2Sample:
    hypotheses : List[Stage2Statement]
    goal : Stage2Statement
    lemma_used : List[Stage1Token]  # intentionnaly still stage 1, we want subtokens instead
                                    # of ids, because we use an extended vocabulary

@dataclass
class CTCoqStatement:
    tokens                  : Tensor    # shape `number tokens x number_subtokens`
    extended_vocabulary_ids : List[int] # len `number of non pad subtokens`
    pointer_pad_mask        : Tensor    # shape `number tokens x number_subtokens`
    distances               : List[Distance]

@dataclass
class CTCoqLemma:
    tokens : Tensor

@dataclass
class CTCoqSample:
    sequences               : Tensor # shape `number_statements x max_number_tokens x max_subtokens`
    extended_vocabulary_ids : Tensor # shape `number_statements x max_len_subtokens_in_seq`
    pointer_pad_mask        : Tensor # shape `number_statements x max_number_tokens x max_subtokens`
    distances_index         : Tensor # shape `number_statements x number_distances x max_number_tokens x max_number_tokens`
    distances_bins          : Tensor # shape `number_statements x number_distances x number_bins`

    # To only use actual info in all of the above tensors
    padding_mask            : Tensor # shape `number_statements x max_number_tokens`

    lemma                   : Tensor  # shape `number_tokens x max_subtokens`
    extended_vocabulary     : Dict[str,int]

    def validate(self):
        statements = [0,0,0,0,0,0]
        tokens = [0,0,0,0,0]
        subtokens = [0,0]
        distances = [0,0]

        statements[0], tokens[0], subtokens[0] = self.sequences.shape
        statements[1], _ = self.extended_vocabulary_ids.shape
        statements[2], tokens[1], subtokens[1] = self.pointer_pad_mask.shape
        statements[3], distances[0], tokens[2], tokens[3] = self.distances_index.shape
        statements[4], distances[1], _ = self.distances_bins.shape
        statements[5], tokens[4] = self.padding_mask.shape

        assert (all(statements[0] == s for s in statements)), "Not all tensors have the same number of statements"
        assert (all(tokens[0] == t for t in tokens)), "Not all tensors have the same number of tokens"
        assert (all(subtokens[0] == st for st in subtokens)), "Not all tensors have the same number of subtokens"
        assert (all(distances[0] == d for d in distances)), "Not all tensors have the same number of distances"

# @dataclass
# class CTCoqBatch:
#     sequences               : Tensor # shape `max_number_statements x max_number_tokens x max_subtokens`
#     extended_vocabulary_ids : Tensor # shape `max_number_statements x max_len_subtokens_in_seq`
#     pointer_pad_mask        : Tensor # shape `max_number_statements x max_number_tokens x max_subtokens`
#     distances_index         : Tensor # shape `max_number_statements x number_distances x max_number_tokens x max_number_tokens`
#     distances_bins          : Tensor # shape `max_number_statements x number_distances x number_bins`

#     # To only use actual info in all of the above tensors
#     padding_mask            : Tensor # shape `number_statements x max_number_tokens`

#     lemma                   : Tensor  # shape `number_tokens x max_subtokens`
#     extended_vocabularies   : List[Dict[str,int]]


@dataclass
class CounTSample:
    input_ids               : Tensor    # shape `number tokens x number_subtokens`
    distance_indices        : Tensor    # shape `number distances x number tokens x number_tokens`
    distance_bins           : Tensor    # shape `number distances x number bins`
    target_ids              : Tensor    # shape `number tokens x number_subtokens`

@dataclass
class CounTInput:
    input_ids               : Tensor    # shape `batch x max number tokens x number_subtokens`
    distance_indices        : Tensor    # shape `batch x number distances x max number tokens x max number_tokens`
    distance_bins           : Tensor    # shape `batch x number distances x number bins`
    # To only use actual info in all of the above tensors
    input_padding_mask            : Tensor    # shape `batch x max_number_tokens`

    def to(self, *args):
        self.input_ids = self.input_ids.to(*args)
        self.distance_indices = self.distance_indices.to(*args)
        self.distance_bins    = self.distance_bins.to(*args)
        self.input_padding_mask = self.input_padding_mask.to(*args)

@dataclass
class CounTTarget:
    target_ids              : Tensor    # shape `batch x max number selected tokens x number_subtokens`
    target_mask             : Tensor    # shape `batch x max number tokens`
    # To only use actual info in all of the above tensors
    target_padding_mask           : Tensor    # shape `batch x max number selected tokens`

    def to(self, *args):
        self.target_ids = self.target_ids.to(*args)
        self.target_mask = self.target_mask.to(*args)
        self.target_padding_mask = self.target_padding_mask.to(*args)

class CounTBatch(NamedTuple):
    input: CounTInput
    target: Tensor # shape `batch x max_number_tokens x num_subtokens`
