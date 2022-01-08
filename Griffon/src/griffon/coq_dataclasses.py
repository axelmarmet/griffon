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
import torch


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

    def __str__(self):
        return self.name + " : " + " ".join(["_".join([str(subtoken) for subtoken in token.original_subtokens]) for token in self.tokens])

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
class GriffonStatement:
    tokens                  : Tensor    # shape `number tokens x number_subtokens`
    extended_vocabulary_ids : List[List[int]] # shape `number tokens x number_subtokens`
    pointer_pad_mask        : Tensor    # shape `number tokens x number_subtokens`
    distances               : List[Distance]

@dataclass
class GriffonLemma:
    tokens : Tensor

@dataclass
class GriffonSample:
    sequences               : Tensor # shape `number_statements x max_number_tokens x max_subtokens`
    extended_vocabulary_ids : Tensor # shape `number_statements x max_number_tokens x max_subtokens`
    distances_indices         : Tensor # shape `number_statements x number_distances x max_number_tokens x max_number_tokens`
    distances_bins          : Tensor # shape `number_statements x number_distances x number_bins`

    # To only use actual info in all of the above tensors
    token_padding_mask      : Tensor # shape `number_statements x max_number_tokens`
    pointer_pad_mask        : Tensor # shape `number_statements x max_number_tokens x max_subtokens`

    lemma                   : Tensor  # shape `number_tokens x max_subtokens`
    extended_vocabulary     : Dict[str,int]

    def validate(self):
        statements = [0,0,0,0,0,0]
        tokens = [0,0,0,0,0,0]
        subtokens = [0,0,0]
        distances = [0,0]

        statements[0], tokens[0], subtokens[0] = self.sequences.shape
        statements[1], tokens[1], subtokens[1] = self.extended_vocabulary_ids.shape
        statements[2], tokens[2], subtokens[2] = self.pointer_pad_mask.shape
        statements[3], distances[0], tokens[3], tokens[4] = self.distances_indices.shape
        statements[4], distances[1], _ = self.distances_bins.shape
        statements[5], tokens[5] = self.token_padding_mask.shape

        assert (all(statements[0] == s for s in statements)), "Not all tensors have the same number of statements"
        assert (all(tokens[0] == t for t in tokens)), "Not all tensors have the same number of tokens"
        assert (all(subtokens[0] == st for st in subtokens)), "Not all tensors have the same number of subtokens"
        assert (all(distances[0] == d for d in distances)), "Not all tensors have the same number of distances"

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.extended_vocabulary_ids = self.extended_vocabulary_ids.pin_memory()
        self.pointer_pad_mask = self.pointer_pad_mask.pin_memory()
        self.distances_indices = self.distances_indices.pin_memory()
        self.distances_bins = self.distances_bins.pin_memory()
        self.token_padding_mask = self.token_padding_mask.pin_memory()
        self.lemma = self.lemma.pin_memory()
        return self

    def to(self, *args):
        self.sequences = self.sequences.to(*args)
        self.extended_vocabulary_ids = self.extended_vocabulary_ids.to(*args)
        self.pointer_pad_mask = self.pointer_pad_mask.to(*args)
        self.distances_indices = self.distances_indices.to(*args)
        self.distances_bins = self.distances_bins.to(*args)
        self.token_padding_mask = self.token_padding_mask.to(*args)
        self.lemma = self.lemma.to(*args)
        return self

@dataclass
class GriffonBatch:
    statements              : Tensor # shape `batch x max_number_statements x max_number_tokens x max_subtokens`
    extended_vocabulary_ids : Tensor # shape `batch x number_statements x max_number_tokens x max_subtokens`
    distances_indices       : Tensor # shape `batch x max_number_statements x number_distances x max_number_tokens x max_number_tokens`
    distances_bins          : Tensor # shape `batch x max_number_statements x number_distances x number_bins`
    lemmas                  : Tensor # shape `batch x number_lemma_tokens x max_subtokens`

    # To only use actual info in all of the above tensors
    statement_token_padding : Tensor # shape `batch x number_statements x max_number_tokens`
    lemma_token_padding     : Tensor # shape `batch x number_lemma_tokens`

    extended_vocabularies   : List[Dict[str,int]]

    def validate(self):
        batches = [0,0,0,0,0,0,0]
        statements = [0,0,0,0,0]
        tokens = [0,0,0,0,0]
        lemma_tokens = [0,0]
        subtokens = [0,0,0]
        distances = [0,0]

        batches[0], statements[0], tokens[0], subtokens[0] = self.statements.shape
        batches[1], statements[1], tokens[1], subtokens[1] = self.extended_vocabulary_ids.shape

        batches[2], statements[2], distances[0], tokens[2], tokens[3] = self.distances_indices.shape
        batches[3], statements[3], distances[1], _ = self.distances_bins.shape
        batches[4], statements[4], tokens[4] = self.statement_token_padding.shape
        batches[5], lemma_tokens[0], subtokens[2] = self.lemmas.shape
        batches[6], lemma_tokens[1] = self.lemma_token_padding.shape

        assert (all(batches[0] == b for b in batches)), "Not all tensors have the same amount of batches"
        assert (all(statements[0] == s for s in statements)), "Not all tensors have the same number of statements"
        assert (all(tokens[0] == t for t in tokens)), "Not all tensors have the same number of tokens"
        assert (all(lemma_tokens[0] == t for t in lemma_tokens)), "Not all tensors have the same number of lemma tokens"
        assert (all(subtokens[0] == st for st in subtokens)), "Not all tensors have the same number of subtokens"
        assert (all(distances[0] == d for d in distances)), "Not all tensors have the same number of distances"

        # check that all -1 subtokens are contained in the padding mask
        invalid_mask = (self.lemmas == -1).sum(dim=-1).bool()
        # check invalid_mask -> lemma_token_padding === !invalid_mask \/ lemma_token_padding
        assert torch.logical_or(torch.logical_not(invalid_mask), self.lemma_token_padding).all()

    def pin_memory(self):
        self.statements = self.statements.pin_memory()
        self.extended_vocabulary_ids = self.extended_vocabulary_ids.pin_memory()
        self.distances_indices = self.distances_indices.pin_memory()
        self.distances_bins = self.distances_bins.pin_memory()
        self.statement_token_padding = self.statement_token_padding.pin_memory()
        self.lemmas = self.lemmas.pin_memory()
        self.lemma_token_padding = self.lemma_token_padding.pin_memory()
        return self

    def to(self, *args):
        self.statements = self.statements.to(*args)
        self.extended_vocabulary_ids = self.extended_vocabulary_ids.to(*args)
        self.distances_indices = self.distances_indices.to(*args)
        self.distances_bins = self.distances_bins.to(*args)
        self.statement_token_padding = self.statement_token_padding.to(*args)
        self.lemmas = self.lemmas.to(*args)
        self.lemma_token_padding = self.lemma_token_padding.to(*args)
        return self

@dataclass
class CTCoqOutput:
    ...

@dataclass
class CounTSample:
    input_ids               : Tensor    # shape `number tokens x number_subtokens`
    distance_indices        : Tensor    # shape `number distances x number tokens x number_tokens`
    distance_bins           : Tensor    # shape `number distances x number bins`
    target_ids              : Tensor    # shape `number tokens x number_subtokens`

@dataclass
class CounTBatchInput:
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

    def pin_memory(self):
        self.input_ids = self.input_ids.pin_memory()
        self.distance_indices = self.distance_indices.pin_memory()
        self.distance_bins    = self.distance_bins.pin_memory()
        self.input_padding_mask = self.input_padding_mask.pin_memory()
        return self

@dataclass
class CounTBatch:
    input: CounTBatchInput
    target: Tensor # shape `batch x max_number_tokens x num_subtokens`

    def to(self, *args):
        self.input.to(*args)
        self.target = self.target.to(*args)

    def pin_memory(self):
        self.input = self.input.pin_memory()
        self.target = self.target.pin_memory()
        return self

    def as_tuple(self):
        return (self.input, self.target)
