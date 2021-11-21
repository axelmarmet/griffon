"""
    various dataclasses
"""
from collections import namedtuple
from dataclasses import dataclass

from typing import List, Dict, Optional, Union

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

Distance = namedtuple("Distance", ["distances", "bins", "name"])

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
    token_to_node : List[int]


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
    tokens : Tensor
    extended_vocabulary_ids : List[int]
    pointer_pad_mask : Tensor
    distances : List[Distance]
    token_to_node : Dict[int,int]

@dataclass
class CTCoqLemma:
    tokens : Tensor

@dataclass
class CTCoqSample:
    hypotheses : List[CTCoqStatement]
    goal : CTCoqStatement
    lemma_used : CTCoqLemma
    extended_vocabulary : Dict[str,int]

