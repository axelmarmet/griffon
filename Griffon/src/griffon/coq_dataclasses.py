"""
    various dataclasses
"""
from collections import namedtuple
from dataclasses import dataclass

from typing import List, Dict

from CoqGym.gallina import GallinaTermParser
from CoqGym.utils import SexpCache

import networkx as nx

from lark.tree import Tree
from torch import Tensor


@dataclass
class GoalObject:
    """Class for general object"""
    id: int
    type_text: float
    ast: Tree

    @classmethod
    def from_dict(cls, parser: GallinaTermParser, cache: SexpCache, inp_dict: Dict):
        """Create GoalObject from dict
        Args:
            parser (GallinaTermParser):
            cache (SexpCache):
            inp_dict (Dict):

        Returns:
            [GoalObject]: the newly create GoalObject
        """
        return cls(
            id=inp_dict["id"],
            type_text=inp_dict["type"],
            ast=parser.parse(cache[inp_dict["sexp"]]))


@dataclass
class GallinaObject:
    """Class for context object"""
    ident: str
    type_text: str
    ast: Tree

    @classmethod
    def from_constant(cls, parser: GallinaTermParser, cache: SexpCache, inp_dict: Dict):
        """Create GallinaObject from dict
        Args:
            parser (GallinaTermParser):
            cache (SexpCache):
            inp_dict (Dict):

        Returns:
            [GallinaObject]: the newly create GallinaObject
        """
        return cls(
            ident=inp_dict["short_ident"],
            type_text=inp_dict["type"],
            ast=parser.parse(cache[inp_dict["sexp"]]))


@dataclass
class Stage1Token:
    subtokens : List[str]

@dataclass
class Stage2Token:
    subtokens : List[int]

Distance = namedtuple("Distance", ["distances", "bins", "name"])

@dataclass
class Stage1Statement:
    name :  str
    tokens : List[Stage1Token]
    ast: Dict[int, List[int]]
    token_to_node : Dict[int,int]

    def __str__(self):
        return self.name + " : " + " ".join(["_".join([str(subtoken) for subtoken in token.subtokens]) for token in self.tokens])

@dataclass
class Stage2Statement:
    name :  str
    tokens : List[Stage2Token]
    adjacency_matrix : Tensor
    distances : List[Distance]
    token_to_node : Dict[str,int]


@dataclass
class Stage1Sample:
    hypotheses : List[Stage1Statement]
    goal : Stage1Statement
    lemma_used : List[Stage1Token]

@dataclass
class Stage2Sample:
    hypotheses : List[Stage2Statement]
    goal : Stage2Statement
    lemma_used : List[Stage2Token]

@dataclass
class RelatedSentences:
    """Class holding everything needed for a
       training step"""
    local_context: List[GallinaObject]
    goal: GoalObject
    used_item: GallinaObject
