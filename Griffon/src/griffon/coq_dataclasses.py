"""
    various dataclasses
"""
from dataclasses import dataclass

from typing import List, Dict

from CoqGym.gallina import GallinaTermParser
from CoqGym.utils import SexpCache

import networkx as nx

from lark.tree import Tree

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
class Stage1Statement:
    name :  str
    tokens : List[str]
    ast: nx.Graph
    token_to_node : Dict[str,int]

# @dataclass
# class Stage2Statement:


@dataclass
class Stage1Sample:
    hypotheses : List[Stage1Statement]
    goal : Stage1Statement
    lemma_used : List[str]


@dataclass
class RelatedSentences:
    """Class holding everything needed for a
       training step"""
    local_context: List[GallinaObject]
    goal: GoalObject
    used_item: GallinaObject
