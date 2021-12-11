from collections import defaultdict
from io import TextIOWrapper
import os
import json
import argparse
from argparse import Namespace
from typing import Dict, List, Optional, Tuple
from copy import deepcopy



import multiprocessing as mp

import itertools

from lark.exceptions import UnexpectedCharacters, ParseError

from CoqGym.ASTactic.tac_grammar import CFG, TreeBuilder, NonterminalNode, TerminalNode
from CoqGym.gallina import GallinaTermParser
from CoqGym.utils import iter_proofs, SexpCache

from nltk.tokenize.regexp import RegexpTokenizer
from griffon.constants import MAX_NUM_TOKEN
from griffon.preprocessing.pipeline.stage1.recreate_term import Stage1StatementCreator

from griffon.utils import find_in_list, get_path_relative_to_data, iter_proofs_in_file
from griffon.coq_dataclasses import Stage1Sample, Stage1Statement

from tqdm import tqdm

import pickle

from glob import glob


def init_process(args:Namespace):
    global statement_creator
    global sexp_cache
    global grammar
    global tree_builder

    statement_creator = Stage1StatementCreator(GallinaTermParser(caching=False), RegexpTokenizer(r"[^\W_]+|[:,().]"))
    sexp_cache = SexpCache(args.sexp_cache, readonly=True)
    grammar = CFG(args.tactic_grammar, "tactic_expr")
    tree_builder = TreeBuilder(grammar)


def process_file(args:Namespace, proof_path:str):

    directories = proof_path.split(os.path.sep)

    projs_split = json.load(open(args.splits_file, "r"))

    coq_project_index = directories.index("data") + 1
    coq_project = directories[coq_project_index]
    filename = directories[-1].split(".")[0]

    filename = "_".join(directories[coq_project_index+1 : -1] + [filename])

    proof_index = 0

    if coq_project in projs_split["projs_train"]:
        split = "train"
    elif coq_project in projs_split["projs_valid"]:
        split = "valid"
    elif coq_project in projs_split["projs_test"]:
        split = "test"
    else:
        raise  ValueError(f"{coq_project} not in splits.json")


    out_dirpath = os.path.join(args.output, split, coq_project, filename)
    if not os.path.exists(out_dirpath):
            os.makedirs(out_dirpath)

    def tactic2actions(tac_str):
        """my super function

        Args:
            tac_str (str): a tactic string

        Returns:
            [some json]: the return
        """
        tree = tree_builder.transform(grammar.parser.parse(tac_str))
        assert tac_str.replace(" ", "") == tree.to_tokens().replace(" ", "") # type: ignore
        actions = []

        def gather_actions(node):
            if isinstance(node, NonterminalNode):
                actions.append(grammar.production_rules.index(node.action))
            else:
                assert isinstance(node, TerminalNode)
                actions.append(node.token)

        tree.traverse_pre(gather_actions) # type: ignore
        return actions

    def parse_goal(goal_to_parse)->Tuple[List[Stage1Statement], Stage1Statement]:
        """parse a goal

        Args:
            goal ([type]): the goal to be parsed

        Returns:
            [type]: the returned thing
        """
        goal_statement = statement_creator(sexp_cache[goal_to_parse["sexp"]], "goal")

        local_context:List[Stage1Statement] = []
        for _, hypothesis in enumerate(goal_to_parse["hypotheses"]):
            for ident in hypothesis["idents"]:
                sexp = sexp_cache[hypothesis["sexp"]]
                local_context.append(statement_creator(sexp, ident))

        return local_context, goal_statement

    def process_proof(filename, proof_data):
        nonlocal proof_index

        for step in proof_data["steps"]:
            # consider only tactics
            if step["command"][1] in [
                "VernacEndProof",
                "VernacBullet",
                "VernacSubproof",
                "VernacEndSubproof",
            ]:
                continue

            # only apply and rewrite for now
            if not (step["command"][0].startswith("apply") or
                    step["command"][0].startswith("rewrite")):
                continue

            assert step["command"][1] == "VernacExtend"
            assert step["command"][0].endswith(".")

            # local context & goal
            if step["goal_ids"]["fg"] == []:
                continue

            goal_id = step["goal_ids"]["fg"][0]
            try:
                local_context, goal = parse_goal(proof_data["goals"][str(goal_id)])
            except AssertionError as exc:
                name = proof_data["name"]
                print(f"failed on proof {name} in {filename}")
                raise exc
            # tactic
            tac_str = step["command"][0][:-1]
            try:
                actions = tactic2actions(tac_str)
            except (UnexpectedCharacters, ParseError):
                continue
            for action in actions:
                if type(action) == str:
                    # search in local context
                    res = find_in_list(local_context, lambda x : x.name == action)
                    lemma = None
                    if res != None:
                        hypotheses = proof_data["goals"][str(goal_id)]['hypotheses']
                        hypothesis = find_in_list(hypotheses, lambda h : res.name in h['idents']) # type: ignore
                        assert hypothesis is not None
                        lemma = statement_creator.only_tokens(sexp_cache[hypothesis["sexp"]])                    # search in environment
                    if lemma is None:
                        res = find_in_list(proof_data["env"]["constants"], lambda x : x["short_ident"] == action)
                        if res != None:
                            try:
                                lemma = statement_creator.only_tokens(sexp_cache[res["sexp"]])
                            except AssertionError as exc:
                                name = proof_data["name"]
                                print(f"failed on proof {name} in {filename}")
                                raise exc

                    if lemma is not None:
                        # prune samples that are too big, a cutoff of 128 is sufficient to keep 98% of all samples
                        if len(goal.tokens) > MAX_NUM_TOKEN or len(lemma) > 2*MAX_NUM_TOKEN:
                            continue
                        local_context = [hypothesis for hypothesis in local_context if len(hypothesis.tokens) <= 128]

                        sample = Stage1Sample(local_context, goal, lemma)
                        out_filename = 'proof{:04d}.pickle'.format(proof_index)
                        proof_index += 1
                        path = os.path.join(out_dirpath, out_filename)
                        assert not os.path.exists(path), f"{path} already exists"
                        pickle.dump(sample, open(path, 'wb'))

    iter_proofs_in_file(
        proof_path, process_proof
    )

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="""
            Extract the proof steps from CoqGym for trainig ASTactic via supervised learning
        """
    )
    arg_parser.add_argument(
        "--coq_gym_root", type=str, default="../../../CoqGym/CoqGym", help="The root folder of CoqGym")
    arg_parser.add_argument(
        "--output", type=str, default="./recommandations/", help="The output file"
    )
    arg_parser.add_argument(
        "--threads", type=int, default=None, help="""
        The number of threads used for preprocessing (default: use all threads available)
        """
    )

    args = arg_parser.parse_args()
    setattr(args, "data_root",      os.path.join(args.coq_gym_root, "data"))
    setattr(args, "splits_file",    os.path.join(args.coq_gym_root, "projs_split.json"))
    setattr(args, "tactic_grammar", os.path.join(args.coq_gym_root, "ASTactic", "tactics.ebnf"))
    setattr(args, "sexp_cache",     os.path.join(args.coq_gym_root, "sexp_cache"))

    proof_files = glob(f"{args.data_root}/**/*.json", recursive=True)
    with mp.Pool(args.threads, initializer=init_process, initargs=[args]) as pool:
        pool.starmap(process_file, zip(itertools.repeat(args), proof_files))

    print(f"Output saved to {args.output}")
