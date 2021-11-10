from io import TextIOWrapper
import os
import json
import argparse
from typing import Dict, List, Tuple

from lark.exceptions import UnexpectedCharacters, ParseError

from CoqGym.ASTactic.tac_grammar import CFG, TreeBuilder, NonterminalNode, TerminalNode
from CoqGym.gallina import GallinaTermParser
from CoqGym.utils import iter_proofs, SexpCache, par_iter_proofs
from griffon.preprocessing.pipeline.stage1.recreate_term import Stage1StatementCreator

from griffon.utils import find_in_list, get_path_relative_to_data
from griffon.coq_dataclasses import GallinaObject, GoalObject, RelatedSentences, Stage1Sample, Stage1Statement

import pickle

def process_dataset(projs_split : Dict, args:argparse.Namespace, threads:int= 4)->Dict[str, List[RelatedSentences]]:

    num_discarded = 0
    total = 0

    split_directories:List[str] = ["train", "valid", "test"]

    file_handles : Dict[str, Dict[str, TextIOWrapper]] = {}
    for dir in split_directories:
        dirpath = os.path.join(args.output, dir)
        if not os.path.exists(dirpath):
                os.makedirs(dirpath)

    statement_creator = Stage1StatementCreator(GallinaTermParser(caching=False))
    sexp_cache = SexpCache(args.sexp_cache, readonly=True)

    grammar = CFG(args.tactic_grammar, "tactic_expr")
    tree_builder = TreeBuilder(grammar)

    def tactic2actions(tac_str):
        """my super function

        Args:
            tac_str (str): a tactic string

        Returns:
            [some json]: the return
        """
        tree = tree_builder.transform(grammar.parser.parse(tac_str))
        assert tac_str.replace(" ", "") == tree.to_tokens().replace(" ", "")
        actions = []

        def gather_actions(node):
            if isinstance(node, NonterminalNode):
                actions.append(grammar.production_rules.index(node.action))
            else:
                assert isinstance(node, TerminalNode)
                actions.append(node.token)

        tree.traverse_pre(gather_actions)
        return actions

    def parse_goal(goal_to_parse)->Tuple[List[Stage1Statement], Stage1Statement]:
        """parse a goal

        Args:
            goal ([type]): the goal to be parsed

        Returns:
            [type]: the returned thing
        """
        goal_statement = statement_creator(sexp_cache[goal_to_parse["sexp"]], "goal")

        local_context:List[GallinaObject] = []
        for _, hypothesis in enumerate(goal_to_parse["hypotheses"]):
            for ident in hypothesis["idents"]:
                sexp = sexp_cache[hypothesis["sexp"]]
                local_context.append(statement_creator(sexp, ident))

        return local_context, goal_statement

    def process_proof(index, filename, proof_data):
        nonlocal num_discarded
        nonlocal total

        rel_filename = get_path_relative_to_data(args.data_root, filename)
        proj = rel_filename.split(os.path.sep)[0]

        if proj in projs_split["projs_train"]:
            split = "train"
        elif proj in projs_split["projs_valid"]:
            split = "valid"
        else:
            split = "test"

        index_directory = f"file{index}"
        dirpath = os.path.join(args.output, split, index_directory)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

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
                index = 0
                if type(action) == str:
                    total += 1
                    # search in local context
                    res = find_in_list(local_context, lambda x : x.name == action)
                    lemma = None
                    if res != None:
                        lemma = res.tokens
                    # search in environment
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
                        sample = Stage1Sample(local_context, goal, lemma)
                        filename = '{:08d}.pickle'.format(index)
                        index += 1
                        path = os.path.join(args.output, split, filename)
                        pickle.dump(sample, open(path, 'wb'))
                    else:
                        #discard
                        num_discarded += 1

    # create the directories once so we don't have to check everytime if the directory
    # in the processing function
    for split in ["train", "valid", "test"]:
        dirname = os.path.join(args.output, split)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    par_iter_proofs(
        args.data_root, process_proof, include_synthetic=False, show_progress=True, threads=threads
    )

    print(f"\ndiscarded {num_discarded} out of {total}")

if __name__ == "__main__":

    print(os.getcwd())
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
    args = arg_parser.parse_args()
    setattr(args, "data_root",      os.path.join(args.coq_gym_root, "data"))
    setattr(args, "splits_file",    os.path.join(args.coq_gym_root, "projs_split.json"))
    setattr(args, "tactic_grammar", os.path.join(args.coq_gym_root, "ASTactic", "tactics.ebnf"))
    setattr(args, "sexp_cache",     os.path.join(args.coq_gym_root, "sexp_cache"))
    projs_split = json.load(open(args.splits_file, "r"))

    # if os.path.exists(args.output):
    #     print("output already exists, not doing anything to be safe")
    #     exit()

    process_dataset(projs_split, args)

    print(f"Output saved to {args.output}")
