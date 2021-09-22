import os
import json
import pickle
from CoqGym.ASTactic.tac_grammar import CFG, TreeBuilder, NonterminalNode, TerminalNode
import sys

sys.setrecursionlimit(100000)
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
)
from CoqGym.gallina import GallinaTermParser
from lark.exceptions import UnexpectedCharacters, ParseError
from CoqGym.utils import iter_proofs, SexpCache
import argparse
from hashlib import md5
import pdb


term_parser = GallinaTermParser(caching=True)
sexp_cache = SexpCache("./sexp_cache", readonly=True)


def parse_goal(g):
    goal = {
        "id": g["id"],
        "text": g["type"],
        "ast": term_parser.parse(sexp_cache[g["sexp"]]),
    }
    local_context = []
    for i, h in enumerate(g["hypotheses"]):
        for ident in h["idents"]:
            local_context.append(
                {
                    "ident": ident,
                    "text": h["type"],
                    "ast": term_parser.parse(sexp_cache[h["sexp"]]),
                }
            )
    return local_context, goal


grammar = CFG("../CoqGymProject/CoqGym/ASTactic/tactics.ebnf", "tactic_expr")
tree_builder = TreeBuilder(grammar)

def tactic2actions(tac_str):
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

proof_steps = []

num_discarded = 0


def process_proof(filename, proof_data):
    if "entry_cmds" in proof_data:
        is_synthetic = True
    else:
        is_synthetic = False
    global num_discarded

    if args.filter and not md5(filename.encode()).hexdigest().startswith(args.filter):
        return

    for i, step in enumerate(proof_data["steps"]):
        # consider only tactics
        if step["command"][1] in [
            "VernacEndProof",
            "VernacBullet",
            "VernacSubproof",
            "VernacEndSubproof",
        ]:
            continue
        # only apply for now
        if not step["command"][0].startswith("apply"):
            continue

        assert step["command"][1] == "VernacExtend"
        assert step["command"][0].endswith(".")
        
        
        # local context & goal
        if step["goal_ids"]["fg"] == []:
            num_discarded += 1
            continue

        goal_id = step["goal_ids"]["fg"][0]
        local_context, goal = parse_goal(proof_data["goals"][str(goal_id)])
        # tactic
        tac_str = step["command"][0][:-1]
        try:
            actions = tactic2actions(tac_str)
        except (UnexpectedCharacters, ParseError) as ex:
            num_discarded += 1
            continue
        proof_steps.append(
            {
                "file": filename,
                "proof_name": proof_data["name"],
                "n_step": i,
                "local_context": local_context,
                "goal": goal,
                "tactic": {"text": tac_str, "actions": actions},
            }
        )

        if is_synthetic:
            proof_steps[-1]["is_synthetic"] = True
            proof_steps[-1]["goal_id"] = proof_data["goal_id"]
            proof_steps[-1]["length"] = proof_data["length"]
        else:
            proof_steps[-1]["is_synthetic"] = False


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Extract the proof steps from CoqGym for trainig ASTactic via supervised learning"
    )
    arg_parser.add_argument(
        "--data_root", type=str, default="./data", help="The folder for CoqGym"
    )
    arg_parser.add_argument(
        "--output", type=str, default="./proof_steps/", help="The output file"
    )
    arg_parser.add_argument("--filter", type=str, help="filter the proofs")
    args = arg_parser.parse_args()
    print(args)

    iter_proofs(
        args.data_root, process_proof, include_synthetic=False, show_progress=True
    )

    for split in ["train", "valid"]:
        for i, step in enumerate(proof_steps[split]):
            dirname = os.path.join(args.output, split)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            if args.filter:
                pickle.dump(
                    step,
                    open(
                        os.path.join(dirname, "%s-%08d.pickle" % (args.filter, i)), "wb"
                    ),
                )
            else:
                pickle.dump(step, open(os.path.join(dirname, "%08d.pickle" % i), "wb"))

    print("\nOutput saved to ", args.output)
