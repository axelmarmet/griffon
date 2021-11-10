import argparse
from argparse import Namespace
import shutil

import json
import os

from griffon.utils import get_path_relative_to_data

from CoqGym.utils import iter_proofs


def write_corpus(args:Namespace):

    projs_split = json.load(open(args.splits_file))

    def process_proof(filename, proof_data):
        rel_filename = get_path_relative_to_data(args.data_root, filename)
        proj = rel_filename.split(os.path.sep)[0]
        if not proj in projs_split["projs_train"]:
            return

        goal_id = proof_data["steps"][0]["goal_ids"]["fg"][0]
        goal_text = proof_data["goals"][str(goal_id)]["type"]
        if goal_text is None:
            return

        text_filename = os.path.join(args.corpus_root, rel_filename).replace(".json", ".txt", 1)
        dirname, _ = os.path.split(text_filename)

        if not os.path.exists(dirname):
                os.makedirs(dirname)

        with open(text_filename, "a") as file:
            file.write(goal_text + "\n")

    iter_proofs(
        args.data_root, process_proof, include_synthetic=False, show_progress=True
    )

# def tokenize(inp:str):

arg_parser = argparse.ArgumentParser(
        description="""
            Write a corpus from the CoqGym dataset
        """
    )

arg_parser.add_argument(
    "--coq_gym_root", type=str, required=True, help="The root of the coq gym project"
)

arg_parser.add_argument(
    "--corpus_root", type=str, default="./corpus", help="The folder where the corpus will be written"
)
args = arg_parser.parse_args()

setattr(args, "data_root",      os.path.join(args.coq_gym_root, "data"))
setattr(args, "splits_file",    os.path.join(args.coq_gym_root, "projs_split.json"))

if os.path.exists(args.corpus_root):
    shutil.rmtree(args.corpus_root)

write_corpus(args)
