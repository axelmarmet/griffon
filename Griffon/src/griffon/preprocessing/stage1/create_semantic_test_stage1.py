import argparse
import json
import os
from re import S
from typing import Dict, List
from griffon.coq_dataclasses import Stage1Statement, Stage1Token
from griffon.utils import iter_proofs_in_file
from griffon.preprocessing.utils import connect_subtokens
from griffon.preprocessing.stage1.recreate_term import Stage1StatementCreator
from CoqGym.utils import SexpCache
from CoqGym.gallina import GallinaTermParser
from nltk.tokenize.regexp import RegexpTokenizer

from glob import glob
import pickle


def process_file(path:str, statement_creator:Stage1StatementCreator, sexp_cache:SexpCache):

    statements = []

    def process_proof(filename, proof_data):

        for step in proof_data["steps"]:
            # consider only tactics
            if step["command"][1] in [
                "VernacEndProof",
                "VernacBullet",
                "VernacSubproof",
                "VernacEndSubproof",
            ]:
                continue
            goal_id = step['goal_ids']['fg'][0]
            goal_to_parse = proof_data['goals'][str(goal_id)]
            goal_statement = statement_creator(sexp_cache[goal_to_parse["sexp"]], proof_data['name'])

            # we remove the fist token since it's the name of the hypothesis
            goal_statement.token_to_node = goal_statement.token_to_node[1:]
            goal_statement.tokens = goal_statement.tokens[1:]

            statements.append(goal_statement)
            break

    iter_proofs_in_file(
        path, process_proof
    )

    return statements

def dump_json(statements:Dict[str, List[Stage1Statement]], out_path:str):

    def transform_statements(statements : List[Stage1Statement])->Dict[str, List[str]]:
        return {statement.name : [connect_subtokens(token) for token in statement.tokens] for statement in statements}

    transformed_dict = {k : transform_statements(v) for k, v in statements.items()}

    json.dump(transformed_dict, open(out_path, "w"), indent=4)

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="""
            Extract the semantic test jsons
        """
    )
    arg_parser.add_argument(
        "--semantic_test_path", type=str, default="semantic_tests", help="The root folder of the semantic tests")

    args = arg_parser.parse_args()

    statement_creator = Stage1StatementCreator(GallinaTermParser(caching=False), RegexpTokenizer(r"[^\W_]+|[:,().]"))

    raw_dir = os.path.join(args.semantic_test_path, "raw")

    sexp_cache = SexpCache(
        os.path.join(raw_dir, "sexp_cache"), readonly=True)


    files = glob(os.path.join(raw_dir, "data", "**", "*.json"))
    assert len(files) > 0, "no files matched"

    output_dir = os.path.join(args.semantic_test_path, "stage1")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    statements_dict = {}
    for path in files:
        filename = path.split(os.path.sep)[-1].split(".")[0]
        assert filename not in statements_dict
        statements = process_file(path, statement_creator, sexp_cache)
        statements_dict[filename] = statements

        pickle.dump(statements, open(os.path.join(output_dir, f"{filename}.pkl"), "wb"))

    dump_json(statements_dict, os.path.join(args.semantic_test_path, "new_masks.json"))
