import argparse
from glob import glob
import json
import os
import pickle

from argparse import Namespace
from typing import Dict, List
from griffon.constants import MASK_TOKEN

from griffon.coq_dataclasses import Stage1Statement, Stage2Statement
from griffon.preprocessing.graph.transform import DistancesTransformer
from griffon.preprocessing.pipeline.stage2.stage2 import get_distances_transformer, get_vocab_transformer
from griffon.preprocessing.pipeline.stage2.vocab import VocabTransform
from griffon.preprocessing.pipeline.utils import connect_subtokens
from griffon.utils import load_config

def process_statements(statements:List[Stage1Statement],
                 vocab_transformer : VocabTransform,
                 dist_transformer : DistancesTransformer)->List[Stage2Statement]:

    stage_2_statements = []
    # now we transform the stage 1 samples in stage 2 samples
    for statement in statements:

        vocabularized_statement = vocab_transformer.process_statement(statement, no_unk=True)
        stage_2_statement = dist_transformer.process_statement(vocabularized_statement)
        stage_2_statements.append(stage_2_statement)

    return stage_2_statements

def verify_masks(original_statement:Stage1Statement, mask:List[str]):

    assert any((tok == MASK_TOKEN for tok in mask)), "mask list has no <mask> token"

    for original_token, mask_token in zip(original_statement.tokens, mask):
        concat_token = connect_subtokens(original_token)
        assert mask_token == concat_token or mask_token == MASK_TOKEN, \
            f"unexpected token {mask_token} in mask list"

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="""
            run stage 2 preprocessing on semantic tests data
        """
    )
    arg_parser.add_argument(
        "--semantic_tests_root", type=str, default="semantic_tests", help="path to the semantic test root"
    )
    arg_parser.add_argument(
        "--vocab", type=str, default="models/vocab.pickle", help="The path to the pickled torchtext vocab"
    )
    arg_parser.add_argument(
        "--config", type=str, default="configs/config.json", help="The path to the config file"
    )
    args = arg_parser.parse_args()

    masks_path = os.path.join(args.semantic_tests_root, "masks.json")
    assert os.path.exists(masks_path), f"{masks_path} does not exist, did you forget to modify \"new_masks.json\""

    masked_statements_per_file:Dict[str, Dict[str, List[str]]] = \
        json.load(open(masks_path, "r"))

    stage_1_tests_path = os.path.join(args.semantic_tests_root, "stage1")
    assert os.path.exists(stage_1_tests_path), f"{stage_1_tests_path} does not exist"
    stage_1_test_paths = glob(os.path.join(stage_1_tests_path, "*.pkl"))

    output_dir = os.path.join(args.semantic_tests_root, "stage2")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    vocab_transformer = get_vocab_transformer(args.vocab)
    distances_transformer = get_distances_transformer(load_config(args.config))

    for path in stage_1_test_paths:
        filename = path.split(os.sep)[-1].split(".")[0]
        statements = pickle.load(open(path, "rb"))
        masked_statements = masked_statements_per_file[filename]

        # let's first validate all the masks
        for statement in statements:
            verify_masks(statement, masked_statements[statement.name])

        # now we can transform the statements
        stage_2_statements = process_statements(
            statements, vocab_transformer, distances_transformer)

        pickle.dump(stage_2_statements, open(os.path.join(output_dir, f"{filename}.pkl"), "wb"))

