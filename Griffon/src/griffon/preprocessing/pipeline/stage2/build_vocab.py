from typing import List

from glob import glob
from tqdm import tqdm

import argparse
from argparse import Namespace

import os
import pickle
from griffon.constants import SPECIAL_TOKENS

from griffon.coq_dataclasses import Stage1Sample

from torchtext.vocab import build_vocab_from_iterator

def subtoken_generator(files:List[str]):
    for file in tqdm(files):

        sample:Stage1Sample = pickle.load(open(file, "rb"))
        # iterate over all hypotheses
        for hypothesis in sample.hypotheses:
            for token in hypothesis.tokens:
                yield token.subtokens

        # iterate over the goal
        for token in sample.goal.tokens:
            yield token.subtokens

        # iterate over the used lemma
        for token in sample.lemma_used:
            yield token.subtokens

def build_vocab(args:Namespace):
    train_root = os.path.join(args.stage1_root, "train")
    assert os.path.exists(train_root)

    proof_files = glob(train_root + "/**/*.pickle", recursive=True)

    return build_vocab_from_iterator(subtoken_generator(proof_files), min_freq=3, specials=SPECIAL_TOKENS)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="""
        Create a vocabulary using the tokens contained in the training split
        of stage1 samples
        """
    )

    arg_parser.add_argument(
        "--stage1_root", type=str, required=True, help="The root of the stage1 processed samples"
    )
    arg_parser.add_argument(
        "--output_folder", type=str, default="models", help="The folder to which the pickled vocab should be added"
    )

    args = arg_parser.parse_args()

    vocab = build_vocab(args)

    output_file = os.path.join(args.output_folder, "vocab.pickle")
    pickle.dump(vocab, open(output_file, "wb"))