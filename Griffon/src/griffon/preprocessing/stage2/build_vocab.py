from typing import List

from glob import glob
from tqdm import tqdm

import argparse
from argparse import Namespace

import json

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

def build_vocab(stage1_root:str, output_path):
    train_root = os.path.join(stage1_root, "train")
    assert os.path.exists(train_root)

    proof_files = glob(train_root + "/**/*.pickle", recursive=True)

    vocab = build_vocab_from_iterator(subtoken_generator(proof_files), min_freq=3, specials=SPECIAL_TOKENS)

    dir_path = os.path.sep.join(output_path.split(os.path.sep)[:-1])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    pickle.dump(vocab, open(output_path, "wb"))