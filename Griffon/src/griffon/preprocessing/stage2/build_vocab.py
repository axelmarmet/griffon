from typing import List

from glob import glob
from tqdm import tqdm

import argparse
from argparse import Namespace

import json

import os
import pickle
from griffon.constants import SPECIAL_TOKENS

from griffon.coq_dataclasses import Stage1Sample, Stage1Token

from torchtext.vocab import build_vocab_from_iterator

from griffon.preprocessing.stage2.vocab import TorchtextVocab

class Stage1Iterable():

    def __init__(self, stage1_root:str):
        self.files = glob(os.path.join(stage1_root, "**", "*.pickle"), recursive=True)
        assert len(self.files) != 0

    def generator(self):

        def flatten(tokens : List[Stage1Token])->List[str]:
            return [subtoken for token in tokens for subtoken in token.subtokens]

        # # delete meee
        # new_masks = json.load(open("masks_for_vocab.json", "r"))
        # for _ in range(3):
        #     for statement_file in new_masks.values():
        #         for statement in statement_file.values():
        #             new_statement:List[str] = []
        #             for token in statement:
        #                 new_statement += token.split("_")
        #             yield new_statement

        for file in self.files:
            sample:Stage1Sample = pickle.load(open(file, "rb"))
            for hypothesis in sample.hypotheses:
                yield flatten(hypothesis.tokens)
            yield flatten(sample.goal.tokens)
            yield flatten(sample.lemma_used)

    def __iter__(self):
        return self.generator()

def build_vocab(stage1_root:str, output_path):
    train_root = os.path.join(stage1_root, "train")
    assert os.path.exists(train_root)

    vocab = TorchtextVocab(build_vocab_from_iterator(iter(Stage1Iterable(train_root)), min_freq=3, specials=SPECIAL_TOKENS))

    dir_path = os.path.sep.join(output_path.split(os.path.sep)[:-1])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    pickle.dump(vocab, open(output_path, "wb"))