
import argparse
import os
import pickle
from argparse import Namespace
from glob import glob

import torch

import shutil

from random import randint, random
import numpy as np
from tqdm import tqdm

from griffon.constants import NUM_SUB_TOKENS, MASK_TOKEN, PAD_TOKEN, TGT_IGNORE_INDEX
from griffon.coq_dataclasses import CounTSample, Stage2Sample, Stage2Statement, Stage2Token
from griffon.utils import pad_list, set_seed

def create_from_stage2(args:Namespace):
    vocab = pickle.load(open(os.path.join(args.stage2_root, "vocab.pkl"), "rb"))
    pad_id = vocab[PAD_TOKEN]

    def transform_statement(statement: Stage2Statement)->CounTSample:

        def create_random_token():
            number_sub_tokens = min(np.random.geometric(0.4), NUM_SUB_TOKENS)

            itos = vocab.get_itos()
            subtokens = [itos[randint(0, len(vocab)-1)] for _ in range(number_sub_tokens)]
            return Stage2Token(vocab(subtokens), subtokens)

        target_ids = []

        selected_something = False
        while not selected_something :
            # reset the lists
            target_ids = []
            # repeat until we have at least some targets
            for i, token in enumerate(statement.tokens):
                prob = random()
                if prob > 0.15:
                    target_ids.append([TGT_IGNORE_INDEX]*NUM_SUB_TOKENS)
                    continue

                selected_something = True
                target_ids.append(token.subtokens)

                # now we check if we want to
                # 1) mask the token (80% chance)
                # 2) replace the token by a random word (10% chance)
                # 3) leave the token intact

                prob /= 0.15
                if prob < 0.8:
                    statement.tokens[i] = Stage2Token([vocab[MASK_TOKEN]] * NUM_SUB_TOKENS,
                                            [MASK_TOKEN]             * NUM_SUB_TOKENS)
                elif prob < 0.9:
                    statement.tokens[i] = create_random_token()

        target_ids = torch.tensor(
            [pad_list(subtoken_ids, NUM_SUB_TOKENS, TGT_IGNORE_INDEX) for subtoken_ids in target_ids]
        )
        input_ids = torch.tensor(
            [pad_list(token.subtokens, NUM_SUB_TOKENS, pad_id) for token in
            statement.tokens]
        )

        distance_indices = torch.stack([distance[0] for distance in statement.distances])
        distance_bins = torch.stack([distance[1] for distance in statement.distances])

        return CounTSample(
            input_ids = input_ids,
            distance_indices = distance_indices,
            distance_bins = distance_bins,
            target_ids = target_ids,
        )

    def create_split(split:str):
        sample_index = 0
        split_dir = os.path.join(args.stage2_root, split)
        assert os.path.exists(split_dir)

        target_dir = os.path.join(args.target_root, split)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        files = sorted(glob(os.path.join(split_dir, "*.pickle")))

        for file in tqdm(files):
            sample:Stage2Sample = pickle.load(open(file, "rb"))
            statements = sample.hypotheses + [sample.goal]

            for statement in statements:
                if len(statement.tokens) < 10:
                    continue
                count_sample = transform_statement(statement)
                target_file = os.path.join(target_dir, 'sample{:08d}.pickle'.format(sample_index))
                sample_index += 1
                pickle.dump(count_sample, open(target_file, "wb"))

    splits = ["train", "test", "valid"]
    for split in splits:
        create_split(split)

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(
        description="""
            create CounT dataset from stage 2
        """
    )
    arg_parser.add_argument(
        "--data_root", type=str, default="data"
    )
    args = arg_parser.parse_args()
    setattr(args, "stage2_root", os.path.join(args.data_root, "base", "stage2"))
    setattr(args, "target_root", os.path.join(args.data_root, "CounT"))

    set_seed(0)

    create_from_stage2(args)
    shutil.copy(os.path.join(args.stage2_root, "vocab.pkl"),
                os.path.join(args.target_root, "vocab.pkl"))
