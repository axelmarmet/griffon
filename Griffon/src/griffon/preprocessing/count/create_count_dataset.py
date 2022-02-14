
import argparse
from collections import defaultdict
import os
import pickle
from argparse import Namespace
from glob import glob

import torch

import shutil

from tqdm import tqdm

from griffon.constants import NUM_SUB_TOKENS, PAD_TOKEN
from griffon.coq_dataclasses import PreCounTSample, Stage2Sample, Stage2Statement
from griffon.utils import pad_list, set_seed

def create_from_stage2(args:Namespace):
    vocab = pickle.load(open(os.path.join(args.stage2_root, "vocab.pkl"), "rb"))
    pad_id = vocab[PAD_TOKEN]

    def transform_statement(statement: Stage2Statement)->PreCounTSample:

        input_ids = torch.tensor(
            [pad_list(token.subtokens, NUM_SUB_TOKENS, pad_id) for token in
            statement.tokens]
        )

        distance_indices = torch.stack([distance[0] for distance in statement.distances])
        distance_bins = torch.stack([distance[1] for distance in statement.distances])

        return PreCounTSample(
            input_ids = input_ids,
            distance_indices = distance_indices,
            distance_bins = distance_bins
        )

    def create_split(split:str):
        indices = defaultdict(int)

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
                num_tokens = len(statement.tokens)
                if num_tokens < 7:
                    continue
                count_sample = transform_statement(statement)
                target_file = os.path.join(target_dir, 'sample{:03d}_{:08d}.pickle'.format(num_tokens, indices[num_tokens]))
                indices[num_tokens] += 1

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
