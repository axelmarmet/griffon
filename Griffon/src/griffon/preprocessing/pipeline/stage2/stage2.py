import argparse
from argparse import Namespace

import os
import pickle

from griffon.utils import load_config
from griffon.coq_dataclasses import Stage1Sample

from griffon.preprocessing.pipeline.stage2.vocab import VocabTransform
from griffon.preprocessing.graph.distances import PersonalizedPageRank, ShortestPaths, AncestorShortestPaths, SiblingShortestPaths, DistanceBinning
from griffon.preprocessing.graph.binning import ExponentialBinning
from griffon.preprocessing.graph.transform import DistancesTransformer

import multiprocessing as mp

import itertools

from glob import glob
import shutil

from tqdm import tqdm


from typing import Tuple


args = Namespace(vocab_path="../../../../../models/vocab.pickle",
                 config_path="../../../../../configs/config.json",
                 stage1_root="../../../../../data/processed/stage1")


def get_vocab_transformer(args: Namespace) -> VocabTransform:
    vocab = pickle.load(open(args.vocab_path, "rb"))
    return VocabTransform(vocab)


def get_distances_transformer(config):
    """
    Extract how distances should be computed from the dataset config
    """
    distances_config = config['distances']
    PPR_ALPHA = distances_config['ppr_alpha']
    PPR_USE_LOG = distances_config['ppr_use_log']
    PPR_THRESHOLD = distances_config['ppr_threshold']

    SP_THRESHOLD = distances_config['sp_threshold']

    ANCESTOR_SP_FORWARD = distances_config['ancestor_sp_forward']
    ANCESTOR_SP_BACKWARD = distances_config['ancestor_sp_backward']
    ANCESTOR_SP_NEGATIVE_REVERSE_DISTS = distances_config['ancestor_sp_negative_reverse_dists']
    ANCESTOR_SP_THRESHOLD = distances_config['ancestor_sp_threshold']

    SIBLING_SP_FORWARD = distances_config['sibling_sp_forward']
    SIBLING_SP_BACKWARD = distances_config['sibling_sp_backward']
    SIBLING_SP_NEGATIVE_REVERSE_DISTS = distances_config['sibling_sp_negative_reverse_dists']
    SIBLING_SP_THRESHOLD = distances_config['sibling_sp_threshold']

    # Extract how distances should be binned from the dataset config
    binning_config = config['binning']
    EXPONENTIAL_BINNING_GROWTH_FACTOR = binning_config['exponential_binning_growth_factor']
    N_FIXED_BINS = binning_config['n_fixed_bins']
    NUM_BINS = binning_config['num_bins']

    distance_metrics = [
        PersonalizedPageRank(threshold=PPR_THRESHOLD,
                             log=PPR_USE_LOG, alpha=PPR_ALPHA),
        ShortestPaths(threshold=SP_THRESHOLD),
        AncestorShortestPaths(forward=ANCESTOR_SP_FORWARD, backward=ANCESTOR_SP_BACKWARD,
                              negative_reverse_dists=ANCESTOR_SP_NEGATIVE_REVERSE_DISTS,
                              threshold=ANCESTOR_SP_THRESHOLD),
        SiblingShortestPaths(forward=SIBLING_SP_FORWARD, backward=SIBLING_SP_BACKWARD,
                             negative_reverse_dists=SIBLING_SP_NEGATIVE_REVERSE_DISTS,
                             threshold=SIBLING_SP_THRESHOLD)]

    db = DistanceBinning(NUM_BINS, N_FIXED_BINS, ExponentialBinning(
        EXPONENTIAL_BINNING_GROWTH_FACTOR))

    return DistancesTransformer(distance_metrics, db)


def process_project(packed_args:Tuple[Namespace,str]):

    args, project_root = packed_args

    root_out_dir = os.path.join(args.stage2_root, "tmp")

    config = load_config(args.config_path)

    vocab_transformer = get_vocab_transformer(args)
    distances_transformer = get_distances_transformer(config)

    samples_pattern = os.path.join(project_root, "**", "*.pickle")
    samples = glob(samples_pattern, recursive=True)

    def process_sample(filename):

        split, project, proof, sample_file = filename.split(os.path.sep)[-4:]
        out_dir = os.path.join(root_out_dir, split, project, proof)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        sample:Stage1Sample = pickle.load(open(filename, "rb"))

        sample = vocab_transformer(sample)
        stage_2_sample = distances_transformer(sample)

        path = os.path.join(out_dir, sample_file)
        pickle.dump(stage_2_sample, open(path, "wb"))

    for i, sample in enumerate(samples):
        process_sample(sample)

def order_files(root:str):
    splits = ["train", "test", "valid"]
    for split in splits:
        proof_index = 0
        output_dir = os.path.join(root, split)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        tmp_root = os.path.join(root, "tmp", split)
        assert os.path.exists(tmp_root)

        files = sorted(glob(os.path.join(tmp_root, "**", "*.pickle"), recursive=True))
        for file in files:
            new_filepath = os.path.join(output_dir, 'proof{:08d}.pickle'.format(proof_index))
            proof_index += 1
            os.rename(file, new_filepath)

    shutil.rmtree(os.path.join(root, "tmp"))

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="""
            run stage 2 preprocessing on stage 1 dataset
        """
    )
    arg_parser.add_argument(
        "--stage1_root", type=str, default="data/processed/stage1", help="The root folder of the stage 1 preprocessed data"
    )
    arg_parser.add_argument(
        "--stage2_root", type=str, default="data/processed/stage2", help="The desired root folder of the stage 2 preprocessed data"
    )
    arg_parser.add_argument(
        "--vocab_path", type=str, default="models/vocab.pickle", help="The path to the pickled torchtext vocab"
    )
    arg_parser.add_argument(
        "--config_path", type=str, default="configs/config.json", help="The path to the config file"
    )
    arg_parser.add_argument(
        "--threads", type=int, default=None, help="""
        The number of threads used for preprocessing (default: use all threads available)
        """
    )

    args = arg_parser.parse_args()
    project_pattern = os.path.join(args.stage1_root, "**", "*")
    projects = glob(project_pattern)

    with mp.Pool(args.threads) as pool:
        for _ in tqdm(pool.imap(process_project, zip(itertools.repeat(args), projects)), total=len(projects)):
            pass

    order_files(args.stage2_root)
