
import argparse
import itertools
import os
import pickle
from argparse import Namespace
from glob import glob
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor

import multiprocessing as mp

import shutil

from random import randint, random
import numpy as np
from tqdm import tqdm

from griffon.constants import EOS_TOKEN, NUM_SUB_TOKENS, MASK_TOKEN, PAD_TOKEN, SOS_TOKEN, TGT_IGNORE_INDEX, UNK_TOKEN
from griffon.coq_dataclasses import GriffonLemma, GriffonSample, GriffonStatement, CounTSample, Stage1Token, Stage2Sample, Stage2Statement, Stage2Token
from griffon.utils import pad_list, pad_mask, set_seed

def init_process(args:Namespace):
    global vocab
    vocab = pickle.load(open(os.path.join(args.stage2_root, "vocab.pkl"), "rb"))

def process_sample(args:Namespace, sample_path:str):
    pad_id = vocab[PAD_TOKEN]
    unk_id = vocab[UNK_TOKEN]

    def transform_statement(statement: Stage2Statement,
                            extended_vocab:Dict[str,int])->GriffonStatement:

        sequence = torch.tensor(
            [pad_list(token.subtokens, NUM_SUB_TOKENS, pad_id) for token in
             statement.tokens])

        extended_vocabulary_ids:List[List[int]] = []
        # Generating extended vocabulary for pointer network. The extended vocabulary essentially mimics an infinite
        # vocabulary size for this sample
        len_vocab = len(vocab)
        for token in statement.tokens:

            token_vocabulary_ids:List[int] = []

            padded_subtokens = pad_list(token.subtokens, NUM_SUB_TOKENS, pad_id)
            padded_original_subtokens = pad_list(token.original_subtokens, NUM_SUB_TOKENS, PAD_TOKEN)

            for subtoken, original_subtoken in zip(padded_subtokens, padded_original_subtokens):
                if subtoken == unk_id:
                    if original_subtoken in extended_vocab:
                        extended_id = extended_vocab[original_subtoken]
                    else:
                        extended_id = len_vocab + len(extended_vocab)
                        extended_vocab[original_subtoken] = extended_id
                    token_vocabulary_ids.append(extended_id)
                else:
                    token_vocabulary_ids.append(subtoken)

            extended_vocabulary_ids.append(token_vocabulary_ids)

        return GriffonStatement(tokens=sequence,
                              extended_vocabulary_ids=extended_vocabulary_ids,
                              distances=statement.distances)

    def transform_lemma(lemma : List[Stage1Token], extended_vocab:Dict[str,int])->GriffonLemma:

        def get_filled_token(token:str)->Stage2Token:
            return Stage2Token([vocab[token]] * NUM_SUB_TOKENS, [token] * NUM_SUB_TOKENS)

        def transform_lemma_token(token : Stage1Token)->Stage2Token:
            new_subtokens = []
            padded_subtokens = pad_list(token.subtokens, NUM_SUB_TOKENS, PAD_TOKEN)
            for subtoken in padded_subtokens:
                if subtoken in vocab:
                    new_subtokens.append(vocab[subtoken])
                else:
                    if subtoken in extended_vocab:
                        new_subtokens.append(extended_vocab[subtoken])
                    else:
                        new_subtokens.append(unk_id)
            return Stage2Token(new_subtokens, padded_subtokens)

        tokens = [get_filled_token(SOS_TOKEN)] + \
                 [transform_lemma_token(token) for token in lemma] + \
                 [get_filled_token(EOS_TOKEN)]

        seq = torch.tensor(
            [pad_list(
                token.subtokens,
                NUM_SUB_TOKENS,
                pad_id) for token in tokens])

        return GriffonLemma(seq)

    def transform_sample(sample: Stage2Sample)->GriffonSample:
        extended_vocab = {}

        selected_hypotheses = sample.hypotheses
        if len(selected_hypotheses) > 64:
            selected_hypotheses = selected_hypotheses[:64]

        transformed_hypotheses = [transform_statement(
                                    hypothesis,
                                    extended_vocab)
                                  for hypothesis in selected_hypotheses]

        transformed_goal = transform_statement(sample.goal, extended_vocab)
        transformed_lemma = transform_lemma(sample.lemma_used, extended_vocab)

        all_statements = transformed_hypotheses + [transformed_goal]

        max_seq_length = max([statement.tokens.shape[0] for statement in all_statements])

        seq_lengths       :List[int] = []

        seq_tensors       :List[Tensor] = []
        distance_matrices :List[Tensor] = []
        binning_vectors   :List[Tensor] = []

        for statement in all_statements:

            sequence = statement.tokens

            sequence_length = sequence.shape[0]

            seq_lengths.append(sequence_length)
            pad_length = max_seq_length - sequence_length

            sequence = F.pad(sequence, [0, 0, 0, pad_length], value=pad_id)
            seq_tensors.append(sequence)

            # pad the distance matrices
            dist_matrices:List[Tensor] = []
            bin_vectors:List[Tensor] = []
            for dist_matrix, binning_vector, _ in statement.distances:
                padded_dist_matrix = F.pad(dist_matrix,
                                           [0, pad_length, 0, pad_length])
                dist_matrices.append(padded_dist_matrix)
                bin_vectors.append(binning_vector)

            distance_matrices.append(torch.stack(dist_matrices, dim = 0))
            binning_vectors.append(torch.stack(bin_vectors, dim = 0))
            del dist_matrices, bin_vectors

        seq_tensor = torch.stack(seq_tensors, dim=0)

        extended_vocabulary_ids = torch.tensor(
            [
                pad_list(statement.extended_vocabulary_ids, max_seq_length, [pad_id]*NUM_SUB_TOKENS)
                for statement
                in all_statements
            ]
        )

        padding_mask = torch.logical_not(pad_mask( torch.tensor(seq_lengths), max_len=max_seq_length).bool())
        distance_matrix = torch.stack(distance_matrices, dim=0)
        binning_matrix = torch.stack(binning_vectors, dim=0)

        ret_sample = GriffonSample(
            statements=seq_tensor,
            extended_vocabulary_ids=extended_vocabulary_ids,
            distances_indices = distance_matrix,
            distances_bins = binning_matrix,
            token_padding_mask=padding_mask,
            lemma = transformed_lemma.tokens,
            extended_vocabulary = extended_vocab
        )

        ret_sample.validate()

        return ret_sample

    split, filename = sample_path.split(os.sep)[-2:]
    sample:Stage2Sample = pickle.load(open(sample_path, 'rb'))

    transformed_sample = transform_sample(sample)
    output_file = os.path.join(args.target_root, split, filename)

    pickle.dump(transformed_sample, open(output_file, "wb"))

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(
        description="""
            create griffon dataset from stage 2
        """
    )
    arg_parser.add_argument(
        "--data_root", type=str, default="data"
    )
    arg_parser.add_argument(
        "--threads", type=int, default=None, help="""
        The number of threads used for preprocessing (default: use all threads available)
        """
    )

    args = arg_parser.parse_args()
    setattr(args, "stage2_root", os.path.join(args.data_root, "base", "stage2"))
    setattr(args, "target_root", os.path.join(args.data_root, "griffon"))

    set_seed(0)

    for split in ["test", "train", "valid"]:
        split_dir = os.path.join(args.target_root, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

    proof_files = glob(f"{args.stage2_root}/**/*.pickle")
    with mp.Pool(args.threads, initializer=init_process, initargs=[args]) as pool:
        pool.starmap(process_sample, zip(itertools.repeat(args), proof_files))

    shutil.copy(os.path.join(args.stage2_root, "vocab.pkl"),
                os.path.join(args.target_root, "vocab.pkl"))