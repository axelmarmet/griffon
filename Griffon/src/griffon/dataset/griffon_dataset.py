import os
import pickle

from glob import glob
import torch

from typing import Dict, List, Tuple
from torch import Tensor

import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from griffon.constants import EOS_TOKEN, MAX_NUM_TOKEN, NUM_SUB_TOKENS, PAD_TOKEN, SOS_TOKEN, UNK_TOKEN
from griffon.coq_dataclasses import GriffonLemma, GriffonSample, GriffonStatement, Stage1Token, Stage2Sample, Stage2Statement, Stage2Token
from griffon.utils import pad_list, pad_mask

class GriffonDataset(Dataset):

    def __init__(self, root_path:str, split:str):

        assert os.path.exists(root_path), f"Path {root_path} does not exist"
        assert split in ["train", "test", "valid"], f"Split {split} is not supported"

        sample_dir = os.path.join(root_path, split)
        assert os.path.exists(sample_dir), f"Sample directory {sample_dir} does not exist"

        self.files = sorted(glob(os.path.join(sample_dir, "*.pickle")))
        self.vocab = pickle.load(open(os.path.join(root_path, "vocab.pkl"), "rb"))

        self.pad_id = self.vocab[PAD_TOKEN]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index)->GriffonSample:
        sample:GriffonSample = pickle.load(open(self.files[index], "rb"))
        return sample

    def simple_collate_fn(self, samples:List[GriffonSample])->GriffonSample:
        assert len(samples) == 1
        return samples[0]

    def to_dataloader(self, num_workers:int):
        return DataLoader(
            self,
            batch_size=1,
            collate_fn=self.simple_collate_fn, # type: ignore
            pin_memory=True,
            num_workers=num_workers)

    # def collate_fn(self, samples: List[CTCoqSample])->CTCoqBatch:
    #     max_number_statements = max([sample.])


    #     raise NotImplementedError

    #     sample = samples[0]

    #     def get_max_seq_length(samples: List[CTCoqSample])->int:
    #         current_max_seq_length = -1
    #         for sample in samples:
    #             max_hypo_seq_length = max([hypo.tokens.shape[0] for hypo in sample.hypotheses])
    #             sample_max_seq_length = max(max_hypo_seq_length, sample.goal.tokens.shape[0])
    #             current_max_seq_length = max(sample_max_seq_length, current_max_seq_length)
    #         return current_max_seq_length

    #     max_number_hypotheses = max([len(sample.hypotheses) for sample in samples])
    #     max_seq_length = get_max_seq_length(samples)

    #     # We use +1 here as in some occasions a [CLS] token is inserted at the beginning which increases sequence length
    #     assert max_seq_length <= MAX_NUM_TOKEN, \
    #         f"Sample with more tokens than TOKENS_THRESHOLD ({MAX_NUM_TOKEN})"

    #     seq_tensors = []
    #     token_type_tensors = []
    #     node_type_tensors = []
    #     seq_lengths = []
    #     max_distance_masks = []
    #     relative_distances = dict()
    #     binning_vector_tensors = dict()
    #     for sample in samples:
    #         sequence = sample.tokens
    #         token_types = sample.token_types
    #         node_types = sample.node_types
    #         distance_matrices = sample.distance_matrices
    #         binning_vectors = sample.binning_vectors
    #         distance_names = sample.distance_names

    #         sequence_length = sequence.shape[0]
    #         seq_lengths.append(sequence_length)
    #         # indicates how much token, token type and node type sequences have to be padded
    #         pad_length = max_seq_length - sequence_length

    #         # Pad sequences
    #         sequence = F.pad(sequence, [0, 0, 0, pad_length], value=self.sequence_pad_value)
    #         if self.use_token_types:
    #             token_types = F.pad(token_types, [0, pad_length], value=self.token_type_pad_value)
    #         node_types = F.pad(node_types, [0, pad_length], value=self.node_type_pad_value)

    #         # Calculate and pad max distance mask
    #         max_distance_mask = torch.zeros_like(distance_matrices[0])
    #         if self.max_distance_mask:
    #             max_distance_mask = self.max_distance_mask(distance_matrices, binning_vectors, distance_names)
    #         max_distance_mask = F.pad(max_distance_mask, [0, pad_length, 0, pad_length], value=1)
    #         max_distance_masks.append(max_distance_mask)

    #         # Every sample has a matrix for every distance type. We want to have all matrices of the same distance
    #         # type grouped together in one dictionary entry (with the dictionary key being the indices of distance
    #         # types)
    #         for i, (dist_matrix, binning_vector) in enumerate(zip(distance_matrices, binning_vectors)):
    #             padded_dist_matrix = F.pad(dist_matrix,
    #                                        [0, pad_length, 0, pad_length])  # pad distance matrices with 0 bin
    #             if i not in relative_distances:
    #                 relative_distances[i] = []
    #                 binning_vector_tensors[i] = []
    #             relative_distances[i].append(padded_dist_matrix)
    #             binning_vector_tensors[i].append(binning_vector)

    #         # Group together sequences
    #         seq_tensors.append(sequence)
    #         if self.use_token_types:
    #             token_type_tensors.append(token_types)
    #         node_type_tensors.append(node_types)

    #     # Transform distance matrices and binning vectors into tensors
    #     for i in relative_distances.keys():
    #         relative_distances[i] = torch.stack(relative_distances[i])  # yields batch_size x N x N
    #         binning_vector_tensors[i] = torch.stack(binning_vector_tensors[i]).T  # yields K x batch_size

    #     seq_tensors = torch.stack(seq_tensors)
    #     seq_lengths = torch.tensor(seq_lengths)
    #     max_distance_masks = torch.stack(max_distance_masks)
    #     padding_mask = pad_mask(seq_lengths, max_len=max_seq_length)
    #     if self.use_token_types:
    #         token_type_tensors = torch.stack(token_type_tensors)
    #     else:
    #         token_type_tensors = None

    #     extended_vocabulary_ids = None
    #     extended_vocabulary = None
    #     pointer_pad_mask = None
    #     if self.use_pointer_network:
    #         # Simply pad extended_vocabulary IDs
    #         seq_len_subtokens = max([len(sample.extended_vocabulary_ids) for sample in batch])
    #         extended_vocabulary_ids = torch.tensor([
    #             pad_list(sample.extended_vocabulary_ids, seq_len_subtokens, self.sequence_pad_value)
    #             for sample in batch])

    #         pointer_pad_mask = pad_sequence([sample.pointer_pad_mask for sample in batch], batch_first=True,
    #                                         padding_value=False)

    #         extended_vocabulary = [v.extended_vocabulary for v in batch]

    #     languages = None
    #     if batch[0].language is not None:
    #         languages = torch.tensor([sample.language for sample in batch])

    #     return CTBaseBatch(tokens=seq_tensors, token_types=token_type_tensors,
    #                        node_types=torch.stack(
    #                            node_type_tensors), relative_distances=list(zip(relative_distances.values(),
    #                                                                            binning_vector_tensors.values())),
    #                        distance_names=distance_names, sequence_lengths=seq_lengths, pad_mask=padding_mask,
    #                        max_distance_mask=max_distance_masks, extended_vocabulary_ids=extended_vocabulary_ids,
    #                        pointer_pad_mask=pointer_pad_mask, extended_vocabulary=extended_vocabulary,
    #                        languages=languages)
