import os
import pickle

from glob import glob
import torch

from typing import Dict, List, Tuple
from torch import Tensor

import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from griffon.constants import MAX_NUM_TOKEN, NUM_SUB_TOKENS, PAD_TOKEN, UNK_TOKEN
from griffon.coq_dataclasses import CTCoqLemma, CTCoqSample, CTCoqStatement, Stage1Token, Stage2Sample, Stage2Statement, Stage2Token
from griffon.utils import pad_list, pad_mask


class CTCoqDataset(Dataset):
    """
    Unites common functionalities used across different datasets such as applying the token mapping to the
    distance matrices and collating the matrices from multiple samples into one big tensor.
    """

    def __init__(self, root_path:str, vocab_path:str):

        assert os.path.exists(root_path), f"Path {root_path} does not exist"
        assert os.path.exists(vocab_path), f"Path {vocab_path} does not exist"

        self.files = glob(os.path.join(root_path, "*.pickle"))
        self.vocab = pickle.load(open(vocab_path, "rb"))

        self.unk_id = self.vocab[UNK_TOKEN]
        self.pad_id = self.vocab[PAD_TOKEN]

        self.num_sub_tokens = NUM_SUB_TOKENS

    def to_dataloader(self):
        # TODO read more into typing sometimes...
        return DataLoader(self, collate_fn=self.collate_fn) # type: ignore

    def __getitem__(self, index):
        sample:Stage2Sample = pickle.load(open(self.files[index], "rb"))
        return self.transform_sample(sample)


    def transform_statement(self, statement: Stage2Statement,
                            extended_vocab:Dict[str,int])->CTCoqStatement:

        sequence = torch.tensor(
            [pad_list(token.subtokens, self.num_sub_tokens, self.pad_id) for token in
             statement.tokens])

        extended_vocabulary_ids = []
        # Generating extended vocabulary for pointer network. The extended vocabulary essentially mimics an infinite
        # vocabulary size for this sample
        len_vocab = len(self.vocab)
        for idx_token, token in enumerate(statement.tokens):
            for idx_subtoken, subtoken in enumerate(token.subtokens):
                if subtoken == self.unk_id:
                    original_subtoken = token.original_subtokens[idx_subtoken]

                    if original_subtoken in extended_vocab:
                        extended_id = extended_vocab[original_subtoken]
                    else:
                        extended_id = len_vocab + len(extended_vocab)
                        extended_vocab[original_subtoken] = extended_id
                    extended_vocabulary_ids.append(extended_id)
                else:
                    extended_vocabulary_ids.append(subtoken)

        pointer_pad_mask = sequence != self.pad_id

        return CTCoqStatement(tokens=sequence,
                              extended_vocabulary_ids=extended_vocabulary_ids,
                              pointer_pad_mask=pointer_pad_mask,
                              distances=statement.distances)

    def transform_lemma(self, lemma : List[Stage1Token], extended_vocab:Dict[str,int])->CTCoqLemma:

        def transform_lemma_token(token : Stage1Token)->Stage2Token:
            new_subtokens = []
            for subtoken in token.subtokens:
                if subtoken in self.vocab:
                    new_subtokens.append(self.vocab[subtoken])
                else:
                    if subtoken in extended_vocab:
                        new_subtokens.append(extended_vocab[subtoken])
                    else:
                        new_subtokens.append(self.unk_id)
            return Stage2Token(new_subtokens, token.subtokens)

        seq = torch.tensor(
            [pad_list(
                transform_lemma_token(token).subtokens,
                self.num_sub_tokens,
                self.pad_id) for token in lemma])

        return CTCoqLemma(seq)

    def transform_sample(self, sample: Stage2Sample):
        """
        Transforms a sample into torch tensors, applies sub token padding (which is independent of the other samples in
        a batch) and applies the token mapping onto the distance matrices (which makes them much bigger)
        """
        extended_vocab = {}

        transformed_hypotheses = [self.transform_statement(
                                    hypothesis,
                                    extended_vocab)
                                  for hypothesis in sample.hypotheses]

        transformed_goal = self.transform_statement(sample.goal, extended_vocab)
        transformed_lemma = self.transform_lemma(sample.lemma_used, extended_vocab)

        all_statements = transformed_hypotheses + [transformed_goal]

        max_seq_length = max([statement.tokens.shape[0] for statement in all_statements])

        seq_lengths       :List[int] = []

        seq_tensors       :List[Tensor] = []
        pointer_pad_masks :List[Tensor] = []
        distance_matrices :List[Tensor] = []
        binning_vectors   :List[Tensor] = []

        for statement in all_statements:

            sequence = statement.tokens

            sequence_length = sequence.shape[0]

            seq_lengths.append(sequence_length)
            pad_length = max_seq_length - sequence_length

            sequence = F.pad(sequence, [0, 0, 0, pad_length], value=self.pad_id)
            seq_tensors.append(sequence)

            pointer_pad_masks.append(
                F.pad(statement.pointer_pad_mask, [0, 0, 0, pad_length], value=0) # 0 is false in this context
            )

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

        seq_len_subtokens = max([len(statement.extended_vocabulary_ids) for statement in all_statements])
        extended_vocabulary_ids = torch.tensor(
            [
                pad_list(statement.extended_vocabulary_ids, seq_len_subtokens, self.pad_id)
                for statement
                in all_statements
            ]
        )
        padding_mask = pad_mask( torch.tensor(seq_lengths), max_len=max_seq_length)
        distance_matrix = torch.stack(distance_matrices, dim=0)
        binning_matrix = torch.stack(binning_vectors, dim=0)
        pointer_pad_mask = torch.stack(pointer_pad_masks, dim=0)

        ret_sample = CTCoqSample(
            sequences=seq_tensor,
            extended_vocabulary_ids=extended_vocabulary_ids,
            pointer_pad_mask=pointer_pad_mask,
            distances_index = distance_matrix,
            distances_bins = binning_matrix,
            padding_mask=padding_mask,
            lemma = transformed_lemma.tokens,
            extended_vocabulary = extended_vocab
        )

        ret_sample.validate()

        return ret_sample


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
