from abc import ABC
from typing import Dict, List, Tuple

from glob import glob
import os
from nltk.tokenize import RegexpTokenizer
import gensim
import numpy as np

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class Tokenizer:

    def __init__(self) -> None:
        regex = r"[^\W_]+"
        self.tokenizer = RegexpTokenizer(regex)

    def __call__(self, sentence:str) -> List[str]:
        return self.tokenizer.tokenize(sentence)

class Vocab:

    def __init__(self, word_vectors:gensim.models.keyedvectors.KeyedVectors):
        self.word_vectors = word_vectors
        self.word_vectors["<unk>"] = np.random.normal(size=word_vectors.vector_size)
        self.word_vectors["<bos>"] = np.random.normal(size=word_vectors.vector_size)
        self.word_vectors["<eos>"] = np.random.normal(size=word_vectors.vector_size)
        self.word_vectors["<pad>"] = np.random.normal(size=word_vectors.vector_size)
        self.UNK_IDX = self.word_vectors.key_to_index["<unk>"]
        self.BOS_IDX = self.word_vectors.key_to_index["<bos>"]
        self.EOS_IDX = self.word_vectors.key_to_index["<eos>"]
        self.PAD_IDX = self.word_vectors.key_to_index["<pad>"]

    def get_embedding_dim(self)->int:
        return self.word_vectors.vector_size

    def __len__(self)->int:
        return len(self.word_vectors)

    def str_to_tensor(self, inp:str)->np.ndarray:
        return self.word_vectors[inp]

    def add_markers(self, tokens:List[str]):
        return ["<bos>"] + tokens + ["</eos>"]

    def tokens_to_tensor(self, inp:List[str])->np.ndarray:
        inp = map(lambda w : w if w in self.word_vectors else "<unk>", inp)
        tensors = [self.word_vectors.key_to_index[word] for word in inp]
        return np.stack(tensors)


def _list_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

class TextTransform(ABC):

    def process_batch(self):
        pass

class StandardTextTransform(TextTransform):
    def __init__(self, vocab:Vocab, tokenizer:Tokenizer, *lambdas):

        self.vocab = vocab
        self.tokenizer = tokenizer
        self.pipeline = _list_transforms(tokenizer,
                                        vocab.add_markers,
                                        vocab.tokens_to_tensor,
                                        *lambdas)

    def process_sentence(self, inp:str):
        return self.pipeline(inp)

    def process_batch(self, batch):
        goal_batch, used_batch = [], []
        for goal, used in batch:
            goal_batch.append(self.process_sentence(goal.rstrip("\n")))
            used_batch.append(self.process_sentence(used.rstrip("\n")))

        goal_batch = pad_sequence(
            goal_batch, padding_value=self.vocab.PAD_IDX)
        used_batch = pad_sequence(
            used_batch, padding_value=self.vocab.PAD_IDX)
        return goal_batch, used_batch



class CopyingTextTransform(TextTransform):
    def __init__(self, vocab:Vocab, tokenizer:Tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.align_ignore = -1

    def _create_oov_vocab(self, tokenized_src:List[List[str]])->Dict[str,int]:
        counter = 0
        oov_vocab = {}
        for src_sentence in tokenized_src:
            for token in src_sentence:
                if token not in self.vocab.word_vectors and \
                   token not in oov_vocab:
                    oov_vocab[token] = counter
                    counter += 1
        return oov_vocab

    def process_input(self, tokens:List[str], oov_vocab:Dict[str,int], max_seq_len:int)->Tuple[Tensor,Tensor]:

        def tokens_to_sparse_oov_tensor()->Tensor:
            """Convert the list of tokens to a sparse indicator matrix of oov indices
            Returns:
                Tensor: of shape ``(max_seq_len, extra_words)`` sparse indicator matrix
            """
            out_shape = [max_seq_len, len(oov_vocab)]
            indices = [[sentence_index, oov_vocab[word]]
                        for sentence_index, word in enumerate(tokens)
                        if word in oov_vocab]
            if indices:
                # transpose the index matrix
                indices = torch.tensor(indices).t()
                values = torch.ones(indices.shape[1])
                return torch.sparse_coo_tensor(indices, values, size=out_shape)
            else:
                return torch.sparse_coo_tensor(size=out_shape)

        in_vocab_tensor = torch.as_tensor(self.vocab.tokens_to_tensor(tokens))
        oov_indicator_tensor = tokens_to_sparse_oov_tensor()
        return in_vocab_tensor, oov_indicator_tensor

    def process_target(self, tokens:List[str], oov_vocab:Dict[str,int])->Tuple[Tensor, Tensor]:

        def tokens_to_align_tensor()->Tensor:
            """Return a tensor of shape ``(tgt_len)`` that represent the indices of
            the tgt tokens in the oov vocab, if index = -1, then the word is not oov
            and should not be copied

            Returns:
                Tensor: [description]
            """
            return torch.tensor([oov_vocab[token] if token in oov_vocab else self.align_ignore for token in tokens])


        in_vocab_tensor = torch.as_tensor(self.vocab.tokens_to_tensor(tokens))
        align_tensor = tokens_to_align_tensor()
        return in_vocab_tensor, align_tensor

    def process_batch(self, batch)->Tuple[Tensor, Tensor, Tensor, Tensor]:
        """process batch for training with a copier network

        Args:
            batch : contains tuple of strings, one for the source one for the target

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
            src_batch : shape ``(max_src_len, batch_dim)``
            src_map_batch : shape ``(max_src_len, batch_dim, oov_vocab_size)` spare`
            tgt_batch : shape ``(max_tgt_len, batch_dim)
            align_batch : shape ``(batch_dim, max_tgt_len)`` indices of the oov words in
                the target, if it is in the vocab the elem is -1
        """
        # this pass is necessary because we need the max_src_len to create
        # the sparse indicator matric for oov words
        tokenized_sources, tokenized_targets = [], []
        max_src_len, max_tgt_len = 0, 0
        for src, tgt in batch:
            # process src sentence
            tokenized_src = self.tokenizer(src)
            tokenized_src = self.vocab.add_markers(tokenized_src)
            max_src_len = max(max_src_len, len(tokenized_src))
            tokenized_sources.append(tokenized_src)

            #process tgt sentence
            tokenized_tgt = self.tokenizer(tgt)
            tokenized_tgt = self.vocab.add_markers(tokenized_tgt)
            max_tgt_len = max(max_tgt_len, len(tokenized_tgt))
            tokenized_targets.append(tokenized_tgt)

        oov_vocab = self._create_oov_vocab(tokenized_sources)
        src_batch, tgt_batch, src_map_batch, align_batch = [], [], [], []

        for src_tokens, tgt_tokens in zip(tokenized_sources, tokenized_targets):
            processed_src, src_map = self.process_input(src_tokens, oov_vocab, max_src_len)
            processed_tgt, align = self.process_target(tgt_tokens, oov_vocab)

            src_batch.append(processed_src)
            tgt_batch.append(processed_tgt)
            src_map_batch.append(src_map)
            align_batch.append(align)

        src_batch = pad_sequence(src_batch, padding_value=self.vocab.PAD_IDX)
        src_map_batch = torch.stack(src_map_batch, dim=1).to_dense()
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.vocab.PAD_IDX)
        align_batch = pad_sequence(align_batch, padding_value=self.align_ignore)

        return src_batch, src_map_batch, tgt_batch, align_batch


class CorpusLoader:
    def __init__(self, corpus_path, pipeline):
        self.files = glob(os.path.join(corpus_path, "**/*.txt"), recursive=True)
        assert self.files
        self.pipeline = pipeline

    def __iter__(self):
        for file in self.files:
            with open(file, "r") as file:
                for line in file:
                    yield self.pipeline(line)