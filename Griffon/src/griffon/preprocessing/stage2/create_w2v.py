import argparse
import os
import pickle
from griffon.preprocessing.stage2.build_vocab import Stage1Iterable

import torch
from torch import Tensor

from griffon.preprocessing.stage2.vocab import AbstractVocab
from gensim.models import Word2Vec

def create_w2v_tensor(vocab:AbstractVocab, stage1_iter:Stage1Iterable, embedding_dim:int)->Tensor:
    model = Word2Vec(stage1_iter,
                 vector_size = embedding_dim,
                 window = 3,
                 min_count = 3,
                 seed = 0,
                 workers = 1)
    word_vectors = model.wv

    num_embeddings = len(vocab)
    embeddings = torch.empty((num_embeddings, embedding_dim))
    for index, word in enumerate(vocab.get_itos()):
        if word not in word_vectors:
            embeddings[index] = torch.randn(embedding_dim)
        else:
            embeddings[index] = torch.Tensor(word_vectors[word])
    return embeddings


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="""
            create word2vec embeddings
        """
    )
    arg_parser.add_argument(
        "--base_root", type=str, required=True, help="the path to the root of the base data"
    )
    args = arg_parser.parse_args()
    setattr(args, "stage1_root", os.path.join(args.base_root, "stage1"))
    setattr(args, "stage2_root", os.path.join(args.base_root, "stage2"))
    setattr(args, "vocab_path", os.path.join(args.stage2_root, "vocab.pkl"))
    setattr(args, "embeddings_path", os.path.join(args.stage2_root, "embeddings.pkl"))

    train_root = os.path.join(args.stage1_root, "train")

    vocab:AbstractVocab = pickle.load(open(args.vocab_path, "rb"))
    stage1_iter:Stage1Iterable = Stage1Iterable(train_root)

    embedding_tensor = create_w2v_tensor(vocab, stage1_iter, 512)
    pickle.dump(embedding_tensor, open(args.embeddings_path, "wb"))