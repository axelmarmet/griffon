# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
from IPython import get_ipython

from typing import List, Dict

import pickle

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from griffon.preprocessing import CopyingTextTransform, Tokenizer
from griffon.dataloader import UsefulItemsDataset, get_data_loader
from griffon.models.copy_generator import CopyGenerator

vocab = pickle.load(open("../models/vocab.pickle", "rb"))
tokenizer = Tokenizer()

ctt = CopyingTextTransform(vocab, tokenizer)

ds = UsefulItemsDataset("../data/processed/short-recommandations/valid")

data_loader = get_data_loader(ds, ctt, 2)

embedding_dim = 10
hidden_dim = 20

class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), embedding_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.copy_generator = CopyGenerator(embedding_dim, hidden_dim, vocab.PAD_IDX)

    def forward(self, inp, src_map):
        embeddings = self.embedding(inp)

        hidden, _ = self.rnn(embeddings)
        attentions = self.attention(hidden)

        hidden = hidden.reshape(-1, hidden_dim)

        return self.copy_generator(hidden, attentions, src_map)

module = TestModule()


for src_batch, src_map_batch, tgt_batch, align_batch in data_loader:

    res = module(src_batch, src_map_batch)


