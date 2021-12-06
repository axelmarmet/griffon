# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from torch.utils.data import DataLoader

from griffon.models.encoder.count import CounT
from griffon.dataset.count_dataset import CounTDataset

import pickle
import json


# %%
ds = CounTDataset("../data/processed/stage2/train", "../models/vocab.pickle")


# %%
config = json.load(open("../configs/architecture.json"))
model = CounT(ds.vocab, config)


# %%
dataloader = DataLoader(ds, batch_size = 32, collate_fn=ds.collate_fn) #type:ignore


# %%
for batch in dataloader:
    res = model(batch)
    print(res)
    break


