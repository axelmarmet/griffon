

import io
import os
from typing import Callable, Dict

import torch
from torch.nn.utils.rnn import pad_sequence

from griffon.preprocessing import TextTransform, Vocab


# lifted from torchtext, didn't import it because of
# the underscore prefix

def test():
    print("Hiii")

def _read_text_iterator(path):
    with io.open(path, encoding="utf-8") as f:
        for row in f:
            yield row

class UsefulItemsDataset(torch.utils.data.IterableDataset):
    """
    A dataset returning tuples of goals and used theorem
    """
    def __init__(self, data_root: str):
        super(UsefulItemsDataset).__init__()
        self.goal_path = os.path.join(data_root, "goals.txt")
        self.used_path = os.path.join(data_root, "used.txt")
        assert os.path.exists(self.goal_path)
        assert os.path.exists(self.used_path)
        # remove this once the dataset is settled
        with open(self.goal_path, "r", encoding="utf-8") as file:
            nonempty_lines = [line.strip("\n") for line in file]
            used_length = len(nonempty_lines)
        with open(self.used_path, "r", encoding="utf-8") as file2:
            nonempty_lines = [line.strip("\n") for line in file2]
            goal_length = len(nonempty_lines)
        assert used_length == goal_length
        self.length = used_length

    def __iter__(self):
        goal_iter = _read_text_iterator(self.goal_path)
        used_iter = _read_text_iterator(self.used_path)
        return zip(goal_iter, used_iter)

    def __len__(self):
        return self.length


def get_data_loader(ds: UsefulItemsDataset, text_transform: TextTransform, batch_size: int):
    """Return dataloader returning batches of given size
    Args:
        ds (UsefulItemsDataset): dataset to iterate over
        text_transform (TextTransform): the text transform pipeline
        batch_size (int): the batch size to use
    """

    return torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=text_transform.process_batch)
