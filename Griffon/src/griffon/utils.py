import os

import torch.distributed as dist

import json
import random
import numpy as np
import torch

from typing import Any, Callable, Dict, List, Optional, TypeVar
from torch import Tensor

from CoqGym.utils import update_env

# necessary functions for distributed training

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


T = TypeVar('T')
def find_in_list(l:List[T], f:Callable[[T], bool])->Optional[T]:
    return next((x for x in l if f(x)), None)

def get_path_relative_to_data(data_root, filename:str):
    filename = filename.replace(data_root, "", 1)
    assert filename[0] == os.path.sep
    return filename[1:]

def load_config(path:str):
    return json.load(open(path, "r"))

def pad_list(list_to_pad:List, pad_size, padding):
    assert len(list_to_pad) <= pad_size, f"Cannot pad list with length {len(list_to_pad)} with pad size {pad_size}"

    padded_list = list_to_pad.copy()
    num_padding = pad_size - len(list_to_pad)
    padded_list.extend([padding for _ in range(num_padding)])
    return padded_list


def pad_mask(sequence_lengths:Tensor, max_len:Optional[int] = None):
    if max_len is None:
        max_len = int(sequence_lengths.max().item())
    batch_size = sequence_lengths.shape[0]
    pad_mask = torch.arange(0, max_len).expand((batch_size, -1))
    pad_mask = 1 - (pad_mask >= sequence_lengths[:, None]).long()
    assert (pad_mask.sum(-1) == sequence_lengths).all()
    return pad_mask


def iter_proofs_in_file(filename, callback:Callable):
    file_data = json.load(open(filename))
    env = {"constants": [], "inductives": []}
    for proof_data in file_data["proofs"]:
        env = update_env(env, proof_data["env_delta"])
        del proof_data["env_delta"]
        proof_data["env"] = env
        callback(filename, proof_data)

def print_metrics(metrics):
    print(json.dumps(metrics, sort_keys=True, indent=4))

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

