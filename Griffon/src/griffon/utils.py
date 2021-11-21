import os

import json

from typing import Callable, List, Optional, TypeVar

from CoqGym.utils import update_env

T = TypeVar('T')
def find_in_list(l:List[T], f:Callable[[T], bool])->Optional[T]:
    return next((x for x in l if f(x)), None)

def get_path_relative_to_data(data_root, filename:str):
    filename = filename.replace(data_root, "", 1)
    assert filename[0] == os.path.sep
    return filename[1:]

def load_config(path:str):
    return json.load(open(path, "r"))

def pad_list(list_to_pad, pad_size, padding):
    assert len(list_to_pad) <= pad_size, f"Cannot pad list with length {len(list_to_pad)} with pad size {pad_size}"

    padded_list = list_to_pad.copy()
    num_padding = pad_size - len(list_to_pad)
    padded_list.extend([padding for _ in range(num_padding)])
    return padded_list


def iter_proofs_in_file(filename, callback:Callable):
    file_data = json.load(open(filename))
    env = {"constants": [], "inductives": []}
    for proof_data in file_data["proofs"]:
        env = update_env(env, proof_data["env_delta"])
        del proof_data["env_delta"]
        proof_data["env"] = env
        callback(filename, proof_data)