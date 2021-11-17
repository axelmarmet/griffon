import os

import json

from typing import Callable, List, Optional, TypeVar

T = TypeVar('T')
def find_in_list(l:List[T], f:Callable[[T], bool])->Optional[T]:
    return next((x for x in l if f(x)), None)

def get_path_relative_to_data(data_root, filename):
    filename:str = filename.replace(data_root, "", 1)
    assert filename[0] == os.path.sep
    return filename[1:]

def load_config(path:str):
    return json.load(open(path, "r"))