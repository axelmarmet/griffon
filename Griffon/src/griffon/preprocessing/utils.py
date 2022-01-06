from glob import glob
import os
import shutil
from griffon.coq_dataclasses import Stage1Token

def connect_subtokens(token : Stage1Token)->str:
    return "_".join(token.subtokens)

def order_files(root:str):
    splits = ["train", "test", "valid"]
    for split in splits:
        proof_index = 0
        output_dir = os.path.join(root, split)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        tmp_root = os.path.join(root, "tmp", split)
        assert os.path.exists(tmp_root)

        files = sorted(glob(os.path.join(tmp_root, "**", "*.pickle"), recursive=True))
        for file in files:
            new_filepath = os.path.join(output_dir, 'proof{:08d}.pickle'.format(proof_index))
            proof_index += 1
            os.rename(file, new_filepath)

    shutil.rmtree(os.path.join(root, "tmp"))