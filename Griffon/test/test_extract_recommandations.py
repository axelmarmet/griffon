from argparse import Namespace
from Griffon.extract_recommandations import process_dataset
import json
import os

def test_process_dataset():

    testing_dir = os.path.join(os.getcwd(), "test", "extraction_tests")
    os.chdir(testing_dir)

    args = Namespace(data_root="./data", filter=None)
    projs_split = json.load(open("projs_split.json"))
    
    res = process_dataset(projs_split, args)

    assert len(res["train"]) == 4
    assert len(res["valid"]) == 2
    assert len(res["test"]) == 8