import os
from typing import List, Tuple
from griffon.preprocessing.stage1.recreate_term import Stage1StatementCreator
from griffon.preprocessing.stage2.stage2 import get_distances_transformer, get_vocab_transformer
from griffon.utils import load_config
from nltk.tokenize.regexp import RegexpTokenizer

from griffon.coq_dataclasses import GriffonStatementBatch
from CoqGym.gallina import GallinaTermParser

def get_statement_batch(statements:List[Tuple[str, str]])->str:

    MODEL_DIR = "/home/axel/Documents/master_proj/Griffon/Griffon/models"

    statement_creator = Stage1StatementCreator(GallinaTermParser(caching=False), RegexpTokenizer(r"[^\W_]+|[:,().]"))
    for sexp, _ in statements:
        print(sexp)
    stage1_statements = [statement_creator(sexp, id) for sexp, id in statements]



    vocab_transformer = get_vocab_transformer(os.path.join(MODEL_DIR, "vocab.pickle"))
    distances_transformer = get_distances_transformer(
        load_config(os.path.join(MODEL_DIR, "distance_config.json"))
    )

    stage2_statements = [distances_transformer.process_statement(
        vocab_transformer.process_statement(statement)
    ) for statement in stage1_statements]

    return "success"

def log_statements(title:str, statements:List[Tuple[str, str]]):

    MODEL_DIR = "/home/axel/Documents/master_proj/Griffon/Griffon/models"
    CHECK_DIR = "/home/axel/Documents/master_proj/Griffon/Griffon/checks"

    statement_creator = Stage1StatementCreator(GallinaTermParser(caching=False), RegexpTokenizer(r"[^\W_]+|[:,().]"))
    stage1_statements = [statement_creator(sexp, id) for sexp, id in statements]
    with open(os.path.join(CHECK_DIR, title), "w") as f:
        for statement in stage1_statements:
            f.write(str(statement) + "\n")

