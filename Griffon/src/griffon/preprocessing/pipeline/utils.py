from griffon.coq_dataclasses import Stage1Token

def connect_subtokens(token : Stage1Token)->str:
    return "_".join(token.subtokens)