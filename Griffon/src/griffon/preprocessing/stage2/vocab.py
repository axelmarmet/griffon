from typing import List
from torchtext.vocab import Vocab as _TorchtextVocab

from griffon.coq_dataclasses import Stage1Sample, Stage1Statement, Stage2Token

from abc import ABCMeta

class AbstractVocab(metaclass=ABCMeta):

    def __len__(self)->int:
        ...

    def get_itos(self)->List[str]:
        ...

    def set_default_index(self, index : int):
        ...

    def __getitem__(self, key:str)->int:
        ...

    def __contains__(self, key:str)->bool:
        ...

    def __call__(self, tokens: List[str]) -> List[int]:
        ...

class TorchtextVocab(AbstractVocab):

    def __init__(self, tt_vocab : _TorchtextVocab):
        self.tt_vocab = tt_vocab

    def __len__(self)->int:
        return len(self.tt_vocab)

    def get_itos(self)->List[str]:
        return self.tt_vocab.get_itos()

    def set_default_index(self, index:int):
        return self.tt_vocab.set_default_index(index)

    def __getitem__(self, key:str)->int:
        return self.tt_vocab[key]

    def __contains__(self, key:str)->bool:
        return key in self.tt_vocab

    def __call__(self, tokens: List[str]) -> List[int]:
        return self.tt_vocab(tokens)


class VocabTransform:

    def __init__(self, vocab:AbstractVocab):
        vocab.set_default_index(vocab["<unk>"])
        self.vocab = vocab

    def process_statement(self, statement:Stage1Statement, no_unk:bool=False)->Stage1Statement:

        vocabularized_tokens = []
        for i, token in enumerate(statement.tokens):
            assert all([isinstance(st, str) for st in
                        token.subtokens]), f"Some sub tokens ({token.subtokens}) do not have string values. Has this sample " \
                                           f"already been vocabularized?"
            if no_unk:
                assert all([subtoken in self.vocab for subtoken in token.subtokens]), \
                    f"Some sub tokens ({token.subtokens}) are not in the vocab"

            vocabularized_token = Stage2Token(subtokens=self.vocab(token.subtokens),
                                              original_subtokens=token.subtokens)

            vocabularized_tokens.append(vocabularized_token)

        statement.vocabularized_tokens = vocabularized_tokens
        return statement

    def __call__(self, sample:Stage1Sample)->Stage1Sample:

        for i, hypothesis in enumerate(sample.hypotheses):
            sample.hypotheses[i] = self.process_statement(hypothesis)

        sample.goal = self.process_statement(sample.goal)

        # we intentionnaly do not process the lemma used
        return sample
