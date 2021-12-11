from torchtext.vocab import Vocab

from griffon.coq_dataclasses import Stage1Sample, Stage1Statement, Stage2Token

class VocabTransform:

    def __init__(self, vocab:Vocab):
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
