from torchtext.vocab import Vocab

from griffon.coq_dataclasses import Stage1Sample, Stage1Statement

class VocabTransform:

    def __init__(self, vocab:Vocab):
        vocab.set_default_index(vocab["<unk>"])
        self.vocab = vocab

    def process_statement(self, statement:Stage1Statement):
        for i, token in enumerate(statement.tokens):
            assert all([isinstance(st, str) for st in
                        token.subtokens]), f"Some sub tokens ({token.subtokens}) do not have string values. Has this sample " \
                                           f"already been vocabularized?"
            statement.tokens[i].subtokens = self.vocab(token.subtokens)

        return statement

    def __call__(self, sample:Stage1Sample):

        for i, hypothesis in enumerate(sample.hypotheses):
            sample.hypotheses[i] = self.process_statement(hypothesis)

        sample.goal = self.process_statement(sample.goal)

        for i, token in enumerate(sample.lemma_used):
            assert all([isinstance(st, str) for st in token.subtokens]), \
                f"Some sub tokens ({token.subtokens}) do not have string values. Has this sample " \
                f"already been vocabularized?"

            sample.lemma_used[i].subtokens = self.vocab(token.subtokens)

        return sample
