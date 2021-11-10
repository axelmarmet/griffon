
import torch
import torch.nn as nn

from griffon.preprocessing import TextTransform
from griffon.models.simple_transformer import generate_square_subsequent_mask


def beam_search_decoder(predictions, top_k = 3):
    #start with an empty sequence with zero score
    output_sequences = [([], 0)]

    #looping through all the predictions
    for token_probs in predictions:
        new_sequences = []

        #append new tokens to old sequences and re-score
        for old_seq, old_score in output_sequences:
            for char_index in range(len(token_probs)):
                new_seq = old_seq + [char_index]
                #considering log-likelihood for scoring
                new_score = old_score + math.log(token_probs[char_index])
                new_sequences.append((new_seq, new_score))

        #sort all new sequences in the de-creasing order of their score
        output_sequences = sorted(new_sequences, key = lambda val: val[1], reverse = True)

        #select top-k based on score
        # *Note- best sequence is with the highest score
        output_sequences = output_sequences[:top_k]

    return output_sequences


def my_beam_search_decoder(model:nn.Module, input_str:str, text_transform:TextTransform, top_k = 3, max_len=50):
    # #start with an empty sequence with zero score
    # output_sequences = [([], 0)]
    vocab = text_transform.vocab
    input_seq = text_transform

    assert input_seq.shape[1] == 1

    num_tokens = input_seq.shape[0]
    src_mask = torch.zeros((num_tokens, num_tokens))

    memory = model.encode(input_seq, src_mask)
    ys = torch.ones(1, 1).fill_(vocab.BOS_IDX).type(torch.long)

    for _ in range(max_len):
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool))
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(input_seq.data).fill_(next_word)], dim=0)
        if next_word == vocab.EOS_IDX:
            break
    return ys


model = torch.load("models/model00:04:40")
inp_str = "ev (S (S (double n)))"