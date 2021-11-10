# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
#get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# 
# Language Translation with nn.Transformer and torchtext
# ======================================================
# 
# This tutorial shows, how to train a translation model from scratch using
# Transformer. We will be using `Multi30k <http://www.statmt.org/wmt16/multimodal-task.html#task1>`__ 
# dataset to train a German to English translation model.
# 
# 
# %% [markdown]
# Data Sourcing and Processing
# ----------------------------
# 
# `torchtext library <https://pytorch.org/text/stable/>`__ has utilities for creating datasets that can be easily
# iterated through for the purposes of creating a language translation
# model. In this example, we show how to use torchtext's inbuilt datasets, 
# tokenize a raw text sentence, build vocabulary, and numericalize tokens into tensor. We will use
# `Multi30k dataset from torchtext library <https://pytorch.org/text/stable/datasets.html#multi30k>`__
# that yields a pair of source-target raw sentences. 
# 
# 
# 
# 

# %%
import pickle
from griffon.preprocessing import Vocab, Tokenizer
import datetime

from torchtext.datasets.multi30k import Multi30k

vocab:Vocab = pickle.load(open("../models/vocab.pickle", "rb"))
tokenizer = Tokenizer()

# %% [markdown]
# Seq2Seq Network using Transformer
# ---------------------------------
# 
# Transformer is a Seq2Seq model introduced in `“Attention is all you
# need” <https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`__
# paper for solving machine translation tasks. 
# Below, we will create a Seq2Seq network that uses Transformer. The network
# consists of three parts. First part is the embedding layer. This layer converts tensor of input indices
# into corresponding tensor of input embeddings. These embedding are further augmented with positional
# encodings to provide position information of input tokens to the model. The second part is the 
# actual `Transformer <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`__ model. 
# Finally, the output of Transformer model is passed through linear layer
# that give un-normalized probabilities for each token in the target language. 
# 
# 
# 

# %%
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, existing_vocab:Vocab):
        super(TokenEmbedding, self).__init__()
        embedding_weights = torch.as_tensor(existing_vocab.word_vectors.vectors)
        self.embedding = nn.Embedding.from_pretrained(embeddings=embedding_weights,
                                                      freeze=False,
                                                      padding_idx=existing_vocab.PAD_IDX)
        self.emb_size = existing_vocab.get_embedding_dim()

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network 
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 nhead: int,
                 vocab:Vocab,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        emb_size = vocab.get_embedding_dim()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, len(vocab))
        self.src_tok_emb = TokenEmbedding(vocab)
        self.tgt_tok_emb = TokenEmbedding(vocab)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, 
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

# %% [markdown]
# During training, we need a subsequent word mask that will prevent model to look into
# the future words when making predictions. We will also need masks to hide
# source and target padding tokens. Below, let's define a function that will take care of both. 
# 
# 
# 

# %%
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == vocab.PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == vocab.PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# %% [markdown]
# Let's now define the parameters of our model and instantiate the same. Below, we also 
# define our loss function which is the cross-entropy loss and the optmizer used for training.
# 
# 
# 

# %%

SRC_VOCAB_SIZE = len(vocab)
TGT_VOCAB_SIZE = len(vocab)
EMB_SIZE = 64
NHEAD = 4
FFN_HID_DIM = 256
BATCH_SIZE = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(num_encoder_layers=NUM_ENCODER_LAYERS, 
                                 num_decoder_layers=NUM_DECODER_LAYERS, 
                                 nhead=NHEAD,
                                 vocab=vocab,
                                 dim_feedforward=FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab.PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# %% [markdown]
# Collation
# ---------
# 
# As seen in the ``Data Sourcing and Processing`` section, our data iterator yields a pair of raw strings. 
# We need to convert these string pairs into the batched tensors that can be processed by our ``Seq2Seq`` network 
# defined previously. Below we define our collate function that convert batch of raw strings into batch tensors that
# can be fed directly into our model.   
# 
# 
# 
# %% [markdown]
# Let's define training and evaluation loop that will be called for each 
# epoch.
# 
# 
# 

# %%
from tqdm import tqdm
from griffon.dataloader import get_data_loader, UsefulItemsDataset

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = UsefulItemsDataset("../data/processed/recommandations/train")
    train_dataloader = get_data_loader(train_iter, vocab, tokenizer, batch_size=BATCH_SIZE)
    
    for src, tgt in tqdm(train_dataloader):
        if src.shape[0] > 300 or tgt.shape[0] > 300:
            continue

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = UsefulItemsDataset("../data/processed/recommandations/valid")
    val_dataloader = get_data_loader(val_iter, vocab, tokenizer, batch_size=BATCH_SIZE)

    for src, tgt in val_dataloader:
        
        if src.shape[0] > 300 or tgt.shape[0] > 300:
            continue
        
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)

# %% [markdown]
# Now we have all the ingredients to train our model. Let's do it!
# 
# 
# 

# %%
from timeit import default_timer as timer
NUM_EPOCHS = 18

train_losses = []
val_losses = []

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(( f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

time = datetime.datetime.now().strftime("%X")
torch.save(transformer.state_dict(), "../models/model" + time)

with open("../models/val_losses" + time + ".txt", "wb") as f:
    pickle.dump(val_losses, f)
with open("../models/train_losses" + time + ".txt", "wb") as f:
    pickle.dump(train_losses, f)

# function to generate output sequence using greedy algorithm 
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


# %%
# print(translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu ."))

# %% [markdown]
# References
# ----------
# 
# 1. Attention is all you need paper.
#    https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
# 2. The annotated transformer. https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
# 
# 

