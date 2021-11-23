
import argparse
from typing import Tuple
from timeit import default_timer as timer

import os
import pickle

from tqdm import tqdm

import torch

from torch import nn
from torch.nn import Transformer

# for typing
from torch import Tensor
from torch.optim import Optimizer

from griffon.preprocessing import StandardTextTransform, TextTransform, Vocab, Tokenizer
from griffon.dataloader import get_data_loader, UsefulItemsDataset
from griffon.models.encodings import PositionalEncoding, TokenEmbedding


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 nhead: int,
                 vocab: Vocab,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()
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


"""
    generate a lower triangle filled of 0, everything is -inf to prevent attention
"""
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask

"""
    create the various masks needed
"""
def create_mask(src, tgt, pad_idx):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),
                           device=device).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def create_network(text_transform: TextTransform,
                   nhead = 4,
                   ffn_hid_dim = 256,
                   num_encoder_layers = 3,
                   num_decoder_layers = 3)->Tuple[nn.Module, Optimizer, nn.Module]:
    """Create a transformer network using a given text transform.

    Args:
        text_transform (TextTransform): the text transformation pipeline
        nhead (int, optional): number of heads for the transformer Defaults to 4.
        ffn_hid_dim (int, optional): hidden dimension of feed forward. Defaults to 256.
        num_encoder_layers (int, optional): number of encoder layers. Defaults to 3.
        num_decoder_layers (int, optional): number of decoder layers. Defaults to 3.

    Returns:
    Tuple[nn.Module, Optimizer, nn.Module] A tuple of the model, its optimizer and its
    loss function
    """
    vocab = text_transform.vocab

    transformer = SimpleTransformer(num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers,
                                    nhead=nhead,
                                    vocab=vocab,
                                    dim_feedforward=ffn_hid_dim)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab.PAD_IDX)

    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    return transformer, optimizer, loss_fn


def train_network(model:nn.Module, optimizer:Optimizer, loss_fn:nn.Module,
                  text_transform:TextTransform, data_root:str, num_epochs=18,
                  batch_size=8, device="cpu"):

    vocab = text_transform.vocab
    transformer = model.to(device)

    def train_epoch(model, optimizer):
        model.train()
        losses = 0
        train_iter = UsefulItemsDataset(os.path.join(data_root, "train"))
        train_dataloader = get_data_loader(
            train_iter, text_transform, batch_size=batch_size)

        for src, tgt in tqdm(train_dataloader):
            if src.shape[0] > 300 or tgt.shape[0] > 300:
                continue

            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input, vocab.PAD_IDX)

            logits = model(src, tgt_input, src_mask, tgt_mask,
                           src_padding_mask, tgt_padding_mask, src_padding_mask)

            optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            losses += loss.item()

        return losses / len(train_dataloader)

    def evaluate(model):
        model.eval()
        losses = 0

        val_iter = UsefulItemsDataset(os.path.join(data_root, "valid"))
        val_dataloader = get_data_loader(
            val_iter, text_transform, batch_size=batch_size)

        for src, tgt in val_dataloader:

            if src.shape[0] > 300 or tgt.shape[0] > 300:
                continue

            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input, vocab.PAD_IDX)

            logits = model(src, tgt_input, src_mask, tgt_mask,
                           src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / len(val_dataloader)

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            (f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}," \
             f"Epoch time = {(end_time - start_time):.3f}s"))

    return transformer, train_losses, val_losses


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="""
        Train a simple seq2seq transformer
    """ )
    arg_parser.add_argument(
        "--data-root", type=str, required=True, help="""
            The root of the data that will be used to train and validate
            the model
        """
    )
    arg_parser.add_argument(
        "--vocab-file", type=str, required=True, help="""
            The path to the pickled vocab that will be used
        """
    )
    args = arg_parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device is {device}")
    # create text transform pipeline
    vocab = pickle.load(open(args.vocab_file, "rb"))
    tokenizer = Tokenizer()
    text_transform = StandardTextTransform(vocab, tokenizer, lambda t : torch.as_tensor(t))

    model, optimizer, loss_fn = create_network(text_transform)

    _, train_l, _ = train_network(model, optimizer, loss_fn, text_transform, args.data_root, num_epochs=3, device=device)