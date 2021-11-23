
import argparse
from typing import Tuple, Optional
from timeit import default_timer as timer

import os
import pickle

from tqdm import tqdm

import torch

from torch import nn

# for typing
from torch import Tensor
from torch.optim import Optimizer

from griffon.models import VerboseTransformer, CopyGenerator

from griffon.preprocessing import CopyingTextTransform, TextTransform, Vocab, Tokenizer
from griffon.dataloader import get_data_loader, UsefulItemsDataset
from griffon.models.verbose_transformer import VerboseTransformer
from griffon.models.encodings.positional_encoding import PositionalEncoding
from griffon.models.encodings.token_embedding import TokenEmbedding
from griffon.models.copy_generator import CopyGeneratorLoss


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class CopyingTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 nhead: int,
                 vocab: Vocab,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        emb_size = vocab.get_embedding_dim()
        self.generator_hidden_dim = 3*emb_size

        self.transformer = VerboseTransformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.copy_generator = CopyGenerator(self.generator_hidden_dim, len(vocab), vocab.PAD_IDX)
        self.src_tok_emb = TokenEmbedding(vocab)
        self.tgt_tok_emb = TokenEmbedding(vocab)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                src_map : Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                memory_mask: Optional[Tensor] = None,
                src_padding_mask: Optional[Tensor] = None,
                tgt_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):

        memory = self.encode(src, src_mask, src_padding_mask)
        outs, attention = self.decode(tgt, memory, tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)

        # we create the adapted input for the formula p_gen = σ(w_h ∗h_t +w_s s_t +w_x x_t +bptr)
        # from DBLP:journals/corr/SeeLM17

        # h becomes the output of the encoder, summed with attention by last attention of the decoder
        h = torch.bmm(
            attention,
            memory.transpose(1,0)
        ).transpose(1,0)
        # s is simply the output of the decoder
        s = outs
        # x is equivalent
        x = self.tgt_tok_emb(tgt)

        generator_hidden = torch.cat([h, s, x], dim=-1)

        # reshape the tensors
        attention = attention.reshape(-1, src.shape[0])
        generator_hidden = generator_hidden.reshape(-1, self.generator_hidden_dim)

        return self.copy_generator(generator_hidden, attention, src_map)

    def encode(self, src: Tensor, src_mask: Tensor, src_key_padding_mask:Optional[Tensor] = None)->Tensor:
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer.encoder(src_emb, src_mask, src_key_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor,
               memory_mask: Optional[Tensor] = None,
               tgt_key_padding_mask: Optional[Tensor] = None,
               memory_key_padding_mask: Optional[Tensor] = None)->Tuple[Tensor, Tensor, Tensor]:

        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)

""""
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

    transformer = CopyingTransformer(num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers,
                                    nhead=nhead,
                                    vocab=vocab,
                                    dim_feedforward=ffn_hid_dim)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    loss_fn = CopyGeneratorLoss(len(vocab), True, ignore_index=vocab.PAD_IDX, unk_index=-1)

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

        train_iter = UsefulItemsDataset(os.path.join(data_root, "valid"))
        train_dataloader = get_data_loader(
            train_iter, text_transform, batch_size=batch_size)

        for src, src_map, tgt, tgt_align in tqdm(train_dataloader):
            if src.shape[0] > 300 or tgt.shape[0] > 300:
                continue

            src = src.to(device)
            src_map = src_map.to(device)
            tgt = tgt.to(device)
            tgt_align = tgt_align.to(device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input, vocab.PAD_IDX)

            probs = model.forward(src=src, src_map=src_map, tgt=tgt_input,
                                  src_mask=src_mask, tgt_mask=tgt_mask,
                                  src_padding_mask=src_padding_mask,
                                  tgt_padding_mask=tgt_padding_mask)



            # we don't care about the initial <bos> token
            tgt_align = tgt_align[1:, :]
            tgt_out = tgt[1:, :]

            tgt_align = tgt_align.reshape(-1)
            tgt_out = tgt_out.reshape(-1)

            optimizer.zero_grad()
            loss = loss_fn(probs, tgt_align, tgt_out)
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

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"device is {device}")
    # create text transform pipeline
    vocab = pickle.load(open(args.vocab_file, "rb"))
    tokenizer = Tokenizer()
    # text_transform = StandardTextTransform(vocab, tokenizer, lambda t : torch.as_tensor(t))
    text_transform = CopyingTextTransform(vocab, tokenizer)

    model, optimizer, loss_fn = create_network(text_transform)

    _, train_l, _ = train_network(model, optimizer, loss_fn, text_transform, args.data_root, num_epochs=3, device=device)