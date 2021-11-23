import math

import torch
import torch.nn as nn
from torch import Tensor

from griffon.preprocessing import Vocab

class TokenEmbedding(nn.Module):
    """
    A module that converts indices to embeddings.
    """
    def __init__(self, existing_vocab:Vocab):
        """Initialize the embedding layer from a preexisting vocabulary.

        Args:
            existing_vocab (Vocab): the pretrained vocabulary.
        """
        super().__init__()
        embedding_weights = torch.as_tensor(existing_vocab.word_vectors.vectors)
        self.embedding = nn.Embedding.from_pretrained(embeddings=embedding_weights,
                                                      freeze=False,
                                                      padding_idx=existing_vocab.PAD_IDX)
        self.emb_size = existing_vocab.get_embedding_dim()

    def forward(self, tokens: Tensor)->Tensor:
        """Compute the embeddings given indices in tensor .

        Args:
            tokens (Tensor): the indices to transform

        Returns:
            Tensor: the embeddings
        """
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
