import unittest

import random

import numpy as np
import torch
from torch import nn

from griffon.models.verbose_transformer import VerboseTransformerDecoderLayer

class TestVerboseDecoderLayer(unittest.TestCase):

    def test_same_output(self):

        seed = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)

        D_MODEL = 4
        NHEAD = 2

        reference_layer = nn.TransformerDecoderLayer(D_MODEL, NHEAD)
        verbose_layer = nn.TransformerDecoderLayer(D_MODEL, NHEAD)
        # verbose_layer = VerboseTransformerDecoderLayer(D_MODEL, NHEAD)

        reference_layer.eval()
        verbose_layer.eval()

        BATCH_DIM = 1
        S_LEN = 1
        T_LEN = 1

        memory = torch.randn((S_LEN, BATCH_DIM, D_MODEL))
        tgt = torch.randn((T_LEN, BATCH_DIM, D_MODEL))

        reference_output = reference_layer(tgt, memory)

        torch.manual_seed(23)
        test_output = verbose_layer(tgt, memory)
        # test_output = reference_layer(tgt, memory)

        print(reference_output)
        print(test_output)
        self.assertTrue((reference_output == test_output).all())

if __name__ == '__main__':
    unittest.main()