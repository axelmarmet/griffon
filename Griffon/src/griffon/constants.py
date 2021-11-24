NUM_BINS = 32
BIN_PADDING = 0  # Padding value to be used for discrete distance matrices when #unique values < NUM_BINS

SOS_TOKEN = "<sos>"
UNK_TOKEN = "<unk>"
MASK_TOKEN = "<mask>"
PAD_TOKEN = "<pad>"

SPECIAL_TOKENS = [
    SOS_TOKEN, UNK_TOKEN, MASK_TOKEN, PAD_TOKEN
]

MAX_NUM_TOKEN  = 128
NUM_SUB_TOKENS = 5