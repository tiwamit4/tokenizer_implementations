"""Public interface for the from-scratch BPE tokenizer."""

from .config import DEFAULT_NUM_MERGES, END_OF_WORD, UNK_TOKEN
from .model_io import load_model, save_model
from .tokenizer import decode, encode_text, encode_word
from .training import train_bpe
from .optimized_training import train_bpe_optimized
from .vocabulary import build_vocab, get_pair_counts, merge_pair

__all__ = [
    "END_OF_WORD",
    "DEFAULT_NUM_MERGES",
    "UNK_TOKEN",
    "build_vocab",
    "decode",
    "encode_text",
    "encode_word",
    "get_pair_counts",
    "load_model",
    "merge_pair",
    "save_model",
    "train_bpe",
    "train_bpe_optimized",
]
