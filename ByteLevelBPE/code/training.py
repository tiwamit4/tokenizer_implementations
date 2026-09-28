"""Train byte-level BPE merge rules."""

from collections import Counter

from .bytes_mapping import BYTE_ENCODER
from .config import SPECIAL_TOKENS, TARGET_VOCAB_SIZE
from .preprocessing import pre_tokenize


def merge_symbols(symbols, pair):
    result = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and symbols[index:index + 2] == pair:
            result.append(pair[0] + pair[1])
            index += 2
        else:
            result.append(symbols[index])
            index += 1
    return tuple(result)


def train_byte_level_bpe(corpus, vocab_size=TARGET_VOCAB_SIZE,
                         special_tokens=SPECIAL_TOKENS):
    base_size = len(dict.fromkeys(special_tokens)) + 256
    if vocab_size < base_size:
        raise ValueError(f"vocab_size must be at least {base_size}")
    chunks = Counter()
    for text in corpus:
        for chunk in pre_tokenize(text):
            symbols = tuple(BYTE_ENCODER[value] for value in chunk.encode("utf-8"))
            chunks[symbols] += 1
    if not chunks:
        raise ValueError("Training corpus is empty")
    merges = []
    known = set(BYTE_ENCODER.values()) | set(special_tokens)
    while len(known) < vocab_size:
        pair_counts = Counter()
        for symbols, frequency in chunks.items():
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += frequency
        if not pair_counts:
            break
        best = min(pair_counts, key=lambda pair: (-pair_counts[pair], pair))
        updated = Counter()
        for symbols, frequency in chunks.items():
            updated[merge_symbols(symbols, best)] += frequency
        chunks = updated
        merges.append(best)
        known.add(best[0] + best[1])
    return merges
