"""Train a compact Unigram model by measuring token-removal loss."""

from collections import Counter
from math import ceil, log

from .config import (
    INITIAL_VOCAB_SIZE, MAX_PIECE_LENGTH, PRUNE_FRACTION, SPACE_MARKER,
    SPECIAL_TOKENS, TARGET_VOCAB_SIZE,
)
from .preprocessing import pre_tokenize
from .segmentation import viterbi


def build_initial_model(words, initial_size, max_piece_length):
    characters = Counter()
    substrings = Counter()
    for word, frequency in words.items():
        for start, character in enumerate(word):
            characters[character] += frequency
            for end in range(start + 2, min(len(word), start + max_piece_length) + 1):
                substrings[word[start:end]] += frequency
    if initial_size < len(characters):
        raise ValueError(f"Initial vocabulary must fit {len(characters)} characters")
    ranked = sorted(substrings.items(), key=lambda item: (-item[1], item[0]))
    frequencies = dict(characters)
    for piece, frequency in ranked[:initial_size - len(characters)]:
        frequencies[piece] = frequency
    total = sum(frequencies.values())
    return {piece: -log(frequency / total) for piece, frequency in frequencies.items()}, set(characters)


def corpus_loss(words, costs):
    total = 0.0
    for word, frequency in words.items():
        _, loss = viterbi(word, costs)
        if loss == float("inf"):
            return loss
        total += frequency * loss
    return total


def train_unigram(corpus, target_size=TARGET_VOCAB_SIZE,
                  initial_size=INITIAL_VOCAB_SIZE,
                  max_piece_length=MAX_PIECE_LENGTH,
                  prune_fraction=PRUNE_FRACTION, marker=SPACE_MARKER,
                  special_tokens=SPECIAL_TOKENS):
    if not 0 < prune_fraction < 1:
        raise ValueError("prune_fraction must be between 0 and 1")
    if max_piece_length < 1:
        raise ValueError("max_piece_length must be positive")
    words = Counter()
    for text in corpus:
        words.update(pre_tokenize(text, marker))
    if not words:
        raise ValueError("Training corpus is empty")
    piece_target = target_size - len(dict.fromkeys(special_tokens))
    initial_piece_size = initial_size - len(dict.fromkeys(special_tokens))
    costs, required = build_initial_model(words, initial_piece_size, max_piece_length)
    if piece_target < len(required):
        raise ValueError(f"Target vocabulary must retain {len(required)} characters plus special tokens")
    if piece_target > len(costs):
        raise ValueError(f"Target vocabulary exceeds {len(costs) + len(special_tokens)} initial tokens")
    while len(costs) > piece_target:
        baseline = corpus_loss(words, costs)
        removable = [piece for piece in costs if piece not in required]
        removal_scores = []
        for piece in removable:
            reduced = dict(costs)
            del reduced[piece]
            removal_scores.append((corpus_loss(words, reduced) - baseline, piece))
        removal_scores.sort(key=lambda item: (item[0], item[1]))
        count = min(
            len(costs) - piece_target,
            max(1, ceil(len(removable) * prune_fraction)),
        )
        for _, piece in removal_scores[:count]:
            del costs[piece]
    return costs
