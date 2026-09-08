"""Vocabulary creation, pair counting, and pair merging."""

from collections import Counter
from collections.abc import Iterable

from .config import END_OF_WORD


def build_vocab(corpus: Iterable[str]) -> Counter:
    """Convert corpus text into a character-based word vocabulary."""
    vocab = Counter()

    for sentence in corpus:
        for word in sentence.split():
            tokens = tuple(list(word) + [END_OF_WORD])
            vocab[tokens] += 1

    return vocab


def get_pair_counts(vocab: Counter) -> Counter:
    """Count adjacent token pairs, weighted by word frequency."""
    pair_counts = Counter()

    for word_tokens, frequency in vocab.items():
        for index in range(len(word_tokens) - 1):
            pair = (word_tokens[index], word_tokens[index + 1])
            pair_counts[pair] += frequency

    return pair_counts


def merge_pair(pair: tuple[str, str], vocab: Counter) -> Counter:
    """Merge a selected pair throughout the vocabulary."""
    new_vocab = Counter()

    for word_tokens, frequency in vocab.items():
        merged_tokens = []
        index = 0

        while index < len(word_tokens):
            is_pair = (
                index < len(word_tokens) - 1
                and word_tokens[index] == pair[0]
                and word_tokens[index + 1] == pair[1]
            )

            if is_pair:
                merged_tokens.append(pair[0] + pair[1])
                index += 2
            else:
                merged_tokens.append(word_tokens[index])
                index += 1

        new_vocab[tuple(merged_tokens)] += frequency

    return new_vocab
