"""Build a WordPiece vocabulary using pair scores."""

from collections import Counter
from fractions import Fraction

from .config import CONTINUATION_PREFIX, LOWERCASE, SPECIAL_TOKENS, TARGET_VOCAB_SIZE
from .preprocessing import pre_tokenize


def get_pair_scores(words):
    token_counts = Counter()
    pair_counts = Counter()
    for symbols, frequency in words.items():
        for symbol in symbols:
            token_counts[symbol] += frequency
        for pair in zip(symbols, symbols[1:]):
            pair_counts[pair] += frequency
    return {
        pair: Fraction(count, token_counts[pair[0]] * token_counts[pair[1]])
        for pair, count in pair_counts.items()
    }


def merge_pair(words, pair, prefix):
    updated = Counter()
    merged = pair[0] + pair[1][len(prefix):]
    for symbols, frequency in words.items():
        result = []
        index = 0
        while index < len(symbols):
            if symbols[index:index + 2] == pair:
                result.append(merged)
                index += 2
            else:
                result.append(symbols[index])
                index += 1
        updated[tuple(result)] += frequency
    return updated, merged


def train_wordpiece(corpus, vocab_size=TARGET_VOCAB_SIZE, lowercase=LOWERCASE,
                    prefix=CONTINUATION_PREFIX, special_tokens=SPECIAL_TOKENS):
    if not prefix:
        raise ValueError("Continuation prefix must not be empty")
    if vocab_size <= 0:
        raise ValueError("Vocabulary size must be positive")
    words = Counter()
    for text in corpus:
        for word in pre_tokenize(text, lowercase):
            words[(word[0],) + tuple(prefix + char for char in word[1:])] += 1
    if not words:
        raise ValueError("Training corpus is empty")
    alphabet = {symbol for symbols in words for symbol in symbols}
    vocabulary = list(dict.fromkeys(special_tokens))
    vocabulary.extend(sorted(alphabet - set(vocabulary)))
    if len(vocabulary) > vocab_size:
        raise ValueError(f"Vocabulary size must be at least {len(vocabulary)} to retain the alphabet")
    known = set(vocabulary)
    while len(vocabulary) < vocab_size:
        scores = get_pair_scores(words)
        if not scores:
            break
        best = min(scores, key=lambda pair: (-scores[pair], pair))
        words, merged = merge_pair(words, best, prefix)
        if merged not in known:
            vocabulary.append(merged)
            known.add(merged)
    return vocabulary
