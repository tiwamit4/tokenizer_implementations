"""Encoding and decoding with a trained BPE model."""

from collections.abc import Iterable

from .config import END_OF_WORD, UNK_TOKEN


def encode_word(
    word: str,
    merges: list[tuple[str, str]],
    token_vocab: set[str],
) -> list[str]:
    """Convert one word into BPE tokens."""
    tokens = list(word) + [END_OF_WORD]

    # Replay the learned rules in training order.
    for first, second in merges:
        new_tokens = []
        index = 0

        while index < len(tokens):
            is_pair = (
                index < len(tokens) - 1
                and tokens[index] == first
                and tokens[index + 1] == second
            )

            if is_pair:
                new_tokens.append(first + second)
                index += 2
            else:
                new_tokens.append(tokens[index])
                index += 1

        tokens = new_tokens

    return [token if token in token_vocab else UNK_TOKEN for token in tokens]


def encode_text(
    text: str,
    merges: list[tuple[str, str]],
    token_vocab: set[str],
) -> list[str]:
    """Convert whitespace-separated text into BPE tokens."""
    return [
        token
        for word in text.split()
        for token in encode_word(word, merges, token_vocab)
    ]


def decode(tokens: Iterable[str]) -> str:
    """Turn BPE tokens back into readable, whitespace-separated text."""
    return "".join(tokens).replace(END_OF_WORD, " ").strip()
