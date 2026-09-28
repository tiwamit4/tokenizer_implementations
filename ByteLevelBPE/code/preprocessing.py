"""GPT-2-style pre-tokenization, including leading spaces and contractions."""

import regex


PATTERN = regex.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def pre_tokenize(text):
    return PATTERN.findall(text)
