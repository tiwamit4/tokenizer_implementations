"""Whitespace marking shared by training and tokenization."""

import re


def pre_tokenize(text, marker="▁"):
    return [marker + word for word in re.findall(r"\S+", text)]


def restore_spaces(tokens, marker="▁"):
    return "".join(tokens).replace(marker, " ").lstrip()
