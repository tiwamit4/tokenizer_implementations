"""JSON persistence for trained BPE models."""

import json
from pathlib import Path

from .config import END_OF_WORD, UNK_TOKEN


def save_model(
    filename: str | Path,
    merges: list[tuple[str, str]],
    token_vocab: set[str],
    num_merges: int,
) -> None:
    """Save a trained tokenizer as JSON."""
    model = {
        "num_merges": num_merges,
        "end_of_word": END_OF_WORD,
        "unknown_token": UNK_TOKEN,
        "merges": merges,
        "token_vocab": sorted(token_vocab),
    }

    with Path(filename).open("w", encoding="utf-8") as file:
        json.dump(model, file, ensure_ascii=False, indent=2)


def load_model(
    filename: str | Path,
) -> tuple[list[tuple[str, str]], set[str], int]:
    """Load merges, vocabulary, and configuration from a JSON model."""
    with Path(filename).open("r", encoding="utf-8") as file:
        model = json.load(file)

    merges = [tuple(pair) for pair in model["merges"]]
    token_vocab = set(model["token_vocab"])

    return merges, token_vocab, model["num_merges"]
