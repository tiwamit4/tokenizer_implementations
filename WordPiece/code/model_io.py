"""JSON model storage; includes the settings required at inference time."""

import json
from pathlib import Path


def save_model(filename, vocabulary, lowercase, prefix, unk_token, max_chars):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = {
        "type": "WordPiece", "version": 1, "vocabulary": vocabulary,
        "lowercase": lowercase, "prefix": prefix, "unk_token": unk_token,
        "max_chars": max_chars, "pre_tokenization": "unicode_words_and_punctuation",
    }
    # Exclusive creation protects existing trained models.
    with path.open("x", encoding="utf-8") as file:
        json.dump(model, file, ensure_ascii=False, indent=2)


def load_model(filename):
    with Path(filename).open(encoding="utf-8") as file:
        model = json.load(file)
    if model.get("type") != "WordPiece" or model.get("version") != 1:
        raise ValueError("Unsupported WordPiece model")
    if model.get("pre_tokenization") != "unicode_words_and_punctuation":
        raise ValueError("Unsupported pre-tokenization")
    return model
