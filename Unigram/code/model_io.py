"""JSON persistence for Unigram models."""

import json
from pathlib import Path


def save_model(filename, vocabulary, scores, special_tokens, marker, unk_token):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "type": "Unigram", "version": 1,
        "vocabulary": vocabulary, "scores": scores,
        "special_tokens": list(special_tokens),
        "space_marker": marker, "unk_token": unk_token,
    }
    with path.open("x", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_model(filename):
    with Path(filename).open(encoding="utf-8") as file:
        data = json.load(file)
    if data.get("type") != "Unigram" or data.get("version") != 1:
        raise ValueError("Unsupported Unigram model")
    if len(data["vocabulary"]) != len(data["scores"]):
        raise ValueError("Vocabulary and score lengths differ")
    return data
