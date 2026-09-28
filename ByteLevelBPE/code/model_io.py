"""JSON persistence for byte-level BPE models."""

import json
from pathlib import Path


def save_model(filename, vocabulary, merges, special_tokens, errors):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "type": "ByteLevelBPE", "version": 1,
        "pre_tokenization": "gpt2-regex-v1",
        "vocabulary": vocabulary, "merges": merges,
        "special_tokens": list(special_tokens), "decode_errors": errors,
    }
    with path.open("x", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_model(filename):
    with Path(filename).open(encoding="utf-8") as file:
        data = json.load(file)
    if data.get("type") != "ByteLevelBPE" or data.get("version") != 1:
        raise ValueError("Unsupported byte-level BPE model")
    if data.get("pre_tokenization") != "gpt2-regex-v1":
        raise ValueError("Unsupported pre-tokenization")
    return data
