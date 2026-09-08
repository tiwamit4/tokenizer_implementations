"""Run a small end-to-end BPE tokenizer demonstration."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from BPE.code.config import (  # type: ignore[import-not-found]
    DEFAULT_NUM_MERGES,
    EXAMPLE_CORPUS,
    EXAMPLE_TEXT,
)
from BPE.code.tokenizer import decode, encode_text
from BPE.code.training import train_bpe


def main() -> None:
    merges, token_vocab = train_bpe(
        EXAMPLE_CORPUS,
        num_merges=DEFAULT_NUM_MERGES,
    )
    encoded = encode_text(EXAMPLE_TEXT, merges, token_vocab)

    print("Merges:", merges)
    print("Tokens:", encoded)
    print("Decoded:", decode(encoded))


if __name__ == "__main__":
    main()
