"""Train SentencePiece using config.py."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SentencePiece.code import config
from SentencePiece.code.data import iter_texts
from SentencePiece.code.tokenizer import SentencePieceTokenizer


def main():
    corpus = config.EXAMPLE_CORPUS
    if config.USE_DATASET:
        corpus = iter_texts(
            config.TRAINING_DIR, config.TRAINING_PATTERN, config.TEXT_COLUMN,
            config.BATCH_SIZE, None if config.USE_ALL_TEXTS else config.MAX_TEXTS)
    tokenizer = SentencePieceTokenizer.train(
        corpus, config.MODEL_PREFIX, config.MODEL_TYPE, config.VOCAB_SIZE,
        config.CHARACTER_COVERAGE, config.BYTE_FALLBACK,
        config.HARD_VOCAB_LIMIT)
    pieces = tokenizer.tokenize(config.EXAMPLE_TEXT)
    ids = tokenizer.encode(config.EXAMPLE_TEXT)
    print("Model type:", config.MODEL_TYPE)
    print("Pieces:", pieces)
    print("IDs:", ids)
    print("Decoded:", tokenizer.decode(ids))
    print("Vocabulary size:", tokenizer.vocabulary_size)
    print("Model:", tokenizer.model_file)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        sys.exit(f"SentencePiece failed: {error}")
