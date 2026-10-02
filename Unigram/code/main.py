"""Run Unigram training using config.py."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Unigram.code import config
from Unigram.code.data import iter_texts
from Unigram.code.tokenizer import UnigramTokenizer


def main():
    if Path(config.MODEL_PATH).exists():
        raise FileExistsError("Model exists; change MODEL_PATH in config.py")
    corpus = config.EXAMPLE_CORPUS
    if config.USE_DATASET:
        corpus = iter_texts(
            config.TRAINING_DIR, config.TRAINING_PATTERN, config.TEXT_COLUMN,
            config.BATCH_SIZE, None if config.USE_ALL_TEXTS else config.MAX_TEXTS)
    tokenizer = UnigramTokenizer.train(
        corpus, config.TARGET_VOCAB_SIZE, config.INITIAL_VOCAB_SIZE,
        config.MAX_PIECE_LENGTH, config.PRUNE_FRACTION,
        config.SPECIAL_TOKENS, config.SPACE_MARKER, config.UNK_TOKEN)
    tokenizer.save(config.MODEL_PATH)
    tokens = tokenizer.tokenize(config.EXAMPLE_TEXT)
    ids = tokenizer.encode(config.EXAMPLE_TEXT)
    print("Tokens:", tokens)
    print("IDs:", ids)
    print("Decoded:", tokenizer.decode(ids))
    print(f"Vocabulary size: {len(tokenizer.vocabulary)}")
    print(f"Saved: {config.MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ImportError, KeyError) as error:
        sys.exit(f"Unigram failed: {error}")
