"""Run directly with python main.py; all settings come from config.py."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from WordPiece.code import config
from WordPiece.code.data import iter_texts
from WordPiece.code.tokenizer import WordPieceTokenizer
from WordPiece.code.training import train_wordpiece


def main():
    if Path(config.MODEL_PATH).exists():
        raise FileExistsError("Model exists; choose a new MODEL_PATH in config.py")
    corpus = config.EXAMPLE_CORPUS
    if config.USE_DATASET:
        corpus = iter_texts(config.TRAINING_DIR, config.TRAINING_PATTERN,
                            config.TEXT_COLUMN, config.BATCH_SIZE,
                            None if config.USE_ALL_TEXTS else config.MAX_TEXTS)
    print("Training WordPiece...")
    vocabulary = train_wordpiece(
        corpus, config.TARGET_VOCAB_SIZE, config.LOWERCASE,
        config.CONTINUATION_PREFIX, config.SPECIAL_TOKENS)
    tokenizer = WordPieceTokenizer(vocabulary, config.LOWERCASE,
                                   config.CONTINUATION_PREFIX, config.UNK_TOKEN,
                                   config.MAX_INPUT_CHARS_PER_WORD)
    tokenizer.save(config.MODEL_PATH)
    print(f"Vocabulary size: {len(vocabulary)}")
    print("Tokens:", tokenizer.tokenize(config.EXAMPLE_TEXT))
    print("IDs:", tokenizer.encode(config.EXAMPLE_TEXT))
    print(f"Saved: {config.MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ImportError, KeyError) as error:
        sys.exit(f"WordPiece failed: {error}")
