"""Run Byte-Level BPE training with settings from config.py."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ByteLevelBPE.code import config
from ByteLevelBPE.code.data import iter_texts
from ByteLevelBPE.code.tokenizer import ByteLevelBPETokenizer


def main():
    if Path(config.MODEL_PATH).exists():
        raise FileExistsError("Model exists; change MODEL_PATH in config.py")
    corpus = config.EXAMPLE_CORPUS
    if config.USE_DATASET:
        corpus = iter_texts(
            config.TRAINING_DIR, config.TRAINING_PATTERN, config.TEXT_COLUMN,
            config.BATCH_SIZE, None if config.USE_ALL_TEXTS else config.MAX_TEXTS)
    tokenizer = ByteLevelBPETokenizer.train(
        corpus, config.TARGET_VOCAB_SIZE, config.SPECIAL_TOKENS,
        config.DECODE_ERRORS)
    tokenizer.save(config.MODEL_PATH)
    ids = tokenizer.encode(config.EXAMPLE_TEXT)
    print("Tokens:", tokenizer.tokenize(config.EXAMPLE_TEXT))
    print("IDs:", ids)
    print("Decoded:", tokenizer.decode(ids))
    print(f"Vocabulary size: {len(tokenizer.vocabulary)}")
    print(f"Saved: {config.MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ImportError, KeyError) as error:
        sys.exit(f"Byte-Level BPE failed: {error}")
