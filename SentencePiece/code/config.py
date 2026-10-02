"""Change SentencePiece settings here, then run main.py."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_TYPE = "unigram"  # Use "unigram" or "bpe".
VOCAB_SIZE = 64
CHARACTER_COVERAGE = 1.0
BYTE_FALLBACK = False
HARD_VOCAB_LIMIT = False
USE_DATASET = False
USE_ALL_TEXTS = False
MAX_TEXTS = 1000
BATCH_SIZE = 4096
TEXT_COLUMN = "text"
TRAINING_DIR = PROJECT_ROOT / "datasets" / "wikitext-103-raw-v1"
TRAINING_PATTERN = "train*.parquet"
MODEL_PREFIX = PROJECT_ROOT / "SentencePiece" / "models" / MODEL_TYPE
EXAMPLE_CORPUS = (
    "SentencePiece trains directly from raw sentences.",
    "It represents spaces with a visible boundary marker.",
    "नमस्ते दुनिया। こんにちは世界。 Hello world.",
    "Unigram and BPE are both supported.",
)
EXAMPLE_TEXT = "Hello दुनिया こんにちは"
