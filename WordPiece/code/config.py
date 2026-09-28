"""Change implementation and training settings here, then run main.py."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTINUATION_PREFIX = "##"
UNK_TOKEN = "[UNK]"
SPECIAL_TOKENS = ("[PAD]", UNK_TOKEN, "[CLS]", "[SEP]", "[MASK]")
LOWERCASE = False
MAX_INPUT_CHARS_PER_WORD = 100
TARGET_VOCAB_SIZE = 1000
USE_DATASET = False
USE_ALL_TEXTS = False
MAX_TEXTS = 1000
BATCH_SIZE = 4096
TEXT_COLUMN = "text"
TRAINING_DIR = PROJECT_ROOT / "datasets" / "wikitext-103-raw-v1"
TRAINING_PATTERN = "train*.parquet"
MODEL_PATH = PROJECT_ROOT / "WordPiece" / "models" / "wordpiece_model.json"
EXAMPLE_CORPUS = (
    "play playing played player",
    "walk walking walked walker",
    "I am playing and walking.",
    "Tokenizers split words into smaller pieces.",
)
EXAMPLE_TEXT = "playing walking unseen"
