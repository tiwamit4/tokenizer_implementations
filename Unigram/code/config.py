"""Change training settings here, then run main.py."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPACE_MARKER = "▁"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = (UNK_TOKEN, "<pad>", "<s>", "</s>")
TARGET_VOCAB_SIZE = 30
INITIAL_VOCAB_SIZE = 60
MAX_PIECE_LENGTH = 12
PRUNE_FRACTION = 0.20
USE_DATASET = False
USE_ALL_TEXTS = False
MAX_TEXTS = 100
BATCH_SIZE = 4096
TEXT_COLUMN = "text"
TRAINING_DIR = PROJECT_ROOT / "datasets" / "wikitext-103-raw-v1"
TRAINING_PATTERN = "train*.parquet"
MODEL_PATH = PROJECT_ROOT / "Unigram" / "models" / "unigram_model.json"
EXAMPLE_CORPUS = (
    "hug hug hug hug hug hug hug hug hug hug",
    "pug pug pug pug pug",
    "pun pun pun pun pun pun pun pun pun pun pun pun",
    "bun bun bun bun",
    "hugs hugs hugs hugs hugs",
)
EXAMPLE_TEXT = "hug pug unhug"
