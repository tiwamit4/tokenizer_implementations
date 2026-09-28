"""Change training settings here, then run main.py."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPECIAL_TOKENS = ("<|endoftext|>",)
TARGET_VOCAB_SIZE = 300
DECODE_ERRORS = "strict"
USE_DATASET = False
USE_ALL_TEXTS = False
MAX_TEXTS = 1000
BATCH_SIZE = 4096
TEXT_COLUMN = "text"
TRAINING_DIR = PROJECT_ROOT / "datasets" / "wikitext-103-raw-v1"
TRAINING_PATTERN = "train*.parquet"
MODEL_PATH = PROJECT_ROOT / "ByteLevelBPE" / "models" / "byte_level_bpe_model.json"
EXAMPLE_CORPUS = (
    "hello world hello tokenizer",
    "Byte-level BPE keeps spaces and handles every UTF-8 character.",
    "नमस्ते दुनिया 👋 café",
)
EXAMPLE_TEXT = "hello 👋 café नमस्ते"
