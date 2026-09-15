"""Configuration values shared by the BPE tokenizer modules."""

# Special tokens
END_OF_WORD = "</w>"
UNK_TOKEN = "<unk>"

# Training defaults
DEFAULT_NUM_MERGES = 100

# # Demonstration data used by main.py
# EXAMPLE_CORPUS = (
#     "low lower lowest",
#     "newer wider",
#     "low low lower",
# )
# EXAMPLE_TEXT = "low lower zoo"

# Dataset download settings
DATASET_REPO_ID = "Salesforce/wikitext"
DATASET_CONFIG_NAME = "wikitext-103-raw-v1"

# Local training defaults (paths are relative to this project, not the terminal)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_DIR = PROJECT_ROOT / "datasets" / DATASET_CONFIG_NAME
DEFAULT_TRAINING_PATTERN = "train*.parquet"
DEFAULT_BATCH_SIZE = 4096
DEFAULT_MAX_TEXTS = 10_000
USE_ALL_TEXTS = False  # Set True only when ready to train on the entire corpus.
DEFAULT_MODEL_PATH = PROJECT_ROOT / "BPE" / "models" / "bpe_model.json"
DEFAULT_TEXT_COLUMN = "text"
