"""Configuration values shared by the BPE tokenizer modules."""

# Special tokens
END_OF_WORD = "</w>"
UNK_TOKEN = "<unk>"

# Training defaults
DEFAULT_NUM_MERGES = 10

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
