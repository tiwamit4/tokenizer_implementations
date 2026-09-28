# Byte-Level BPE from scratch

This tokenizer begins with all 256 byte values, so every UTF-8 string can be
encoded without an unknown token. It learns frequent adjacent-pair merges and
replays those merges in rank order when encoding new text.

## Flow

```text
Text
  -> GPT-2-style pre-tokenization
  -> UTF-8 bytes
  -> reversible byte-to-Unicode symbols
  -> count adjacent pairs
  -> merge the most frequent pair
  -> repeat until target size or no pairs remain
  -> save vocabulary, ordered merges, and settings

New text
  -> same pre-tokenization and byte mapping
  -> apply learned merges by rank
  -> vocabulary IDs
  -> reverse byte mapping
  -> decode UTF-8 text
```

Spaces are retained inside pre-tokenized chunks. Non-ASCII characters use
multiple UTF-8 bytes before merges are applied.

## Run

For a cell-by-cell walkthrough, open
[byte_level_bpe_step_by_step.ipynb](byte_level_bpe_step_by_step.ipynb) with the
`Python (.venv - tokenizers)` kernel and run the cells in order.

From `ByteLevelBPE/code`:

```powershell
python main.py
```

Or from the project root:

```powershell
python -m ByteLevelBPE.code.main
```

All settings are in `code/config.py`. The default run uses a small built-in
corpus and saves `ByteLevelBPE/models/byte_level_bpe_model.json`. Existing
models are not overwritten. Set `USE_DATASET = True` to read a limited sample
from the local WikiText Parquet files. No data is downloaded.

## Files

- `bytes_mapping.py`: reversible mapping for all 256 byte values.
- `preprocessing.py`: GPT-2-style Unicode-aware splitting.
- `training.py`: pair counts and merge learning.
- `tokenizer.py`: tokenization, IDs, decoding, and model interface.
- `model_io.py`: JSON persistence.
- `data.py`: optional local Parquet reader.
- `main.py`: configuration-driven entry point.

## Python example

```python
from ByteLevelBPE.code import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer.train(
    ["hello world", "नमस्ते 👋 café"],
    vocab_size=280,
)
ids = tokenizer.encode("unseen 🧪 text")
print(ids)
print(tokenizer.decode(ids))
tokenizer.save("byte_level_bpe_model.json")
```

The implementation follows the reversible byte mapping and pre-tokenization
used in [OpenAI's GPT-2 encoder](https://github.com/openai/gpt-2/blob/master/src/encoder.py).
It is written for study: training recomputes pair counts after every merge and
keeps the unique pre-tokenized chunks in memory.

Run all tests from the project root:

```powershell
python -m unittest discover -s tests -v
```
