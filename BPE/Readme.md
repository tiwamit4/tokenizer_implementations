Large dataset
    ↓
Read in batches/chunks
    ↓
Clean and normalize text
    ↓
Pre-tokenize text into words
    ↓
Build word-frequency counts
    ↓
Initialize byte/character vocabulary
    ↓
Count adjacent token pairs
    ↓
Merge the most frequent pair
    ↓
Update only affected pair counts
    ↓
Repeat until vocabulary-size target
    ↓
Save vocabulary + merges + configuration
    ↓
Evaluate on validation data

---

# BPE tokenizer implementation

This folder contains a character-level Byte Pair Encoding tokenizer written
from scratch. The existing large-dataset flow above describes how the project
can later be scaled with batch processing and incremental pair-count updates.

## Code structure

```text
BPE/
|-- Readme.md
|
`-- code/
    |-- __init__.py       Public package interface
    |-- config.py         Tokens, defaults, and example settings
    |-- vocabulary.py     Vocabulary building and pair operations
    |-- training.py       BPE training loop
    |-- tokenizer.py      Encoding and decoding
    |-- model_io.py       JSON model saving and loading
    `-- main.py           Program entry point and demonstration
```

## Run the example

Run this command from the `All_Tokenizer` project directory:

```powershell
python -m BPE.code.main
```

Alternatively, run it directly from the `BPE\code` directory:

```powershell
python main.py
```

The example trains on a small corpus, encodes new text, displays `<unk>` for an
unseen character, and decodes the resulting tokens.

## Use the tokenizer

```python
from BPE.code import decode, encode_text, train_bpe

corpus = [
    "low lower lowest",
    "newer wider",
    "low low lower",
]

merges, token_vocab = train_bpe(corpus, num_merges=10)
tokens = encode_text("low lower zoo", merges, token_vocab)

print(tokens)
print(decode(tokens))
```

## Save and load a model

```python
from BPE.code import load_model, save_model

save_model("bpe_model.json", merges, token_vocab, num_merges=10)

loaded_merges, loaded_vocab, merge_count = load_model("bpe_model.json")
```

The JSON file stores the ordered merge rules, token vocabulary, special tokens,
and configured number of merges. Loading it allows text to be tokenized without
training the model again.

## Module responsibilities

1. `build_vocab()` splits words into characters and adds `</w>`.
2. `get_pair_counts()` counts adjacent token pairs using word frequencies.
3. `merge_pair()` replaces the selected pair throughout the vocabulary.
4. `train_bpe()` repeats pair counting and merging while recording each rule.
5. `encode_text()` applies learned rules to new input in training order.
6. `decode()` joins tokens and converts `</w>` markers back into spaces.
7. Unknown characters are represented by `<unk>`.

## Current scope

This implementation is intentionally straightforward for learning. It scans the
vocabulary during every merge. For a very large corpus, use chunked input,
incremental pair-count updates, and a priority queue, or move to an optimized BPE
library after validating the algorithm with this version.

## Download WikiText-103 Raw

WikiText-103 Raw is the suggested larger English dataset for this project. The
download script stores it inside `BPE/dataset` and downloads only the raw
WikiText-103 configuration.

Install the downloader dependency:

```powershell
python -m pip install huggingface_hub
```

Run this command from the project root:

```powershell
python BPE/download_dataset.py
```

The download starts only when you execute the command above.
