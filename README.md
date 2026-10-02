# Tokenizer Implementations

Implementations and step-by-step examples for learning how common subword tokenizers are trained, encoded, decoded, and saved.

## Project status

| Tokenizer | Implementation | Notebook | Tests |
|---|---:|---:|---:|
| BPE | Complete | Available | Available |
| WordPiece | Complete | Available | Available |
| Unigram | Complete | Available | Available |
| Byte-Level BPE | Complete | Available | Available |
| SentencePiece | Complete | Available | Available |

BPE, WordPiece, Unigram, and Byte-Level BPE contain study implementations built in this repository. SentencePiece uses the official `sentencepiece` Python package and demonstrates both its Unigram and BPE model types.

## Project structure

```text
tokenizer_implementations/
|-- BPE/
|   |-- code/
|   |-- example/
|   `-- Readme.md
|-- WordPiece/
|   |-- code/
|   |-- wordpiece_step_by_step.ipynb
|   `-- README.md
|-- Unigram/
|   |-- code/
|   |-- unigram_step_by_step.ipynb
|   `-- README.md
|-- ByteLevelBPE/
|   |-- code/
|   |-- byte_level_bpe_step_by_step.ipynb
|   `-- README.md
|-- SentencePiece/
|   |-- code/
|   |-- sentencepiece_step_by_step.ipynb
|   `-- Readme.md
|-- datasets/
|-- tests/
|-- requirements.txt
`-- README.md
```

Each tokenizer keeps its implementation, configuration, examples, and documentation in its own directory. Shared datasets belong in `datasets/`, while project-level tests belong in `tests/`.

## Tokenizer comparison

| Tokenizer | Main idea | Unknown text | Typical feature |
|---|---|---|---|
| BPE | Repeatedly merges the most frequent adjacent pair | Uses an unknown token unless the base vocabulary covers the input | Simple merge-based training |
| WordPiece | Chooses merges using a frequency-based score | A word that cannot be segmented becomes `[UNK]` | Continuation pieces commonly use `##` |
| Unigram | Selects the best segmentation from a scored piece vocabulary | Falls back to an unknown token | Supports multiple possible segmentations |
| Byte-Level BPE | Applies BPE to reversible byte symbols | Every UTF-8 input can be represented | No unknown character is required |
| SentencePiece | Trains Unigram or BPE directly from raw sentences | Controlled by its model vocabulary and settings | Represents spaces with `▁` |

## Environment setup

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the shared environment already exists at `D:\project\.venv`:

```powershell
D:\project\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The notebooks use the kernel named **Python (.venv - tokenizers)**.

## Run an implementation

Settings such as the corpus, vocabulary size, model path, and training limits are kept in each tokenizer's `code/config.py`.

Run a tokenizer from its `code` directory:

```powershell
cd BPE\code
python main.py
```

Replace `BPE` with `WordPiece`, `Unigram`, `ByteLevelBPE`, or `SentencePiece` to run another implementation:

```powershell
cd D:\project\tokenizer_implementations\SentencePiece\code
python main.py
```

Read the tokenizer-specific README before changing its dataset or model settings.

## Run the tests

From the repository root:

```powershell
python -m unittest discover -s tests -v
```

The test suite checks training behavior, encoding and decoding, model persistence, invalid settings, dataset streaming, and deterministic results where applicable.

## Learning order

A useful order is:

1. BPE
2. WordPiece
3. Unigram
4. Byte-Level BPE
5. SentencePiece

This order starts with basic pair merging, introduces alternative vocabulary-learning strategies, moves to byte-level coverage, and finishes with a production library that supports both Unigram and BPE.

## Large datasets

Keep large corpus files under `datasets/`; they are excluded from Git. The data modules stream records where supported so the entire dataset does not need to be loaded into memory. Configure dataset paths and text limits in the tokenizer's `config.py`.
