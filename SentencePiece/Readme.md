# SentencePiece tokenizer

This folder shows how to train and use SentencePiece with its official Python package. SentencePiece is not one separate tokenization algorithm: it can train either a **Unigram** model or a **BPE** model directly from raw text.

## Flow

```text
Raw text sentences
        |
        v
SentencePiece normalization
        |
        v
Spaces represented by the `▁` marker
        |
        v
Choose a model type
   +---------+---------+
   |                   |
   v                   v
Unigram              BPE
score/prune       learn merges
   |                   |
   +---------+---------+
             |
             v
     model.model + model.vocab
             |
             v
 SentencePieceProcessor
             |
      +------+------+
      v             v
    pieces        token IDs
      \             /
       +-----+-----+
             v
           decode
```

The `▁` character marks a space or the beginning of a word. For example, `hello world` may become pieces such as `▁hello` and `▁world`. This lets decoding restore spaces without separate whitespace rules.

## Files

```text
SentencePiece/
├── code/
│   ├── config.py       # settings used by main.py
│   ├── data.py         # streams text from Parquet files
│   ├── training.py     # trains Unigram or BPE models
│   ├── tokenizer.py    # encode, decode, sampling, and lookup methods
│   └── main.py         # runnable example
├── models/             # generated .model and .vocab files
├── sentencepiece_step_by_step.ipynb
└── requirements.txt
```

## Setup

From the repository root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If your shared environment is at `D:\project\.venv`, activate it with:

```powershell
D:\project\.venv\Scripts\Activate.ps1
```

## Run

Open `SentencePiece/code/config.py` and choose:

```python
MODEL_TYPE = "unigram"  # or "bpe"
VOCAB_SIZE = 64
```

Then run from `SentencePiece/code`:

```powershell
python main.py
```

The program uses the small example corpus in `config.py` unless `USE_DATASET` is set to `True`. With dataset mode enabled, it streams text from Parquet files instead of loading the whole dataset into memory.

SentencePiece refuses to overwrite an existing model. Change `MODEL_PREFIX` or remove the old generated model deliberately before training again.

## Model types

- `unigram` begins with many candidate pieces and keeps pieces that best explain the corpus.
- `bpe` repeatedly combines frequent neighboring symbols.
- Both produce the same SentencePiece model format and use the same processor API.
- Unigram also supports subword sampling, which can produce different valid segmentations during training.

## Notebook

Open `sentencepiece_step_by_step.ipynb` and select **Python (.venv - tokenizers)**. It trains temporary Unigram and BPE models, compares their pieces, converts pieces to IDs, decodes them, and demonstrates Unigram sampling.

## Official reference

- [SentencePiece repository](https://github.com/google/sentencepiece)
- [SentencePiece Python examples](https://github.com/google/sentencepiece/blob/master/python/README.md)
