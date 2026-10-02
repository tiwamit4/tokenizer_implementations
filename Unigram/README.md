# Unigram tokenizer from scratch

This implementation starts with a large set of substring candidates and removes
the pieces that contribute least to the corpus likelihood. Encoding uses Viterbi
dynamic programming to select the lowest-cost segmentation.

## Flow

```text
Training text
  -> mark word starts with ▁
  -> count words
  -> collect required characters and frequent substrings
  -> convert substring frequencies to negative-log costs
  -> calculate current corpus loss with Viterbi
  -> temporarily remove each non-required piece
  -> measure the resulting loss increase
  -> prune the lowest-impact pieces in a batch
  -> repeat until target vocabulary size
  -> save pieces, scores, special tokens, and space marker

New text
  -> apply the same ▁ word-start marking
  -> build the segmentation lattice
  -> Viterbi selects the minimum-cost path
  -> map pieces to IDs
  -> join pieces and restore spaces during decoding
```

Single-character pieces are never pruned, so characters seen during training
remain tokenizable. A word containing an unseen character becomes `<unk>`.
Whitespace is normalized to single spaces during decoding.

## Run

For a cell-by-cell walkthrough, open
[unigram_step_by_step.ipynb](unigram_step_by_step.ipynb) with the
`Python (.venv - tokenizers)` kernel and run the cells in order.

From `Unigram/code`:

```powershell
python main.py
```

Or from the project root:

```powershell
python -m Unigram.code.main
```

All settings are in `code/config.py`. The default run uses a small built-in
corpus. Existing models are not overwritten. Set `USE_DATASET = True` to use a
limited sample from the local WikiText files; no dataset is downloaded.
For another corpus, choose vocabulary sizes large enough to retain its complete
character set.

## Files

- `config.py`: vocabulary sizes, pruning settings, inputs, and output path.
- `preprocessing.py`: word-start marking and space restoration.
- `segmentation.py`: Viterbi dynamic programming.
- `training.py`: candidate creation, corpus loss, and vocabulary pruning.
- `tokenizer.py`: tokenization, IDs, decoding, and model interface.
- `model_io.py`: JSON persistence.
- `data.py`: optional local Parquet reader.
- `main.py`: configuration-driven entry point.

## Python example

```python
from Unigram.code import UnigramTokenizer

corpus = ["hug hug hug", "pug pug", "unhug"]
tokenizer = UnigramTokenizer.train(
    corpus,
    target_size=24,
    initial_size=50,
)
tokens = tokenizer.tokenize("unhug pug")
ids = tokenizer.encode("unhug pug")
print(tokens)
print(ids)
print(tokenizer.decode(ids))
tokenizer.save("unigram_model.json")
```

## Scope

This version directly recalculates corpus loss after removing each candidate,
which makes the algorithm easy to inspect but expensive on large vocabularies.
SentencePiece uses substantially more efficient training machinery. Start with
the built-in corpus before enabling a dataset sample.

The training direction, loss-based pruning, protected characters, and Viterbi
segmentation follow the explanation in the
[Hugging Face Unigram lesson](https://huggingface.co/learn/llm-course/en/chapter6/7).
The broader Unigram and subword-regularization method is described in the
[SentencePiece research paper](https://arxiv.org/abs/1804.10959).

Run all tests from the project root:

```powershell
python -m unittest discover -s tests -v
```
