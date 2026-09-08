# Tokenizer Implementations

From-scratch implementations and learning examples for multiple text
tokenization algorithms.

## Project structure

```text
tokenizer_implementations/
|-- BPE/
|-- WordPiece/
|-- Unigram/
|-- SentencePiece/
|-- ByteLevelBPE/
|-- datasets/
|-- tests/
|-- requirements.txt
`-- README.md
```

Each tokenizer has its own directory so its implementation, examples, models,
and documentation can be developed independently. Shared datasets belong in
`datasets`, while project-level tests belong in `tests`.

## Environment setup

From the project root, install all current dependencies with:

```powershell
python -m pip install -r requirements.txt
```

See each tokenizer directory for implementation-specific instructions.
