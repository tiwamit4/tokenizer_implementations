# WordPiece from scratch

This tokenizer builds a vocabulary from text, splits words into subword pieces,
and maps those pieces to token IDs. It uses `##` for continuation pieces and
`[UNK]` when a word cannot be split. Models are saved as JSON.

## Implementation flow

### Training

```text
Read settings from code/config.py
    |
    v
Choose demo corpus or local dataset sample
    |
    v
Normalize text and split words/punctuation
    |
    v
Count words and initialize character pieces
Example: playing -> p ##l ##a ##y ##i ##n ##g
    |
    v
Create vocabulary: special tokens + base alphabet
    |
    v
Count pieces and adjacent pairs using word frequencies
    |
    v
Score pairs: pair_count / (first_count * second_count)
    |
    v
Select highest-scoring pair and merge it in every affected word
Example: play + ##ing -> playing
    |
    v
Add merged piece to vocabulary; retain earlier pieces
    |
    v
Target vocabulary size reached, or no pairs remain?
    |                                      |
    No: repeat counting and scoring        Yes
                                            |
                                            v
Save ordered vocabulary and inference settings to JSON
```

### Encoding and decoding

```text
New text + trained/loaded vocabulary
    |
    v
Apply the same normalization and pre-tokenization
    |
    v
For each word, find longest matching piece from current position
First piece: no prefix; later pieces: ## prefix
    |
    +-- Match found -> advance and repeat until word is complete
    |
    +-- No match, or word exceeds character limit
            -> discard partial pieces and return one [UNK]
    |
    v
Combine sentence pieces and map them to vocabulary IDs
    |
    v
Decode IDs to pieces; join ## continuations to preceding words
```

Example with a vocabulary containing `play` and `##ing`:

```text
playing -> [play, ##ing] -> token IDs -> playing
playz   -> [[UNK]]      -> unknown-token ID -> [UNK]
```

Encoding uses only the final vocabulary, not training merge rules. Decoding
cannot reconstruct unknown text or original whitespace/punctuation spacing.

## Step-by-step notebook

Open [wordpiece_step_by_step.ipynb](wordpiece_step_by_step.ipynb) and select
your `.venv` kernel. Run the cells in order to follow a small corpus through
training, tokenization, and saving a model. Each step includes an example.

## Run

From `WordPiece/code`, using your activated virtual environment:

```powershell
python main.py
```

Or from the project root:

```powershell
python -m WordPiece.code.main
```

Change the settings in `code/config.py` before running the script.
The default run trains on a small built-in corpus and saves
`WordPiece/models/wordpiece_model.json`. Existing models are never overwritten;
choose a different `MODEL_PATH` for another run.

To use the already downloaded WikiText training files, set `USE_DATASET = True`.
The default sample limit is 1,000 nonempty rows. Install `requirements.txt` for
Parquet support. The training input is batched, but all unique words remain in
memory and pair scores are recomputed after every merge. This trainer is not
intended for full-corpus production training. No downloads are performed.

## Files

- `config.py`: special tokens, training inputs, limits, and model path.
- `preprocessing.py`: Unicode word and punctuation splitting.
- `training.py`: initial alphabet, pair scores, and vocabulary growth.
- `tokenizer.py`: longest-match splitting, IDs, decoding, and model interface.
- `model_io.py`: JSON storage, including inference settings.
- `data.py`: optional local Parquet batch reader.
- `main.py`: configuration-driven training and demonstration.

## Use in Python

```python
from WordPiece.code import WordPieceTokenizer

tokenizer = WordPieceTokenizer.train(
    ["play playing played", "walk walking"], vocab_size=100
)
print(tokenizer.tokenize("playing"))
ids = tokenizer.encode("playing walking")
print(ids)
print(tokenizer.decode(ids))
tokenizer.save("wordpiece_model.json")
loaded = WordPieceTokenizer.load("wordpiece_model.json")
```

## Algorithm and limitations

Training starts with initial characters and prefixed continuation characters.
It selects the highest score:

```text
pair frequency / (first-piece frequency * second-piece frequency)
```

Ties use lexical ordering for reproducibility. Encoding uses only the vocabulary,
not the learned merge order. If any segment cannot be found, the entire word
becomes `[UNK]`.

Training uses the pair-scoring method described in the
[Hugging Face WordPiece lesson](https://huggingface.co/learn/llm-course/en/chapter6/6),
which approximates WordPiece training rather than reproducing Google's original
trainer. Longest-match splitting follows
[BERT's tokenizer](https://github.com/google-research/bert/blob/master/tokenization.py).

Preprocessing is simpler than BERT BasicTokenizer: Unicode words/punctuation,
optional lowercasing, no accent stripping or special Chinese-character handling.
Decoding produces canonical spaces, not original whitespace/punctuation spacing.
Special tokens have reserved IDs but are not automatically inserted into input.

Run tests from the project root:

```powershell
python -m unittest discover -s tests -v
```
