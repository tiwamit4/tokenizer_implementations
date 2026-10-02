"""Train official SentencePiece Unigram or BPE models from text iterators."""

from itertools import chain
from pathlib import Path

import sentencepiece as spm


SUPPORTED_MODEL_TYPES = {"unigram", "bpe"}


def train_sentencepiece(sentences, model_prefix, model_type, vocab_size,
                        character_coverage=1.0, byte_fallback=False,
                        hard_vocab_limit=False):
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError("model_type must be 'unigram' or 'bpe'")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if not 0 < character_coverage <= 1:
        raise ValueError("character_coverage must be in (0, 1]")
    prefix = Path(model_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    model_path = prefix.with_suffix(".model")
    vocab_path = prefix.with_suffix(".vocab")
    if model_path.exists() or vocab_path.exists():
        raise FileExistsError("SentencePiece output exists; change MODEL_PREFIX")
    iterator = iter(sentences)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise ValueError("Training corpus is empty") from error
    spm.SentencePieceTrainer.train(
        sentence_iterator=chain([first], iterator),
        model_prefix=str(prefix),
        model_type=model_type,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        byte_fallback=byte_fallback,
        hard_vocab_limit=hard_vocab_limit,
        shuffle_input_sentence=False,
        num_threads=1,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        minloglevel=2,
    )
    return model_path, vocab_path
