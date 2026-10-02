"""Small interface around the official SentencePieceProcessor."""

from pathlib import Path

import sentencepiece as spm

from .training import train_sentencepiece


class SentencePieceTokenizer:
    def __init__(self, model_file):
        path = Path(model_file)
        if not path.is_file():
            raise FileNotFoundError(path)
        self.model_file = path
        self.processor = spm.SentencePieceProcessor(model_file=str(path))

    @classmethod
    def train(cls, sentences, model_prefix, model_type, vocab_size,
              character_coverage=1.0, byte_fallback=False,
              hard_vocab_limit=False):
        model_path, _ = train_sentencepiece(
            sentences, model_prefix, model_type, vocab_size,
            character_coverage, byte_fallback, hard_vocab_limit)
        return cls(model_path)

    @property
    def vocabulary_size(self):
        return self.processor.vocab_size()

    def tokenize(self, text):
        return self.processor.encode(text, out_type=str)

    def encode(self, text, add_bos=False, add_eos=False):
        return self.processor.encode(
            text, out_type=int, add_bos=add_bos, add_eos=add_eos)

    def decode(self, values):
        return self.processor.decode(values)

    def sample(self, text, alpha=0.1, nbest_size=-1):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        return self.processor.encode(
            text, out_type=str, enable_sampling=True,
            alpha=alpha, nbest_size=nbest_size)

    def piece_to_id(self, piece):
        return self.processor.piece_to_id(piece)

    def id_to_piece(self, index):
        if not isinstance(index, int) or not 0 <= index < self.vocabulary_size:
            raise ValueError(f"Invalid piece ID: {index!r}")
        return self.processor.id_to_piece(index)

    def pieces(self):
        return [self.processor.id_to_piece(i)
                for i in range(self.vocabulary_size)]
