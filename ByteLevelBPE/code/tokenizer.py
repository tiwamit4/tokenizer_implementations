"""Encode arbitrary UTF-8 text with learned byte-level BPE merges."""

from .bytes_mapping import BYTE_DECODER, BYTE_ENCODER
from .config import DECODE_ERRORS, SPECIAL_TOKENS
from .model_io import load_model, save_model
from .preprocessing import pre_tokenize
from .training import merge_symbols, train_byte_level_bpe


class ByteLevelBPETokenizer:
    def __init__(self, merges, special_tokens=SPECIAL_TOKENS, errors=DECODE_ERRORS):
        self.merges = [tuple(pair) for pair in merges]
        self.special_tokens = tuple(dict.fromkeys(special_tokens))
        self.errors = errors
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        vocabulary = list(self.special_tokens)
        vocabulary.extend(BYTE_ENCODER[value] for value in range(256))
        known = set(vocabulary)
        for first, second in self.merges:
            merged = first + second
            if merged not in known:
                vocabulary.append(merged)
                known.add(merged)
        self.vocabulary = vocabulary
        self.token_to_id = {token: index for index, token in enumerate(vocabulary)}

    @classmethod
    def train(cls, corpus, vocab_size, special_tokens=SPECIAL_TOKENS,
              errors=DECODE_ERRORS):
        return cls(train_byte_level_bpe(corpus, vocab_size, special_tokens),
                   special_tokens, errors)

    def _apply_merges(self, symbols):
        symbols = tuple(symbols)
        while len(symbols) > 1:
            candidates = {pair for pair in zip(symbols, symbols[1:])
                          if pair in self.merge_ranks}
            if not candidates:
                break
            best = min(candidates, key=self.merge_ranks.__getitem__)
            symbols = merge_symbols(symbols, best)
        return list(symbols)

    def tokenize(self, text):
        output = []
        for chunk in pre_tokenize(text):
            symbols = [BYTE_ENCODER[value] for value in chunk.encode("utf-8")]
            output.extend(self._apply_merges(symbols))
        return output

    def encode(self, text):
        return [self.token_to_id[token] for token in self.tokenize(text)]

    def decode_tokens(self, tokens):
        output = []
        byte_buffer = bytearray()
        for token in tokens:
            if token in self.special_tokens:
                if byte_buffer:
                    output.append(byte_buffer.decode("utf-8", errors=self.errors))
                    byte_buffer.clear()
                output.append(token)
            else:
                try:
                    byte_buffer.extend(BYTE_DECODER[symbol] for symbol in token)
                except KeyError as error:
                    raise ValueError(f"Invalid byte-level token: {token!r}") from error
        if byte_buffer:
            output.append(byte_buffer.decode("utf-8", errors=self.errors))
        return "".join(output)

    def decode(self, ids):
        tokens = []
        for index in ids:
            if not isinstance(index, int) or not 0 <= index < len(self.vocabulary):
                raise ValueError(f"Invalid token ID: {index!r}")
            tokens.append(self.vocabulary[index])
        return self.decode_tokens(tokens)

    def save(self, filename):
        save_model(filename, self.vocabulary, self.merges,
                   self.special_tokens, self.errors)

    @classmethod
    def load(cls, filename):
        data = load_model(filename)
        tokenizer = cls(data["merges"], data["special_tokens"], data["decode_errors"])
        if tokenizer.vocabulary != data["vocabulary"]:
            raise ValueError("Saved vocabulary does not match merge rules")
        return tokenizer
