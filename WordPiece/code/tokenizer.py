"""Greedy longest-match tokenization and vocabulary-ID encoding."""

from .config import CONTINUATION_PREFIX, LOWERCASE, MAX_INPUT_CHARS_PER_WORD, UNK_TOKEN
from .model_io import load_model, save_model
from .preprocessing import pre_tokenize
from .training import train_wordpiece


class WordPieceTokenizer:
    def __init__(self, vocabulary, lowercase=LOWERCASE, prefix=CONTINUATION_PREFIX,
                 unk_token=UNK_TOKEN, max_chars=MAX_INPUT_CHARS_PER_WORD):
        self.vocabulary = list(vocabulary)
        if not prefix or max_chars <= 0:
            raise ValueError("Prefix must be nonempty and max_chars must be positive")
        if len(set(self.vocabulary)) != len(self.vocabulary) or unk_token not in self.vocabulary:
            raise ValueError("Vocabulary must be unique and include the unknown token")
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}
        self.lowercase = lowercase
        self.prefix = prefix
        self.unk_token = unk_token
        self.max_chars = max_chars

    @classmethod
    def train(cls, corpus, vocab_size, lowercase=LOWERCASE):
        vocabulary = train_wordpiece(corpus, vocab_size, lowercase)
        return cls(vocabulary, lowercase=lowercase)

    def tokenize_word(self, word):
        if self.lowercase:
            word = word.lower()
        if len(word) > self.max_chars:
            return [self.unk_token]
        pieces = []
        start = 0
        while start < len(word):
            end = len(word)
            found = None
            while end > start:
                candidate = word[start:end]
                if start:
                    candidate = self.prefix + candidate
                if candidate in self.token_to_id:
                    found = candidate
                    break
                end -= 1
            if found is None:
                return [self.unk_token]  # Discard partial pieces for this word.
            pieces.append(found)
            start = end
        return pieces

    def tokenize(self, text):
        return [piece for word in pre_tokenize(text, self.lowercase)
                for piece in self.tokenize_word(word)]

    def encode(self, text):
        return [self.token_to_id[token] for token in self.tokenize(text)]

    def decode_tokens(self, tokens):
        words = []
        for token in tokens:
            if token.startswith(self.prefix):
                if not words:
                    raise ValueError("Continuation token without a preceding word")
                words[-1] += token[len(self.prefix):]
            else:
                words.append(token)
        # Canonical spaces; original whitespace/punctuation spacing is not retained.
        return " ".join(words)

    def decode(self, ids):
        if any(not isinstance(index, int) or index < 0 or index >= len(self.vocabulary)
               for index in ids):
            raise ValueError("Invalid token ID")
        return self.decode_tokens([self.vocabulary[index] for index in ids])

    def save(self, filename):
        save_model(filename, self.vocabulary, self.lowercase,
                   self.prefix, self.unk_token, self.max_chars)

    @classmethod
    def load(cls, filename):
        model = load_model(filename)
        return cls(model["vocabulary"], model["lowercase"], model["prefix"],
                   model["unk_token"], model["max_chars"])
