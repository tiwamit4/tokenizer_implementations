"""Viterbi tokenization, IDs, decoding, and model persistence."""

from .config import SPACE_MARKER, SPECIAL_TOKENS, UNK_TOKEN
from .model_io import load_model, save_model
from .preprocessing import pre_tokenize, restore_spaces
from .segmentation import viterbi
from .training import train_unigram


class UnigramTokenizer:
    def __init__(self, piece_costs, special_tokens=SPECIAL_TOKENS,
                 marker=SPACE_MARKER, unk_token=UNK_TOKEN):
        self.special_tokens = tuple(dict.fromkeys(special_tokens))
        if unk_token not in self.special_tokens:
            raise ValueError("Unknown token must be a special token")
        if not marker:
            raise ValueError("Space marker must not be empty")
        self.marker = marker
        self.unk_token = unk_token
        ordered_pieces = sorted(piece_costs, key=lambda piece: (piece_costs[piece], piece))
        self.vocabulary = list(self.special_tokens) + ordered_pieces
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("Vocabulary entries must be unique")
        self.scores = [None] * len(self.special_tokens) + [piece_costs[p] for p in ordered_pieces]
        self.piece_costs = dict(piece_costs)
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}

    @classmethod
    def train(cls, corpus, target_size, initial_size, max_piece_length=12,
              prune_fraction=0.2, special_tokens=SPECIAL_TOKENS,
              marker=SPACE_MARKER, unk_token=UNK_TOKEN):
        costs = train_unigram(corpus, target_size, initial_size, max_piece_length,
                              prune_fraction, marker, special_tokens)
        return cls(costs, special_tokens, marker, unk_token)

    def tokenize(self, text):
        output = []
        for word in pre_tokenize(text, self.marker):
            pieces, _ = viterbi(word, self.piece_costs)
            output.extend(pieces if pieces is not None else [self.unk_token])
        return output

    def encode(self, text):
        return [self.token_to_id[token] for token in self.tokenize(text)]

    def decode_tokens(self, tokens):
        marked = []
        for token in tokens:
            if token in self.special_tokens:
                marked.append(self.marker + token)
            else:
                marked.append(token)
        return restore_spaces(marked, self.marker)

    def decode(self, ids):
        tokens = []
        for index in ids:
            if not isinstance(index, int) or not 0 <= index < len(self.vocabulary):
                raise ValueError(f"Invalid token ID: {index!r}")
            tokens.append(self.vocabulary[index])
        return self.decode_tokens(tokens)

    def save(self, filename):
        save_model(filename, self.vocabulary, self.scores,
                   self.special_tokens, self.marker, self.unk_token)

    @classmethod
    def load(cls, filename):
        data = load_model(filename)
        special_count = len(data["special_tokens"])
        costs = dict(zip(data["vocabulary"][special_count:], data["scores"][special_count:]))
        tokenizer = cls(costs, data["special_tokens"],
                        data["space_marker"], data["unk_token"])
        if tokenizer.vocabulary != data["vocabulary"]:
            raise ValueError("Saved vocabulary ordering is inconsistent")
        return tokenizer
