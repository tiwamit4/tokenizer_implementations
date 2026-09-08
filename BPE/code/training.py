"""BPE model training."""

from collections.abc import Iterable

from .config import DEFAULT_NUM_MERGES, END_OF_WORD, UNK_TOKEN
from .vocabulary import build_vocab, get_pair_counts, merge_pair


def train_bpe(
    corpus: Iterable[str],
    num_merges: int = DEFAULT_NUM_MERGES,
) -> tuple[list[tuple[str, str]], set[str]]:
    """Learn BPE merge rules and return the rules and token vocabulary."""
    if num_merges < 0:
        raise ValueError("num_merges must be non-negative")

    vocab = build_vocab(corpus)
    if not vocab:
        raise ValueError("The training corpus is empty")

    # Keep original characters encodable even when training merges all of them.
    token_vocab = {UNK_TOKEN, END_OF_WORD}
    for word_tokens in vocab:
        token_vocab.update(word_tokens)

    merges = []

    for _ in range(num_merges):
        pair_counts = get_pair_counts(vocab)
        if not pair_counts:
            break

        # Lexical tie-breaking makes repeated training deterministic.
        best_pair = min(
            pair_counts,
            key=lambda pair: (-pair_counts[pair], pair),
        )

        vocab = merge_pair(best_pair, vocab)
        merges.append(best_pair)
        token_vocab.add(best_pair[0] + best_pair[1])

    return merges, token_vocab
