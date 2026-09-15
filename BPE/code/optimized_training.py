"""BPE training with incremental pair counts and a priority queue."""

from collections import Counter, defaultdict
from collections.abc import Iterable
import heapq

from .config import DEFAULT_NUM_MERGES, END_OF_WORD, UNK_TOKEN
from .vocabulary import build_vocab, merge_pair


def _pairs(symbols):
    return Counter(zip(symbols, symbols[1:]))


def train_bpe_optimized(corpus: Iterable[str], num_merges=DEFAULT_NUM_MERGES):
    """Read texts once, then update only words affected by each selected pair.

    Memory scales with unique words and their pair memberships, not document
    count. Very diverse corpora can still require considerable RAM.
    """
    if num_merges < 0:
        raise ValueError("num_merges must be non-negative")
    vocabulary = build_vocab(corpus)
    if not vocabulary:
        raise ValueError("The training corpus is empty")
    words = list(vocabulary)
    frequencies = list(vocabulary.values())
    del vocabulary
    tokens = {END_OF_WORD, UNK_TOKEN}
    counts = Counter()
    members = defaultdict(set)
    for word_id, symbols in enumerate(words):
        tokens.update(symbols)
        for pair, occurrences in _pairs(symbols).items():
            counts[pair] += occurrences * frequencies[word_id]
            members[pair].add(word_id)
    heap = [(-count, pair) for pair, count in counts.items()]
    heapq.heapify(heap)
    merges = []
    for _ in range(num_merges):
        # Discard stale heap entries after incremental count changes.
        while heap:
            negative_count, best = heapq.heappop(heap)
            if counts[best] > 0 and -negative_count == counts[best]:
                break
        else:
            break
        changed = set()
        for word_id in tuple(members[best]):
            old_symbols = words[word_id]
            old_pairs = _pairs(old_symbols)
            new_symbols = next(iter(merge_pair(best, Counter({old_symbols: 1}))))
            new_pairs = _pairs(new_symbols)
            for pair in old_pairs.keys() | new_pairs.keys():
                counts[pair] += (
                    new_pairs[pair] - old_pairs[pair]
                ) * frequencies[word_id]
                if new_pairs[pair]:
                    members[pair].add(word_id)
                else:
                    members[pair].discard(word_id)
                    if not members[pair]:
                        del members[pair]
                changed.add(pair)
            words[word_id] = new_symbols
        for pair in changed:
            if counts[pair] > 0:
                heapq.heappush(heap, (-counts[pair], pair))
        # Bound stale heap growth without rebuilding counts.
        if len(heap) > max(1, len(members)) * 4:
            heap = [(-counts[pair], pair) for pair in members if counts[pair] > 0]
            heapq.heapify(heap)
        merges.append(best)
        tokens.add("".join(best))
    return merges, tokens
