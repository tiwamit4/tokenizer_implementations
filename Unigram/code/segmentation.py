"""Viterbi segmentation for a Unigram token-cost model."""

from math import inf


def viterbi(text, costs):
    best = [inf] * (len(text) + 1)
    previous = [None] * (len(text) + 1)
    best[0] = 0.0
    max_length = max(map(len, costs), default=0)
    for start in range(len(text)):
        if best[start] == inf:
            continue
        for end in range(start + 1, min(len(text), start + max_length) + 1):
            piece = text[start:end]
            if piece not in costs:
                continue
            candidate = best[start] + costs[piece]
            # Strict comparison keeps shorter-end/earlier paths deterministic.
            if candidate < best[end]:
                best[end] = candidate
                previous[end] = (start, piece)
    if previous[-1] is None:
        return None, inf
    pieces = []
    position = len(text)
    while position:
        start, piece = previous[position]
        pieces.append(piece)
        position = start
    return list(reversed(pieces)), best[-1]
