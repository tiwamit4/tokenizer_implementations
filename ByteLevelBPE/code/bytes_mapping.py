"""Reversible mapping between all 256 byte values and visible Unicode symbols."""


def bytes_to_unicode():
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    code_points = byte_values.copy()
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            code_points.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, map(chr, code_points)))


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {symbol: value for value, symbol in BYTE_ENCODER.items()}

assert len(BYTE_ENCODER) == len(BYTE_DECODER) == 256
