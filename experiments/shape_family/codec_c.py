"""Codec C: one cycle of B. Encode, decode to pixels, encode again.

Reports whether the stored description is byte-identical. This is one cycle,
not a claim of indefinite reversibility.
"""

from __future__ import annotations

import codec_b


def encode_cycle(px, width, height):
    first, info1 = codec_b.encode_with_breakdown(px, width, height)
    decoded = codec_b.decode(first)
    second, info2 = codec_b.encode_with_breakdown(decoded, width, height)
    return {
        "first": first,
        "second": second,
        "decoded": decoded,
        "byte_identical": first == second,
        "info1": info1,
        "info2": info2,
    }
