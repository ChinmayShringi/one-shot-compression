"""From-scratch entropy coding for NEXUS codec.

Implements:
1. Huffman coding - tree construction, canonical encoding, serialization
2. Range coder (arithmetic coding variant) - for adaptive contexts

All built from pure mathematics. No compression library dependencies.
"""

from __future__ import annotations

import heapq
from collections import Counter
from typing import Sequence

from .bitstream import BitWriter, BitReader


# ---------------------------------------------------------------------------
# Huffman Coding - built from scratch
# ---------------------------------------------------------------------------

class HuffmanNode:
    """Node in a Huffman tree."""
    __slots__ = ("freq", "symbol", "left", "right")

    def __init__(self, freq: int, symbol: int = -1,
                 left: HuffmanNode | None = None,
                 right: HuffmanNode | None = None) -> None:
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other: HuffmanNode) -> bool:
        return self.freq < other.freq

    @property
    def is_leaf(self) -> bool:
        return self.symbol >= 0


def build_huffman_tree(frequencies: dict[int, int]) -> HuffmanNode | None:
    """Build Huffman tree from symbol frequencies using a min-heap.

    The algorithm (Huffman 1952):
    1. Create a leaf node for each symbol with its frequency
    2. While more than one node remains:
       a. Extract the two nodes with lowest frequency
       b. Create a parent node with their combined frequency
       c. Insert the parent back into the heap
    3. The last remaining node is the root
    """
    if not frequencies:
        return None
    if len(frequencies) == 1:
        sym = next(iter(frequencies))
        return HuffmanNode(frequencies[sym], symbol=sym)

    heap: list[HuffmanNode] = []
    for sym, freq in frequencies.items():
        heapq.heappush(heap, HuffmanNode(freq, symbol=sym))

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, parent)

    return heap[0]


def extract_code_lengths(root: HuffmanNode | None) -> dict[int, int]:
    """Extract code length for each symbol by traversing the tree."""
    if root is None:
        return {}
    lengths: dict[int, int] = {}

    def _walk(node: HuffmanNode, depth: int) -> None:
        if node.is_leaf:
            lengths[node.symbol] = max(depth, 1)  # min length 1
            return
        if node.left:
            _walk(node.left, depth + 1)
        if node.right:
            _walk(node.right, depth + 1)

    _walk(root, 0)
    return lengths


def build_canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    """Build canonical Huffman codes from code lengths.

    Canonical Huffman (Schwartz & Kallick 1964):
    1. Sort symbols by code length (then by symbol value for ties)
    2. Assign codes sequentially, shifting left when length increases
    3. This allows the codebook to be stored as just the lengths

    Returns: {symbol: (code, length)}
    """
    if not lengths:
        return {}

    # Sort by (length, symbol)
    sorted_syms = sorted(lengths.items(), key=lambda x: (x[1], x[0]))

    codes: dict[int, tuple[int, int]] = {}
    code = 0
    prev_len = sorted_syms[0][1]

    for sym, length in sorted_syms:
        code <<= (length - prev_len)
        codes[sym] = (code, length)
        code += 1
        prev_len = length

    return codes


class HuffmanCodec:
    """Complete Huffman encoder/decoder built from scratch."""

    def __init__(self, frequencies: dict[int, int]) -> None:
        tree = build_huffman_tree(frequencies)
        self._lengths = extract_code_lengths(tree)
        self._codes = build_canonical_codes(self._lengths)
        # Build decode table: {(code, length): symbol}
        self._decode_table = {v: k for k, v in self._codes.items()}
        self._max_length = max(self._lengths.values()) if self._lengths else 0
        self._frequencies = frequencies

    @property
    def code_lengths(self) -> dict[int, int]:
        return self._lengths

    def encode_symbol(self, writer: BitWriter, symbol: int) -> None:
        """Encode a single symbol to the bitstream."""
        code, length = self._codes[symbol]
        # Write MSB first for canonical Huffman
        for i in range(length - 1, -1, -1):
            writer.write_bit((code >> i) & 1)

    def decode_symbol(self, reader: BitReader) -> int:
        """Decode a single symbol from the bitstream."""
        code = 0
        for length in range(1, self._max_length + 1):
            code = (code << 1) | reader.read_bit()
            key = (code, length)
            if key in self._decode_table:
                return self._decode_table[key]
        raise ValueError("Invalid Huffman code")

    def serialize_table(self, writer: BitWriter) -> None:
        """Write the codebook to the bitstream.

        Format: max_symbol(varint) + lengths as run-length encoded bytes.
        Only the lengths are needed to reconstruct canonical codes.
        """
        if not self._lengths:
            writer.write_varint(0)
            return
        max_sym = max(self._lengths.keys())
        writer.write_varint(max_sym + 1)
        # Write length for each symbol 0..max_sym (0 = symbol not present)
        for sym in range(max_sym + 1):
            length = self._lengths.get(sym, 0)
            writer.write_byte(min(length, 255))

    @classmethod
    def deserialize_table(cls, reader: BitReader) -> HuffmanCodec:
        """Read codebook from bitstream and reconstruct codec."""
        n_symbols = reader.read_varint()
        if n_symbols == 0:
            return cls({})
        lengths = {}
        for sym in range(n_symbols):
            length = reader.read_byte()
            if length > 0:
                lengths[sym] = length
        # Reconstruct frequencies (relative ordering preserved by length)
        # We don't need actual frequencies for decoding, just the codes
        dummy_freqs = {}
        for sym, length in lengths.items():
            dummy_freqs[sym] = 1 << (32 - length)  # longer code = lower freq
        codec = cls.__new__(cls)
        codec._lengths = lengths
        codec._codes = build_canonical_codes(lengths)
        codec._decode_table = {v: k for k, v in codec._codes.items()}
        codec._max_length = max(lengths.values()) if lengths else 0
        codec._frequencies = dummy_freqs
        return codec


def compute_frequencies(data: Sequence[int]) -> dict[int, int]:
    """Count symbol frequencies in data."""
    return dict(Counter(data))


# ---------------------------------------------------------------------------
# Range Coder - arithmetic coding variant, built from scratch
# ---------------------------------------------------------------------------

class RangeEncoder:
    """Range encoder (arithmetic coding) from first principles.

    Based on the Schindler range coder model:
    - Maintains a range [low, low + range)
    - Narrows range based on symbol probability
    - Outputs bytes when the top byte of low is determined

    This is mathematically equivalent to arithmetic coding but uses
    byte-level renormalization instead of bit-level, making it faster.
    """

    TOP = 1 << 24
    BOTTOM = 1 << 16

    def __init__(self) -> None:
        self._low = 0
        self._range = 0xFFFFFFFF
        self._buffer: list[int] = []

    def encode(self, cum_freq: int, freq: int, total: int) -> None:
        """Encode a symbol with cumulative frequency range [cum_freq, cum_freq+freq).

        Parameters:
            cum_freq: cumulative frequency of symbols before this one
            freq: frequency of this symbol
            total: total frequency of all symbols
        """
        self._range //= total
        self._low += cum_freq * self._range
        self._range *= freq
        # Renormalize: output bytes when top bits are determined
        while self._range < self.BOTTOM:
            if self._low < 0xFF000000:
                self._buffer.append((self._low >> 24) & 0xFF)
                self._low <<= 8
                self._low &= 0xFFFFFFFF
                self._range <<= 8
            elif (self._low & 0xFF000000) == 0xFF000000:
                self._buffer.append(0xFF)
                self._low <<= 8
                self._low &= 0xFFFFFFFF
                self._range <<= 8
            else:
                # Carry propagation
                self._buffer.append((self._low >> 24) & 0xFF)
                self._low <<= 8
                self._low &= 0xFFFFFFFF
                self._range <<= 8

    def finish(self) -> bytes:
        """Flush remaining state and return encoded bytes."""
        for _ in range(4):
            self._buffer.append((self._low >> 24) & 0xFF)
            self._low <<= 8
            self._low &= 0xFFFFFFFF
        return bytes(self._buffer)


class RangeDecoder:
    """Range decoder (arithmetic coding) from first principles."""

    TOP = 1 << 24
    BOTTOM = 1 << 16

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0
        self._low = 0
        self._range = 0xFFFFFFFF
        self._code = 0
        # Initialize code from first 4 bytes
        for _ in range(4):
            self._code = (self._code << 8) | self._next_byte()

    def _next_byte(self) -> int:
        if self._pos < len(self._data):
            b = self._data[self._pos]
            self._pos += 1
            return b
        return 0

    def get_freq(self, total: int) -> int:
        """Get the cumulative frequency for the current symbol."""
        self._range //= total
        return (self._code - self._low) // self._range

    def decode(self, cum_freq: int, freq: int, total: int) -> None:
        """Consume a decoded symbol and advance state."""
        self._range //= total
        self._low += cum_freq * self._range
        self._range *= freq
        while self._range < self.BOTTOM:
            self._code = ((self._code << 8) | self._next_byte()) & 0xFFFFFFFF
            self._low <<= 8
            self._low &= 0xFFFFFFFF
            self._range <<= 8
