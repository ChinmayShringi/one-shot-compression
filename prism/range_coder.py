"""Adaptive Range Coder - built entirely from scratch.

Implements byte-level renormalization range coding with an adaptive
frequency model. This approaches Shannon entropy, unlike Huffman or
DEFLATE which leave bits on the table.

Architecture:
  RangeEncoder/Decoder: core arithmetic coding engine
  AdaptiveModel: per-symbol frequency tracking with decay
  ContextModel: maps context keys → AdaptiveModels
"""

import struct
from typing import List, Optional


# Range coder constants
TOP = 1 << 24
BOT = 1 << 16
MAX_FREQ = 1 << 14  # Max total frequency before rescale


class AdaptiveModel:
    """Adaptive frequency model for a symbol alphabet.

    Tracks cumulative frequencies for arithmetic coding.
    Uses periodic halving to weight recent symbols more heavily.
    """

    __slots__ = ('size', 'freq', 'total')

    def __init__(self, size: int):
        self.size = size
        self.freq = [1] * size  # Laplace smoothing
        self.total = size

    def get_freq(self, symbol: int):
        """Return (cumulative_low, cumulative_high, total) for symbol."""
        cum = 0
        for i in range(symbol):
            cum += self.freq[i]
        return cum, cum + self.freq[symbol], self.total

    def get_symbol(self, scaled_value: int):
        """Find symbol for a scaled cumulative value. Return (symbol, cum_low, cum_high)."""
        cum = 0
        for i in range(self.size):
            if cum + self.freq[i] > scaled_value:
                return i, cum, cum + self.freq[i]
            cum += self.freq[i]
        # Shouldn't reach here, but return last symbol
        return self.size - 1, cum - self.freq[-1], cum

    def update(self, symbol: int):
        """Update frequency count for symbol."""
        self.freq[symbol] += 1
        self.total += 1
        if self.total >= MAX_FREQ:
            self._rescale()

    def _rescale(self):
        """Halve all frequencies, keeping minimum of 1."""
        self.total = 0
        for i in range(self.size):
            self.freq[i] = max(1, self.freq[i] >> 1)
            self.total += self.freq[i]


class RangeEncoder:
    """Range encoder with byte-level renormalization."""

    def __init__(self):
        self.low: int = 0
        self.range: int = 0xFFFFFFFF
        self.buffer: int = 0
        self.carry_count: int = 0
        self.first_byte: bool = True
        self.output: bytearray = bytearray()

    def encode(self, cum_low: int, cum_high: int, total: int):
        """Encode a symbol with given cumulative frequency range."""
        r = self.range // total
        self.low += cum_low * r
        if cum_high < total:
            self.range = (cum_high - cum_low) * r
        else:
            self.range -= cum_low * r

        # Handle carry and renormalize
        while self.range < TOP:
            if self.low < 0xFF << 24:
                self._output_byte(self.buffer)
                while self.carry_count > 0:
                    self._output_byte(0xFF)
                    self.carry_count -= 1
                self.buffer = (self.low >> 24) & 0xFF
            elif self.low >= 0x100 << 24:
                self._output_byte(self.buffer + 1)
                while self.carry_count > 0:
                    self._output_byte(0x00)
                    self.carry_count -= 1
                self.buffer = (self.low >> 24) & 0xFF
                self.low &= 0xFFFFFFFF  # Keep 32 bits
            else:
                self.carry_count += 1

            self.low = (self.low << 8) & 0xFFFFFFFF
            self.range <<= 8

    def _output_byte(self, byte: int):
        if not self.first_byte:
            self.output.append(byte & 0xFF)
        self.first_byte = False

    def finish(self) -> bytes:
        """Flush remaining state and return compressed bytes."""
        # Output remaining bytes
        self._output_byte(self.buffer + 1)
        while self.carry_count > 0:
            self._output_byte(0x00)
            self.carry_count -= 1
        # Flush low
        self.output.append((self.low >> 24) & 0xFF)
        self.output.append((self.low >> 16) & 0xFF)
        self.output.append((self.low >> 8) & 0xFF)
        self.output.append(self.low & 0xFF)
        return bytes(self.output)


class RangeDecoder:
    """Range decoder with byte-level renormalization."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.low: int = 0
        self.range: int = 0xFFFFFFFF
        self.code: int = 0

        # Initialize code from first 4 bytes
        for _ in range(4):
            self.code = (self.code << 8) | self._read_byte()

    def _read_byte(self) -> int:
        if self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            return b
        return 0

    def get_freq(self, total: int) -> int:
        """Get the scaled cumulative frequency for decoding."""
        self.range //= total
        return (self.code - self.low) // self.range

    def decode(self, cum_low: int, cum_high: int, total: int):
        """Update decoder state after identifying symbol."""
        r = self.range  # Already divided by total in get_freq
        self.low += cum_low * r
        if cum_high < total:
            self.range = (cum_high - cum_low) * r
        else:
            self.range = total * r - cum_low * r  # Remaining range

        # Renormalize
        while self.range < TOP:
            self.code = ((self.code << 8) | self._read_byte()) & 0xFFFFFFFF
            self.low = (self.low << 8) & 0xFFFFFFFF
            self.range <<= 8


def encode_symbols(symbols: List[int], alphabet_size: int) -> bytes:
    """Encode a list of symbols using adaptive range coding."""
    model = AdaptiveModel(alphabet_size)
    enc = RangeEncoder()
    for s in symbols:
        cum_low, cum_high, total = model.get_freq(s)
        enc.encode(cum_low, cum_high, total)
        model.update(s)
    return enc.finish()


def decode_symbols(data: bytes, count: int, alphabet_size: int) -> List[int]:
    """Decode count symbols from range-coded data."""
    model = AdaptiveModel(alphabet_size)
    dec = RangeDecoder(data)
    symbols = []
    for _ in range(count):
        total = model.total
        freq = dec.get_freq(total)
        symbol, cum_low, cum_high = model.get_symbol(freq)
        dec.decode(cum_low, cum_high, total)
        model.update(symbol)
        symbols.append(symbol)
    return symbols


class ContextualEncoder:
    """Range encoder with context-dependent probability models.

    Each context key maps to its own AdaptiveModel, so symbols
    in different contexts get different probability distributions.
    """

    def __init__(self, alphabet_size: int, n_contexts: int = 256):
        self.alphabet_size = alphabet_size
        self.models = {}
        self.enc = RangeEncoder()
        self.n_contexts = n_contexts

    def _get_model(self, ctx: int) -> AdaptiveModel:
        # Limit number of contexts to prevent memory explosion
        ctx = ctx % self.n_contexts
        if ctx not in self.models:
            self.models[ctx] = AdaptiveModel(self.alphabet_size)
        return self.models[ctx]

    def encode(self, symbol: int, context: int):
        model = self._get_model(context)
        cum_low, cum_high, total = model.get_freq(symbol)
        self.enc.encode(cum_low, cum_high, total)
        model.update(symbol)

    def finish(self) -> bytes:
        return self.enc.finish()


class ContextualDecoder:
    """Range decoder with context-dependent probability models."""

    def __init__(self, data: bytes, alphabet_size: int, n_contexts: int = 256):
        self.alphabet_size = alphabet_size
        self.models = {}
        self.dec = RangeDecoder(data)
        self.n_contexts = n_contexts

    def _get_model(self, ctx: int) -> AdaptiveModel:
        ctx = ctx % self.n_contexts
        if ctx not in self.models:
            self.models[ctx] = AdaptiveModel(self.alphabet_size)
        return self.models[ctx]

    def decode(self, context: int) -> int:
        model = self._get_model(context)
        total = model.total
        freq = self.dec.get_freq(total)
        symbol, cum_low, cum_high = model.get_symbol(freq)
        self.dec.decode(cum_low, cum_high, total)
        model.update(symbol)
        return symbol
