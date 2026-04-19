"""Adaptive Arithmetic Coder - Moffat/Neal/Witten style.

Uses 32-bit integer arithmetic with proper renormalization.
Adaptive frequency model updates after each symbol.

This is a proven algorithm from:
  Witten, Neal, Cleary: "Arithmetic Coding for Data Compression"
  Communications of the ACM, 1987.
"""

import struct
import numpy as np

# Precision constants
CODE_BITS = 32
TOP_VALUE = (1 << CODE_BITS) - 1
FIRST_QTR = TOP_VALUE // 4 + 1
HALF = 2 * FIRST_QTR
THIRD_QTR = 3 * FIRST_QTR
MAX_FREQ = FIRST_QTR  # Keep total < FIRST_QTR to avoid overflow


class AdaptiveModel:
    """Adaptive frequency model with periodic rescaling."""
    __slots__ = ('n', 'freq', 'cum', 'total')

    def __init__(self, n_symbols):
        self.n = n_symbols
        self.freq = [1] * n_symbols
        self.cum = list(range(n_symbols + 1))  # cum[i] = sum(freq[0..i-1])
        self.total = n_symbols

    def get_range(self, symbol):
        """Return (cum_low, cum_high, total) for encoding."""
        return self.cum[symbol], self.cum[symbol + 1], self.total

    def find_symbol(self, value):
        """Find symbol for scaled value. Returns (symbol, cum_low, cum_high)."""
        # Binary search on cumulative array
        lo, hi = 0, self.n
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cum[mid + 1] <= value:
                lo = mid + 1
            else:
                hi = mid
        return lo, self.cum[lo], self.cum[lo + 1]

    def update(self, symbol):
        """Increment count for symbol and rebuild cumulative."""
        self.freq[symbol] += 1
        self.total += 1
        # Update cumulative (only entries after symbol)
        for i in range(symbol + 1, self.n + 1):
            self.cum[i] += 1
        # Rescale if total too high
        if self.total >= MAX_FREQ:
            self._rescale()

    def _rescale(self):
        """Halve all frequencies, minimum 1."""
        self.total = 0
        for i in range(self.n):
            self.freq[i] = max(1, self.freq[i] >> 1)
            self.total += self.freq[i]
        # Rebuild cumulative
        self.cum[0] = 0
        for i in range(self.n):
            self.cum[i + 1] = self.cum[i] + self.freq[i]


class ArithEncoder:
    """Arithmetic encoder with bit-level output."""

    def __init__(self):
        self.low = 0
        self.high = TOP_VALUE
        self.bits_to_follow = 0
        self.output_bits = []

    def encode(self, cum_low, cum_high, total):
        """Encode one symbol given its cumulative frequency range."""
        r = self.high - self.low + 1
        self.high = self.low + (r * cum_high) // total - 1
        self.low = self.low + (r * cum_low) // total

        while True:
            if self.high < HALF:
                self._output_bit_plus_pending(0)
            elif self.low >= HALF:
                self._output_bit_plus_pending(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= FIRST_QTR and self.high < THIRD_QTR:
                self.bits_to_follow += 1
                self.low -= FIRST_QTR
                self.high -= FIRST_QTR
            else:
                break
            self.low = 2 * self.low
            self.high = 2 * self.high + 1

    def _output_bit_plus_pending(self, bit):
        self.output_bits.append(bit)
        while self.bits_to_follow > 0:
            self.output_bits.append(1 - bit)
            self.bits_to_follow -= 1

    def finish(self):
        """Flush encoder state."""
        self.bits_to_follow += 1
        if self.low < FIRST_QTR:
            self._output_bit_plus_pending(0)
        else:
            self._output_bit_plus_pending(1)

        # Pack bits into bytes
        result = bytearray()
        byte = 0
        bit_count = 0
        for bit in self.output_bits:
            byte = (byte << 1) | bit
            bit_count += 1
            if bit_count == 8:
                result.append(byte)
                byte = 0
                bit_count = 0
        if bit_count > 0:
            byte <<= (8 - bit_count)
            result.append(byte)

        return bytes(result)


class ArithDecoder:
    """Arithmetic decoder."""

    def __init__(self, data):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0
        self.low = 0
        self.high = TOP_VALUE
        self.value = 0

        # Read initial bits
        for _ in range(CODE_BITS):
            self.value = 2 * self.value + self._read_bit()

    def _read_bit(self):
        if self.byte_pos >= len(self.data):
            return 0
        bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
        return bit

    def get_value(self, total):
        """Get scaled value for symbol lookup."""
        r = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // r

    def decode(self, cum_low, cum_high, total):
        """Update decoder state after symbol identification."""
        r = self.high - self.low + 1
        self.high = self.low + (r * cum_high) // total - 1
        self.low = self.low + (r * cum_low) // total

        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.value -= HALF
                self.low -= HALF
                self.high -= HALF
            elif self.low >= FIRST_QTR and self.high < THIRD_QTR:
                self.value -= FIRST_QTR
                self.low -= FIRST_QTR
                self.high -= FIRST_QTR
            else:
                break
            self.low = 2 * self.low
            self.high = 2 * self.high + 1
            self.value = 2 * self.value + self._read_bit()


def encode_symbols_adaptive(symbols, alphabet_size):
    """Encode symbol sequence with adaptive arithmetic coding."""
    model = AdaptiveModel(alphabet_size)
    enc = ArithEncoder()

    for s in symbols:
        cum_low, cum_high, total = model.get_range(s)
        enc.encode(cum_low, cum_high, total)
        model.update(s)

    enc.finish()
    return enc.finish()


def decode_symbols_adaptive(data, count, alphabet_size):
    """Decode count symbols with adaptive arithmetic coding."""
    model = AdaptiveModel(alphabet_size)
    dec = ArithDecoder(data)
    symbols = []

    for _ in range(count):
        total = model.total
        value = dec.get_value(total)
        symbol, cum_low, cum_high = model.find_symbol(value)
        dec.decode(cum_low, cum_high, total)
        model.update(symbol)
        symbols.append(symbol)

    return symbols
