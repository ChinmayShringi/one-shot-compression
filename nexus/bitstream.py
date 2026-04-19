"""Bit-level I/O for NEXUS codec. Zero compression library dependencies.

Handles individual bits, variable-length integers, and byte-aligned blocks.
All data flows through this layer before hitting disk.
"""

from __future__ import annotations


class BitWriter:
    """Write individual bits to a byte buffer."""

    __slots__ = ("_buffer", "_byte", "_bit_pos")

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._byte = 0
        self._bit_pos = 0  # next bit position within current byte (0-7)

    def write_bit(self, bit: int) -> None:
        """Write a single bit (0 or 1)."""
        self._byte |= (bit & 1) << self._bit_pos
        self._bit_pos += 1
        if self._bit_pos == 8:
            self._buffer.append(self._byte)
            self._byte = 0
            self._bit_pos = 0

    def write_bits(self, value: int, n: int) -> None:
        """Write n bits from value (LSB first)."""
        for i in range(n):
            self.write_bit((value >> i) & 1)

    def write_byte(self, value: int) -> None:
        """Write 8 bits."""
        self.write_bits(value & 0xFF, 8)

    def write_uint16(self, value: int) -> None:
        """Write 16-bit unsigned integer (little-endian)."""
        self.write_byte(value & 0xFF)
        self.write_byte((value >> 8) & 0xFF)

    def write_uint32(self, value: int) -> None:
        """Write 32-bit unsigned integer (little-endian)."""
        for i in range(4):
            self.write_byte((value >> (i * 8)) & 0xFF)

    def write_varint(self, value: int) -> None:
        """Write variable-length integer (unsigned, 7 bits per byte + continue flag).

        Smaller values use fewer bytes. Encodes values 0-127 in 1 byte,
        128-16383 in 2 bytes, etc.
        """
        while value >= 0x80:
            self.write_byte((value & 0x7F) | 0x80)
            value >>= 7
        self.write_byte(value & 0x7F)

    def write_signed_varint(self, value: int) -> None:
        """Write signed integer using zigzag encoding + varint.

        Maps: 0->0, -1->1, 1->2, -2->3, 2->4, ...
        This keeps small-magnitude values compact regardless of sign.
        """
        zigzag = (value << 1) ^ (value >> 63) if value >= 0 else (((-value) << 1) - 1)
        self.write_varint(zigzag)

    def write_bytes(self, data: bytes) -> None:
        """Write raw bytes."""
        for b in data:
            self.write_byte(b)

    def flush(self) -> bytes:
        """Flush remaining bits (zero-padded) and return complete buffer."""
        if self._bit_pos > 0:
            self._buffer.append(self._byte)
            self._byte = 0
            self._bit_pos = 0
        return bytes(self._buffer)

    @property
    def bits_written(self) -> int:
        return len(self._buffer) * 8 + self._bit_pos


class BitReader:
    """Read individual bits from a byte buffer."""

    __slots__ = ("_data", "_byte_pos", "_bit_pos")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._byte_pos = 0
        self._bit_pos = 0

    def read_bit(self) -> int:
        """Read a single bit."""
        if self._byte_pos >= len(self._data):
            raise EOFError("End of bitstream")
        bit = (self._data[self._byte_pos] >> self._bit_pos) & 1
        self._bit_pos += 1
        if self._bit_pos == 8:
            self._byte_pos += 1
            self._bit_pos = 0
        return bit

    def read_bits(self, n: int) -> int:
        """Read n bits and return as integer (LSB first)."""
        value = 0
        for i in range(n):
            value |= self.read_bit() << i
        return value

    def read_byte(self) -> int:
        """Read 8 bits."""
        return self.read_bits(8)

    def read_uint16(self) -> int:
        """Read 16-bit unsigned integer (little-endian)."""
        return self.read_byte() | (self.read_byte() << 8)

    def read_uint32(self) -> int:
        """Read 32-bit unsigned integer (little-endian)."""
        result = 0
        for i in range(4):
            result |= self.read_byte() << (i * 8)
        return result

    def read_varint(self) -> int:
        """Read variable-length integer."""
        value = 0
        shift = 0
        while True:
            b = self.read_byte()
            value |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        return value

    def read_signed_varint(self) -> int:
        """Read signed zigzag-encoded varint."""
        zigzag = self.read_varint()
        return (zigzag >> 1) ^ (-(zigzag & 1))

    @property
    def bits_remaining(self) -> int:
        return (len(self._data) - self._byte_pos) * 8 - self._bit_pos
