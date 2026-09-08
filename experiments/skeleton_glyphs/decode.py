"""Pure decoder: payload bytes -> 128-byte packed raster.

No files, no seed, no generator, no network. Two calls on the same payload match.
"""

from __future__ import annotations

import struct
import zlib

import raster

MAGIC = b"SG"
VERSION = 1
HEADER_LEN = 8

REPR_SKEL = 0
REPR_STROKE = 1
CORR_PACKED = 0
CORR_COORDS = 1


class BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.i = 0
        self.acc = 0
        self.n = 0

    def read(self, nbits: int) -> int:
        value = 0
        for _ in range(nbits):
            if self.n == 0:
                if self.i >= len(self.data):
                    raise ValueError("chain code truncated")
                self.acc = self.data[self.i]
                self.i += 1
                self.n = 8
            self.n -= 1
            bit = (self.acc >> self.n) & 1
            value = (value << 1) | bit
        return value


def inspect_payload(payload: bytes) -> dict:
    if len(payload) < HEADER_LEN:
        raise ValueError("payload shorter than header")
    if payload[0:2] != MAGIC:
        raise ValueError("bad magic")
    version = payload[2]
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    flags = payload[3]
    desc_len = struct.unpack_from("<H", payload, 4)[0]
    corr_len = struct.unpack_from("<H", payload, 6)[0]
    end = HEADER_LEN + desc_len + corr_len
    if end != len(payload):
        raise ValueError("payload length does not match header fields")
    return {
        "version": version,
        "flags": flags,
        "repr": flags & 0x1,
        "radius": (flags >> 1) & 0x3,
        "desc_zlib": (flags >> 3) & 0x1,
        "corr_zlib": (flags >> 4) & 0x1,
        "corr_format": (flags >> 5) & 0x1,
        "has_corr": (flags >> 6) & 0x1,
        "desc_len": desc_len,
        "corr_len": corr_len,
        "header_len": HEADER_LEN,
        "payload_bytes": len(payload),
        "desc_bytes": desc_len,
        "corr_bytes": corr_len,
    }


def _apply_xor(base: list[list[int]], xor_img: list[list[int]]) -> list[list[int]]:
    out = raster.new_image(len(base[0]), len(base))
    for y in range(len(base)):
        for x in range(len(base[0])):
            out[y][x] = 1 if base[y][x] != xor_img[y][x] else 0
    return out


def _decode_coords(raw: bytes, width: int, height: int) -> list[list[int]]:
    if len(raw) < 2:
        raise ValueError("coord correction truncated")
    count = struct.unpack_from("<H", raw, 0)[0]
    need = 2 + count * 2
    if len(raw) != need:
        raise ValueError("coord correction length mismatch")
    img = raster.new_image(width, height)
    i = 2
    for _ in range(count):
        x = raw[i]
        y = raw[i + 1]
        i += 2
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("coord out of range")
        img[y][x] = 1
    return img


def _decode_correction(
    data: bytes,
    corr_format: int,
    corr_zlib: int,
    width: int,
    height: int,
) -> list[list[int]]:
    raw = zlib.decompress(data) if corr_zlib else data
    if corr_format == CORR_PACKED:
        return raster.unpack_bits(raw, width, height)
    if corr_format == CORR_COORDS:
        return _decode_coords(raw, width, height)
    raise ValueError("unknown correction format")


def _rasterize_strokes(data: bytes, width: int, height: int) -> list[list[int]]:
    if len(data) < 2:
        raise ValueError("stroke list truncated")
    n = struct.unpack_from("<H", data, 0)[0]
    i = 2
    img = raster.new_image(width, height)
    for _ in range(n):
        if i + 4 > len(data):
            raise ValueError("stroke header truncated")
        x = data[i]
        y = data[i + 1]
        i += 2
        n_steps = struct.unpack_from("<H", data, i)[0]
        i += 2
        n_bytes = (n_steps * 3 + 7) // 8
        if i + n_bytes > len(data):
            raise ValueError("stroke chain truncated")
        chunk = data[i : i + n_bytes]
        i += n_bytes
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("stroke start out of range")
        img[y][x] = 1
        reader = BitReader(chunk)
        for _s in range(n_steps):
            d = reader.read(3)
            if d > 7:
                raise ValueError("bad chain code")
            dx, dy = raster.DIRS[d]
            x += dx
            y += dy
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError("stroke left the raster")
            img[y][x] = 1
        # leftover bits in the last byte are padding; reader may not consume them
    if i != len(data):
        raise ValueError("stroke list did not consume description")
    return img


def decode_image(payload: bytes, width: int = raster.WIDTH, height: int = raster.HEIGHT) -> list[list[int]]:
    info = inspect_payload(payload)
    desc = payload[HEADER_LEN : HEADER_LEN + info["desc_len"]]
    corr = payload[HEADER_LEN + info["desc_len"] :]
    if info["desc_zlib"]:
        desc = zlib.decompress(desc)
    if info["repr"] == REPR_SKEL:
        skel = raster.unpack_bits(desc, width, height)
    elif info["repr"] == REPR_STROKE:
        skel = _rasterize_strokes(desc, width, height)
    else:
        raise ValueError("unknown representation")
    recon = raster.dilate(skel, info["radius"])
    if info["corr_len"] == 0:
        if info["has_corr"]:
            raise ValueError("has_corr flag set but corr_len is 0")
        return recon
    if not info["has_corr"]:
        raise ValueError("corr_len > 0 but has_corr flag clear")
    xor_img = _decode_correction(
        corr, info["corr_format"], info["corr_zlib"], width, height
    )
    return _apply_xor(recon, xor_img)


def decode(payload: bytes, width: int = raster.WIDTH, height: int = raster.HEIGHT) -> bytes:
    img = decode_image(payload, width, height)
    return raster.pack_bits(img, width, height)
