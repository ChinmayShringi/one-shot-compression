"""Encoder: 128-byte packed raster -> payload.

Reads only the committed raster bytes passed in. Does not import the
generator, read the seed, or take generator arguments.
"""

from __future__ import annotations

import struct
import zlib

import raster
from decode import (
    CORR_COORDS,
    CORR_PACKED,
    HEADER_LEN,
    MAGIC,
    REPR_SKEL,
    REPR_STROKE,
    VERSION,
    decode,
)

# Frozen once. Recorded in FROZEN_CONSTANTS.txt. Not searched after held-out.
RADIUS = 1
CHAIN_CODE_BITS = 3
ZLIB_LEVEL = 9


class BitWriter:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.acc = 0
        self.n = 0

    def write(self, value: int, nbits: int) -> None:
        for shift in range(nbits - 1, -1, -1):
            self.acc = (self.acc << 1) | ((value >> shift) & 1)
            self.n += 1
            if self.n == 8:
                self.buf.append(self.acc)
                self.acc = 0
                self.n = 0

    def finish(self) -> bytes:
        if self.n:
            self.acc <<= 8 - self.n
            self.buf.append(self.acc)
            self.acc = 0
            self.n = 0
        return bytes(self.buf)


def _neighbors(p: tuple[int, int], pixels: set[tuple[int, int]]) -> list[tuple[int, tuple[int, int]]]:
    x, y = p
    out = []
    for d, (dx, dy) in enumerate(raster.DIRS):
        q = (x + dx, y + dy)
        if q in pixels:
            out.append((d, q))
    return out


def extract_strokes(skel: list[list[int]]) -> list[tuple[int, int, list[int]]]:
    """Pixel-disjoint 8-connected path cover of the skeleton. Deterministic."""
    h = len(skel)
    w = len(skel[0]) if h else 0
    pixels = {(x, y) for y in range(h) for x in range(w) if skel[y][x]}
    degree = {p: len(_neighbors(p, pixels)) for p in pixels}
    unused = set(pixels)
    strokes: list[tuple[int, int, list[int]]] = []

    def start_key(p: tuple[int, int]) -> tuple[int, int]:
        return (p[1], p[0])

    while unused:
        endpoints = [p for p in unused if degree[p] == 1]
        if endpoints:
            start = min(endpoints, key=start_key)
        else:
            junctions = [p for p in unused if degree[p] >= 3]
            if junctions:
                start = min(junctions, key=start_key)
            else:
                start = min(unused, key=start_key)
        unused.remove(start)
        current = start
        dirs: list[int] = []
        while True:
            opts = [(d, q) for d, q in _neighbors(current, pixels) if q in unused]
            if not opts:
                break
            # Prefer extending through original degree-2 chain pixels, then smallest code.
            d, nxt = min(opts, key=lambda t: (0 if degree[t[1]] == 2 else 1, t[0]))
            dirs.append(d)
            unused.remove(nxt)
            current = nxt
        strokes.append((start[0], start[1], dirs))
    return strokes


def encode_strokes(strokes: list[tuple[int, int, list[int]]]) -> bytes:
    out = bytearray()
    out += struct.pack("<H", len(strokes))
    for x, y, dirs in strokes:
        if not (0 <= x <= 255 and 0 <= y <= 255):
            raise ValueError("stroke start does not fit in a byte")
        if len(dirs) > 65535:
            raise ValueError("stroke too long")
        out.append(x)
        out.append(y)
        out += struct.pack("<H", len(dirs))
        bw = BitWriter()
        for d in dirs:
            if not 0 <= d <= 7:
                raise ValueError("bad direction")
            bw.write(d, CHAIN_CODE_BITS)
        out += bw.finish()
    return bytes(out)


def _encode_coords(xor_img: list[list[int]]) -> bytes:
    coords = []
    for y, row in enumerate(xor_img):
        for x, v in enumerate(row):
            if v:
                coords.append((x, y))
    if len(coords) > 65535:
        raise ValueError("too many correction pixels")
    out = bytearray(struct.pack("<H", len(coords)))
    for x, y in coords:
        out.append(x)
        out.append(y)
    return bytes(out)


def _maybe_zlib(data: bytes) -> tuple[bytes, int]:
    compressed = zlib.compress(data, ZLIB_LEVEL)
    if len(compressed) < len(data):
        return compressed, 1
    return data, 0


def _correction_candidates(xor_img: list[list[int]], width: int, height: int) -> list[tuple[int, int, bytes]]:
    """Frozen correction codec: smaller of packed-mask and coord-list, each raw or zlib-9."""
    packed = raster.pack_bits(xor_img, width, height)
    coords = _encode_coords(xor_img)
    cands: list[tuple[int, int, bytes]] = []
    stored, z = _maybe_zlib(packed)
    cands.append((CORR_PACKED, z, stored))
    stored, z = _maybe_zlib(coords)
    cands.append((CORR_COORDS, z, stored))
    return cands


def _build_payload(
    repr_id: int,
    desc_raw: bytes,
    xor_img: list[list[int]],
    width: int,
    height: int,
) -> bytes:
    desc_stored, desc_zlib = _maybe_zlib(desc_raw)
    any_corr = any(v for row in xor_img for v in row)
    if not any_corr:
        corr_stored = b""
        corr_zlib = 0
        corr_format = CORR_PACKED
        has_corr = 0
    else:
        best: tuple[int, int, int, bytes] | None = None
        for fmt, zflag, stored in _correction_candidates(xor_img, width, height):
            key = (len(stored), fmt, zflag)
            if best is None or key < (best[0], best[1], best[2]):
                best = (len(stored), fmt, zflag, stored)
        assert best is not None
        _ln, corr_format, corr_zlib, corr_stored = best
        has_corr = 1
    if len(desc_stored) > 65535 or len(corr_stored) > 65535:
        raise ValueError("section longer than uint16")
    flags = 0
    flags |= repr_id & 0x1
    flags |= (RADIUS & 0x3) << 1
    flags |= (desc_zlib & 0x1) << 3
    flags |= (corr_zlib & 0x1) << 4
    flags |= (corr_format & 0x1) << 5
    flags |= (has_corr & 0x1) << 6
    header = bytearray()
    header += MAGIC
    header.append(VERSION)
    header.append(flags)
    header += struct.pack("<H", len(desc_stored))
    header += struct.pack("<H", len(corr_stored))
    payload = bytes(header) + desc_stored + corr_stored
    if len(payload) != HEADER_LEN + len(desc_stored) + len(corr_stored):
        raise ValueError("payload assembly error")
    return payload


def _strokes_match_skel(strokes: list[tuple[int, int, list[int]]], skel: list[list[int]]) -> bool:
    covered = raster.new_image(len(skel[0]), len(skel))
    seen = 0
    for x, y, dirs in strokes:
        if not (0 <= x < len(skel[0]) and 0 <= y < len(skel)):
            return False
        if covered[y][x]:
            return False
        covered[y][x] = 1
        seen += 1
        for d in dirs:
            dx, dy = raster.DIRS[d]
            x += dx
            y += dy
            if not (0 <= x < len(skel[0]) and 0 <= y < len(skel)):
                return False
            if covered[y][x]:
                return False
            covered[y][x] = 1
            seen += 1
    if seen != sum(sum(row) for row in skel):
        return False
    return raster.images_equal(covered, skel)


def encode(raster_bytes_128: bytes, width: int = 32, height: int = 32) -> bytes:
    if width != 32 or height != 32:
        raise ValueError("this experiment encodes only 32x32")
    if len(raster_bytes_128) != raster.PACKED_LEN:
        raise ValueError("expected 128 packed bytes")
    original = raster.unpack_bits(raster_bytes_128, width, height)
    skel = raster.zhang_suen(original)
    recon = raster.dilate(skel, RADIUS)
    xor_img = raster.xor_images(original, recon)

    skel_desc = raster.pack_bits(skel, width, height)
    payload_skel = _build_payload(REPR_SKEL, skel_desc, xor_img, width, height)

    strokes = extract_strokes(skel)
    if _strokes_match_skel(strokes, skel):
        stroke_desc = encode_strokes(strokes)
        payload_stroke = _build_payload(REPR_STROKE, stroke_desc, xor_img, width, height)
    else:
        payload_stroke = None

    # Keep the smaller exact payload. Tie -> packed skeleton bitmap.
    chosen = payload_skel
    if payload_stroke is not None and len(payload_stroke) < len(payload_skel):
        chosen = payload_stroke

    redraw = decode(chosen, width, height)
    if redraw != raster_bytes_128:
        raise RuntimeError("encoder produced a payload that does not redraw exactly")
    return chosen


if __name__ == "__main__":
    import pathlib

    root = pathlib.Path(__file__).resolve().parent
    fix_dir = root / "fixtures"
    out_dir = root / "payloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(24):
        src = fix_dir / f"glyph_{i:02d}.bin"
        data = src.read_bytes()
        payload = encode(data, 32, 32)
        (out_dir / f"glyph_{i:02d}.sg").write_bytes(payload)
        print(f"{i:02d} {len(payload)}")
