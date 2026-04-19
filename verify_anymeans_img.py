#!/usr/bin/env python3
"""Run the ANYMEANS codec against ./img.png and verify exact reconstruction."""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

from anymeans_codec import compress_to_anymeans, decompress_from_anymeans

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "img.png"
ARTIFACT = ROOT / "output" / "img.anymeans"
RECON = ROOT / "output" / "img.anymeans.decoded.png"


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter_png(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not PNG")

    off = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()

    while off < len(data):
        length = struct.unpack(">I", data[off:off + 4])[0]
        off += 4
        chunk_type = data[off:off + 4]
        off += 4
        chunk_data = data[off:off + length]
        off += length + 4  # skip chunk + crc

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", chunk_data)
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break

    if bit_depth != 8:
        raise ValueError(f"unsupported bit depth: {bit_depth}")

    channels = {2: 3, 6: 4}.get(color_type)
    if channels is None:
        raise ValueError(f"unsupported color type: {color_type}")

    bpp = channels
    stride = width * bpp
    raw = zlib.decompress(bytes(idat))

    out = bytearray(height * stride)
    src_i = 0
    dst_i = 0
    prev = bytearray(stride)

    for _ in range(height):
        f = raw[src_i]
        src_i += 1
        row = bytearray(raw[src_i:src_i + stride])
        src_i += stride

        if f == 1:  # Sub
            for i in range(stride):
                row[i] = (row[i] + (row[i - bpp] if i >= bpp else 0)) & 0xFF
        elif f == 2:  # Up
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif f == 3:  # Average
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif f == 4:  # Paeth
            for i in range(stride):
                a = row[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + _paeth(a, b, c)) & 0xFF
        elif f != 0:
            raise ValueError(f"unsupported PNG filter: {f}")

        out[dst_i:dst_i + stride] = row
        dst_i += stride
        prev = row

    return width, height, bytes(out)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ARTIFACT.parent.mkdir(exist_ok=True)

    result = compress_to_anymeans(SRC, ARTIFACT, source_url=SRC.resolve().as_uri())
    restored = decompress_from_anymeans(ARTIFACT, RECON)

    artifact_size = ARTIFACT.stat().st_size
    src_sha = _sha256(SRC)
    recon_sha = _sha256(RECON)

    if artifact_size > 1024:
        raise SystemExit(f"FAIL: artifact is {artifact_size} bytes (>1024)")
    if src_sha != recon_sha:
        raise SystemExit("FAIL: byte-level hash mismatch")

    src_w, src_h, src_pixels = _unfilter_png(SRC)
    rec_w, rec_h, rec_pixels = _unfilter_png(RECON)

    if (src_w, src_h) != (rec_w, rec_h):
        raise SystemExit(f"FAIL: dimension mismatch {(src_w, src_h)} != {(rec_w, rec_h)}")
    if src_pixels != rec_pixels:
        raise SystemExit("FAIL: pixel data mismatch")

    print("PASS")
    print(f"artifact={ARTIFACT} ({artifact_size} bytes)")
    print(f"mode={result['status']} sha256={src_sha}")
    print(f"decoded={RECON} verify={restored['status']}")
    print(f"pixel_exact={len(src_pixels)} channel-bytes matched")


if __name__ == "__main__":
    main()
