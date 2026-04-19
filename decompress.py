#!/usr/bin/env python3
"""Reconstruct images from custom compressed formats.

Usage:
    python decompress.py <input_file> <output.png> [--target-size WxH]

Supported: .svd, .cpal, .chyb, .cpar, and standard image formats.
"""

import argparse
import struct
import sys
import zlib

import numpy as np
from PIL import Image

MAGIC_SVD = b"SVD1"
MAGIC_CPAL = b"CPAL"
MAGIC_CHYB = b"CHYB"
MAGIC_CPAR = b"CPAR"


def detect_format(path: str) -> str:
    """Detect format by magic bytes or extension."""
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == MAGIC_SVD:
        return "svd"
    if magic == MAGIC_CPAL:
        return "cpal"
    if magic == MAGIC_CHYB:
        return "chyb"
    if magic == MAGIC_CPAR:
        return "cpar"
    # Standard format -- let Pillow handle it
    return "standard"


def decompress_svd(data: bytes, target_size: tuple[int, int] | None) -> Image.Image:
    """Reconstruct from SVD binary format."""
    # Header: magic(4) + width(2) + height(2) + rank(2) + channels(1) + pad(1)
    magic, w, h, rank, channels, _ = struct.unpack_from("<4sHHHBB", data, 0)
    body = zlib.decompress(data[12:])
    offset = 0
    img_arr = np.zeros((h, w, channels), dtype=np.float64)

    for ch in range(channels):
        u_min, u_max = struct.unpack_from("<ff", body, offset)
        offset += 8
        u_range = u_max - u_min if u_max != u_min else 1.0
        U_q = np.frombuffer(body, dtype=np.uint8, count=h * rank, offset=offset)
        U_k = U_q.reshape(h, rank).astype(np.float64) / 255.0 * u_range + u_min
        offset += h * rank

        S_k = np.frombuffer(body, dtype=np.float32, count=rank, offset=offset)
        S_k = S_k.astype(np.float64)
        offset += rank * 4

        vt_min, vt_max = struct.unpack_from("<ff", body, offset)
        offset += 8
        vt_range = vt_max - vt_min if vt_max != vt_min else 1.0
        Vt_q = np.frombuffer(body, dtype=np.uint8, count=rank * w, offset=offset)
        Vt_k = Vt_q.reshape(rank, w).astype(np.float64) / 255.0 * vt_range + vt_min
        offset += rank * w

        img_arr[:, :, ch] = U_k @ np.diag(S_k) @ Vt_k

    result = np.clip(img_arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(result)
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return img


def decompress_custom_palette(data: bytes,
                              target_size: tuple[int, int] | None) -> Image.Image:
    """Reconstruct from custom palette format."""
    # Header: magic(4) + width(2) + height(2) + n_colors(1) + bpp(1)
    magic, w, h, n_colors, bpp = struct.unpack_from("<4sHHBB", data, 0)
    offset = 10
    palette = np.frombuffer(data, dtype=np.uint8, count=n_colors * 3,
                            offset=offset).reshape(n_colors, 3)
    offset += n_colors * 3
    compressed_indices = data[offset:]
    packed = zlib.decompress(compressed_indices)
    indices = _unpack_indices(packed, bpp, w * h)
    pixels = palette[indices].reshape(h, w, 3)
    img = Image.fromarray(pixels)
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return img


def _unpack_indices(packed: bytes, bits_per_pixel: int, count: int) -> np.ndarray:
    """Unpack bit-packed indices."""
    result = np.zeros(count, dtype=np.uint8)
    mask = (1 << bits_per_pixel) - 1
    bit_pos = 0
    for i in range(count):
        byte_offset = bit_pos // 8
        bit_offset = bit_pos % 8
        if byte_offset < len(packed):
            val = packed[byte_offset] >> bit_offset
            if bit_offset + bits_per_pixel > 8 and byte_offset + 1 < len(packed):
                val |= packed[byte_offset + 1] << (8 - bit_offset)
            result[i] = val & mask
        bit_pos += bits_per_pixel
    return result


def decompress_hybrid(data: bytes,
                      target_size: tuple[int, int] | None) -> Image.Image:
    """Reconstruct from hybrid thumbnail + edge format."""
    # Header: magic(4) + thumb_dim(1) + thumb_q(1) + edge_dim(1) +
    #          thumb_size(2) + edge_size(2) + pad(1)
    magic, thumb_dim, thumb_q, edge_dim, thumb_size, edge_size, _ = \
        struct.unpack_from("<4sBBBHHB", data, 0)

    thumb_data = data[12:12 + thumb_size]
    edge_data = data[12 + thumb_size:12 + thumb_size + edge_size]

    import io
    thumb_img = Image.open(io.BytesIO(thumb_data)).convert("RGB")

    # Decompress edge map
    edge_packed = zlib.decompress(edge_data)
    edge_bits = np.unpackbits(np.frombuffer(edge_packed, dtype=np.uint8))
    edge_pixels = edge_bits[:edge_dim * edge_dim].reshape(edge_dim, edge_dim)

    # Upscale thumbnail
    out_size = target_size if target_size else (thumb_dim * 4, thumb_dim * 4)
    result = thumb_img.resize(out_size, Image.LANCZOS)

    # Apply edge enhancement
    edge_resized = Image.fromarray((edge_pixels * 255).astype(np.uint8)).resize(
        out_size, Image.LANCZOS)
    import cv2
    result_arr = np.array(result)
    edge_arr = np.array(edge_resized).astype(np.float32) / 255.0
    # Sharpen edges
    for ch in range(3):
        result_arr[:, :, ch] = np.clip(
            result_arr[:, :, ch].astype(np.float32) + edge_arr * 30 - 15,
            0, 255,
        ).astype(np.uint8)
    return Image.fromarray(result_arr)


def decompress_parametric(data: bytes,
                          target_size: tuple[int, int] | None) -> Image.Image:
    """Reconstruct from parametric format."""
    # Header: magic(4) + n_colors(1) + grad_rows(1) + label_size(2)
    magic, n_colors, grad_rows, label_size = struct.unpack_from("<4sBBH", data, 0)
    offset = 8
    centers = np.frombuffer(data, dtype=np.uint8, count=n_colors * 3,
                            offset=offset).reshape(n_colors, 3)
    offset += n_colors * 3
    row_avgs = np.frombuffer(data, dtype=np.uint8, count=grad_rows * 3,
                             offset=offset).reshape(grad_rows, 3)
    offset += grad_rows * 3
    label_compressed = data[offset:offset + label_size]
    label_data = zlib.decompress(label_compressed)
    labels = np.frombuffer(label_data, dtype=np.uint8).reshape(grad_rows, grad_rows)

    # Reconstruct: blend color map with gradient
    color_img = centers[labels]  # (grad_rows, grad_rows, 3)
    gradient = np.repeat(row_avgs[:, np.newaxis, :], grad_rows, axis=1)
    # Blend: 60% color map + 40% gradient
    blended = (0.6 * color_img.astype(np.float32) +
               0.4 * gradient.astype(np.float32))
    result = np.clip(blended, 0, 255).astype(np.uint8)
    img = Image.fromarray(result)
    out_size = target_size if target_size else (grad_rows * 8, grad_rows * 8)
    return img.resize(out_size, Image.LANCZOS)


def decompress_standard(path: str,
                        target_size: tuple[int, int] | None) -> Image.Image:
    """Decode standard format and optionally resize."""
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "jxl":
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            subprocess.run(["djxl", path, tmp.name], check=True, capture_output=True)
            img = Image.open(tmp.name).convert("RGB")
            import os
            os.unlink(tmp.name)
    else:
        img = Image.open(path).convert("RGB")
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return img


def decompress(input_path: str, output_path: str,
               target_size: tuple[int, int] | None = None) -> None:
    fmt = detect_format(input_path)
    if fmt == "standard":
        img = decompress_standard(input_path, target_size)
    else:
        with open(input_path, "rb") as f:
            data = f.read()
        if fmt == "svd":
            img = decompress_svd(data, target_size)
        elif fmt == "cpal":
            img = decompress_custom_palette(data, target_size)
        elif fmt == "chyb":
            img = decompress_hybrid(data, target_size)
        elif fmt == "cpar":
            img = decompress_parametric(data, target_size)
        else:
            raise ValueError(f"Unknown format: {fmt}")
    img.save(output_path, "PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompress custom image formats")
    parser.add_argument("input", help="Input compressed file")
    parser.add_argument("output", help="Output PNG path")
    parser.add_argument("--target-size", help="Target size WxH (e.g. 1630x1626)")
    args = parser.parse_args()
    target = None
    if args.target_size:
        w, h = args.target_size.split("x")
        target = (int(w), int(h))
    decompress(args.input, args.output, target)


if __name__ == "__main__":
    main()
