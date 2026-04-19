from __future__ import annotations

import json
import os
import struct
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from anymeans_codec import MAX_BYTES, CodecError, compress_to_anymeans, compress_url_to_anymeans, decompress_from_anymeans


def _make_png(width: int = 256, height: int = 256) -> bytes:
    img = Image.frombytes("RGB", (width, height), os.urandom(width * height * 3))
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def test_small_payload_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "tiny.bin"
    src.write_bytes(b"hello" * 20)

    packed = tmp_path / "tiny.amc"
    res = compress_to_anymeans(src, packed)

    assert packed.stat().st_size <= MAX_BYTES
    assert res["status"] == "payload"

    out = tmp_path / "tiny.out"
    dec = decompress_from_anymeans(packed, out)
    assert dec["status"] == "ok"
    assert out.read_bytes() == src.read_bytes()


def test_large_payload_uses_image_lossy_fallback(tmp_path: Path) -> None:
    src = tmp_path / "large.png"
    src.write_bytes(_make_png(320, 240))

    packed = tmp_path / "large.amc"
    res = compress_to_anymeans(src, packed)
    assert res["status"] == "lossy"
    assert packed.stat().st_size <= MAX_BYTES

    out = tmp_path / "large.out"
    dec = decompress_from_anymeans(packed, out)
    assert dec["status"] == "ok"
    assert dec["mode"] == "lossy"
    with Image.open(out) as img:
        assert img.format == "JPEG"
        assert img.size[0] <= 96 and img.size[1] <= 96

    with pytest.raises(CodecError):
        compress_to_anymeans(src, packed, source_url="https://example.com/input.png")


def test_compress_url_rejected() -> None:
    with pytest.raises(CodecError):
        compress_url_to_anymeans("http://127.0.0.1:9999/nope.bin", Path("/tmp/noop.amc"))


def test_decode_rejects_urlref_mode(tmp_path: Path) -> None:
    metadata = {"algo": "urlref-v1", "sha256": "0" * 64, "size": 1, "url": "https://example.com/x"}
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    blob = b"AMC1" + bytes([1]) + struct.pack(">H", len(meta_bytes)) + meta_bytes
    packed = tmp_path / "legacy_urlref.amc"
    packed.write_bytes(blob)

    with pytest.raises(CodecError):
        decompress_from_anymeans(packed, tmp_path / "out.bin")
