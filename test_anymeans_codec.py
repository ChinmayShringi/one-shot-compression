from __future__ import annotations

import http.server
import json
import os
import struct
import socketserver
import threading
from pathlib import Path

import pytest

from anymeans_codec import MAX_BYTES, CodecError, compress_to_anymeans, compress_url_to_anymeans, decompress_from_anymeans


class _SilentHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # pragma: no cover
        return


def _serve_dir(path: Path):
    handler = lambda *args, **kwargs: _SilentHandler(*args, directory=str(path), **kwargs)
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread, httpd.server_address[1]


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


def test_large_payload_requires_side_information(tmp_path: Path) -> None:
    src = tmp_path / "large.bin"
    src.write_bytes(os.urandom(20000))

    packed = tmp_path / "large.amc"
    with pytest.raises(CodecError):
        compress_to_anymeans(src, packed)
    with pytest.raises(CodecError):
        compress_to_anymeans(src, packed, source_url="https://example.com/input.bin")


def test_urlref_roundtrip_from_local_http(tmp_path: Path) -> None:
    src = tmp_path / "image.bin"
    src.write_bytes((b"A" * 5000) + (b"B" * 7000))

    httpd, thread, port = _serve_dir(tmp_path)
    try:
        url = f"http://127.0.0.1:{port}/{src.name}"
        packed = tmp_path / "remote.amc"
        with pytest.raises(CodecError):
            compress_url_to_anymeans(url, packed)
    finally:
        httpd.shutdown()
        thread.join(timeout=2)


def test_decode_rejects_urlref_mode(tmp_path: Path) -> None:
    metadata = {"algo": "urlref-v1", "sha256": "0" * 64, "size": 1, "url": "https://example.com/x"}
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    blob = b"AMC1" + bytes([1]) + struct.pack(">H", len(meta_bytes)) + meta_bytes
    packed = tmp_path / "legacy_urlref.amc"
    packed.write_bytes(blob)

    with pytest.raises(CodecError):
        decompress_from_anymeans(packed, tmp_path / "out.bin")
