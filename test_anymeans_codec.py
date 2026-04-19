from __future__ import annotations

import http.server
import os
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


def test_large_payload_requires_source_url(tmp_path: Path) -> None:
    src = tmp_path / "large.bin"
    src.write_bytes(os.urandom(20000))

    packed = tmp_path / "large.amc"
    with pytest.raises(CodecError):
        compress_to_anymeans(src, packed)


def test_urlref_roundtrip_from_local_http(tmp_path: Path) -> None:
    src = tmp_path / "image.bin"
    src.write_bytes((b"A" * 5000) + (b"B" * 7000))

    httpd, thread, port = _serve_dir(tmp_path)
    try:
        url = f"http://127.0.0.1:{port}/{src.name}"
        packed = tmp_path / "remote.amc"

        res = compress_url_to_anymeans(url, packed)
        assert res["status"] == "urlref"
        assert packed.stat().st_size <= MAX_BYTES

        out = tmp_path / "remote.out"
        dec = decompress_from_anymeans(packed, out)
        assert dec["status"] == "ok"
        assert out.read_bytes() == src.read_bytes()
    finally:
        httpd.shutdown()
        thread.join(timeout=2)
