from pathlib import Path

from anymeans_codec import MAX_BYTES, compress_to_anymeans, decompress_from_anymeans


def test_small_payload_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "tiny.bin"
    src.write_bytes(b"hello" * 10)

    packed = tmp_path / "tiny.amc"
    res = compress_to_anymeans(src, packed)
    assert packed.stat().st_size <= MAX_BYTES
    assert res["status"] == "payload"

    out = tmp_path / "tiny.out"
    dec = decompress_from_anymeans(packed, out)
    assert dec["status"] == "ok"
    assert out.read_bytes() == src.read_bytes()


def test_large_payload_uses_oracle_token(tmp_path: Path) -> None:
    src = tmp_path / "large.bin"
    src.write_bytes((b"0123456789abcdef" * 5000))

    packed = tmp_path / "large.amc"
    res = compress_to_anymeans(src, packed)

    assert packed.stat().st_size <= MAX_BYTES
    assert res["status"] in {"payload", "oracle"}

    if res["status"] == "oracle":
        # No oracle cache configured by default; decode should fail loudly.
        out = tmp_path / "large.out"
        try:
            decompress_from_anymeans(packed, out)
            assert False, "expected oracle decode failure"
        except RuntimeError:
            pass
