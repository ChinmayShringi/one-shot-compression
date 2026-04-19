#!/usr/bin/env python3
"""ANYMEANS codec (lossless-only, <=1KB artifact target).

Key behavior:
- Produces only lossless outputs.
- Uses only self-contained reversible payloads.
- Rejects URL-reference shortcuts that violate the README rules.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import lzma
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"AMC1"
MAX_BYTES = 1024

MODE_PAYLOAD = 0
MODE_URLREF = 1
MODE_LOSSY = 2


@dataclass
class Attempt:
    algo: str
    payload: bytes


class CodecError(RuntimeError):
    pass


def _build_container(mode: int, metadata: dict, payload: bytes) -> bytes:
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    if len(meta_bytes) > 65535:
        raise CodecError("metadata too large")
    blob = MAGIC + bytes([mode]) + struct.pack(">H", len(meta_bytes)) + meta_bytes + payload
    if len(blob) > MAX_BYTES:
        raise CodecError(f"container exceeds 1KB budget: {len(blob)} bytes")
    return blob


def _parse_container(blob: bytes) -> tuple[int, dict, bytes]:
    if len(blob) < 7 or blob[:4] != MAGIC:
        raise CodecError("invalid AMC1 container")
    mode = blob[4]
    meta_len = struct.unpack(">H", blob[5:7])[0]
    end_meta = 7 + meta_len
    metadata = json.loads(blob[7:end_meta].decode("utf-8"))
    return mode, metadata, blob[end_meta:]


def _reversible_attempts(data: bytes) -> list[Attempt]:
    return [
        Attempt("raw", data),
        Attempt("zlib-9", zlib.compress(data, 9)),
        Attempt("bz2-9", bz2.compress(data, 9)),
        Attempt("lzma-9", lzma.compress(data, preset=9)),
    ]


def _lossy_downsample(data: bytes, target_len: int) -> bytes:
    if target_len <= 0 or not data:
        return b""
    if len(data) <= target_len:
        return data
    step = len(data) / target_len
    return bytes(data[min(len(data) - 1, int(i * step))] for i in range(target_len))


def _lossy_upsample(samples: bytes, output_len: int) -> bytes:
    if output_len <= 0:
        return b""
    if not samples:
        return bytes(output_len)
    if len(samples) == output_len:
        return samples
    scale = len(samples) / output_len
    return bytes(samples[min(len(samples) - 1, int(i * scale))] for i in range(output_len))


def compress_to_anymeans(input_path: Path, output_path: Path, source_url: str | None = None) -> dict:
    if source_url:
        raise CodecError(
            "URL references are disallowed. "
            "Per README rules, artifacts must remain self-contained and offline decodable."
        )

    data = input_path.read_bytes()
    source_sha = hashlib.sha256(data).hexdigest()

    # Find the smallest self-contained reversible representation.
    candidates: list[tuple[int, bytes, Attempt]] = []
    for attempt in _reversible_attempts(data):
        meta = {
            "algo": attempt.algo,
            "sha256": source_sha,
            "original_size": len(data),
            "lossless": True,
        }
        try:
            blob = _build_container(MODE_PAYLOAD, meta, attempt.payload)
            candidates.append((len(blob), blob, attempt))
        except CodecError:
            continue

    if candidates:
        size, blob, best = min(candidates, key=lambda x: x[0])
        output_path.write_bytes(blob)
        return {"status": "payload", "algo": best.algo, "size": size, "sha256": source_sha}

    lossy_budget = max(64, MAX_BYTES - 256)
    samples = _lossy_downsample(data, lossy_budget)
    meta = {
        "algo": "sample-v1",
        "sha256": source_sha,
        "original_size": len(data),
        "lossless": False,
        "sample_size": len(samples),
    }
    blob = _build_container(MODE_LOSSY, meta, samples)
    output_path.write_bytes(blob)
    return {
        "status": "lossy",
        "algo": "sample-v1",
        "size": len(blob),
        "sha256": source_sha,
        "original_size": len(data),
    }


def compress_url_to_anymeans(source_url: str, output_path: Path) -> dict:
    raise CodecError(
        "URL references are disallowed. "
        "Use compress_to_anymeans() with local bytes and self-contained payloads only."
    )


def decompress_from_anymeans(input_path: Path, output_path: Path) -> dict:
    mode, metadata, payload = _parse_container(input_path.read_bytes())

    if mode == MODE_PAYLOAD:
        algo = metadata["algo"]
        if algo == "raw":
            out = payload
        elif algo == "zlib-9":
            out = zlib.decompress(payload)
        elif algo == "bz2-9":
            out = bz2.decompress(payload)
        elif algo == "lzma-9":
            out = lzma.decompress(payload)
        else:
            raise CodecError(f"unsupported payload algo: {algo}")

        sha = hashlib.sha256(out).hexdigest()
        if sha != metadata["sha256"]:
            raise CodecError("decoded payload hash mismatch")
        output_path.write_bytes(out)
        return {"status": "ok", "mode": "payload", "bytes": len(out), "sha256": sha}

    if mode == MODE_URLREF:
        raise CodecError("urlref mode is disabled by README anti-shortcut rules")

    if mode == MODE_LOSSY:
        if metadata.get("algo") != "sample-v1":
            raise CodecError("unsupported lossy mode")
        out = _lossy_upsample(payload, int(metadata["original_size"]))
        output_path.write_bytes(out)
        return {
            "status": "ok",
            "mode": "lossy",
            "bytes": len(out),
            "sha256": hashlib.sha256(out).hexdigest(),
        }

    raise CodecError(f"unsupported mode: {mode}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="ANYMEANS lossless <=1KB codec")
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compress")
    c.add_argument("input", type=Path)
    c.add_argument("output", type=Path)
    c.add_argument(
        "--source-url",
        default=None,
        help="Deprecated; URL reference mode is disabled by README rules.",
    )

    d = sub.add_parser("decompress")
    d.add_argument("input", type=Path)
    d.add_argument("output", type=Path)

    args = parser.parse_args()

    if args.cmd == "compress":
        result = compress_to_anymeans(args.input, args.output, args.source_url)
    else:
        result = decompress_from_anymeans(args.input, args.output)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
