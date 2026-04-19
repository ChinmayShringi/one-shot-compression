#!/usr/bin/env python3
"""ANYMEANS codec: force media into <=1KB artifacts.

Design goals:
- Try practical iterative compression/transcoding first.
- If impossible, emit an "oracle token" (<1KB) that can restore bytes only when
  an external oracle/cache already holds the original object.

This is intentionally explicit about the trade-off: universal <1KB for arbitrary
media requires side information outside the bitstream.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import lzma
import mimetypes
import shutil
import struct
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image
except Exception:  # Pillow is optional in this repo.
    Image = None

MAGIC = b"AMC1"
MAX_BYTES = 1024

MODE_RAW = 0
MODE_TRANSCODED = 1
MODE_ORACLE = 2


@dataclass
class Attempt:
    name: str
    payload: bytes
    notes: str = ""


class CodecError(RuntimeError):
    pass


def _build_container(mode: int, metadata: dict, payload: bytes) -> bytes:
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    if len(meta_bytes) > 65535:
        raise CodecError("metadata too large")
    return MAGIC + bytes([mode]) + struct.pack(">H", len(meta_bytes)) + meta_bytes + payload


def _parse_container(blob: bytes) -> tuple[int, dict, bytes]:
    if len(blob) < 7 or blob[:4] != MAGIC:
        raise CodecError("invalid anymeans container")
    mode = blob[4]
    meta_len = struct.unpack(">H", blob[5:7])[0]
    end_meta = 7 + meta_len
    metadata = json.loads(blob[7:end_meta].decode("utf-8"))
    return mode, metadata, blob[end_meta:]


def _byte_compression_attempts(data: bytes) -> list[Attempt]:
    return [
        Attempt("zlib-9", zlib.compress(data, 9)),
        Attempt("bz2-9", bz2.compress(data, 9)),
        Attempt("lzma-9", lzma.compress(data, preset=9)),
    ]


def _image_attempts(path: Path) -> list[Attempt]:
    if Image is None:
        return []
    attempts: list[Attempt] = []
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        scales = [1.0, 0.5, 0.25, 0.125, 0.0625]
        qualities = [85, 65, 45, 30, 20, 10, 5]
        for s in scales:
            sw = max(1, int(w * s))
            sh = max(1, int(h * s))
            resized = rgb.resize((sw, sh))
            for q in qualities:
                for fmt, opts in (
                    ("JPEG", {"quality": q, "optimize": True}),
                    ("WEBP", {"quality": q, "method": 6}),
                ):
                    with tempfile.NamedTemporaryFile(suffix=f".{fmt.lower()}") as tf:
                        resized.save(tf.name, fmt, **opts)
                        payload = Path(tf.name).read_bytes()
                    attempts.append(Attempt(
                        f"{fmt.lower()}-q{q}-s{s}",
                        payload,
                        notes=f"{sw}x{sh}",
                    ))
    return attempts


def _ffmpeg_attempts(path: Path, media_kind: str) -> list[Attempt]:
    if shutil.which("ffmpeg") is None:
        return []

    attempts: list[Attempt] = []
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        if media_kind == "audio":
            variants = [
                ("opus6", ["-c:a", "libopus", "-b:a", "6k"]),
                ("opus4", ["-c:a", "libopus", "-b:a", "4k"]),
                ("aac8", ["-c:a", "aac", "-b:a", "8k"]),
            ]
            for name, codec_args in variants:
                out = td_path / f"{name}.mka"
                cmd = ["ffmpeg", "-y", "-i", str(path), *codec_args, str(out)]
                proc = subprocess.run(cmd, capture_output=True)
                if proc.returncode == 0 and out.exists():
                    attempts.append(Attempt(name, out.read_bytes()))
        elif media_kind == "video":
            variants = [
                ("h264-8k-96p", ["-vf", "scale=96:-2", "-c:v", "libx264", "-b:v", "8k", "-an"]),
                ("vp9-6k-64p", ["-vf", "scale=64:-2", "-c:v", "libvpx-vp9", "-b:v", "6k", "-an"]),
                ("av1-5k-48p", ["-vf", "scale=48:-2", "-c:v", "libaom-av1", "-b:v", "5k", "-an"]),
            ]
            for name, codec_args in variants:
                out = td_path / f"{name}.mkv"
                cmd = ["ffmpeg", "-y", "-i", str(path), "-t", "1", *codec_args, str(out)]
                proc = subprocess.run(cmd, capture_output=True)
                if proc.returncode == 0 and out.exists():
                    attempts.append(Attempt(name, out.read_bytes()))
    return attempts


def compress_to_anymeans(input_path: Path, output_path: Path, source_hint: str | None = None) -> dict:
    data = input_path.read_bytes()
    mime = mimetypes.guess_type(str(input_path))[0] or "application/octet-stream"
    sha = hashlib.sha256(data).hexdigest()

    candidates: list[Attempt] = []

    if len(data) <= MAX_BYTES:
        candidates.append(Attempt("raw", data, "already <=1KB"))

    candidates.extend(_byte_compression_attempts(data))

    if mime.startswith("image/"):
        candidates.extend(_image_attempts(input_path))
    elif mime.startswith("audio/"):
        candidates.extend(_ffmpeg_attempts(input_path, "audio"))
    elif mime.startswith("video/"):
        candidates.extend(_ffmpeg_attempts(input_path, "video"))

    fitting = [a for a in candidates if len(a.payload) <= MAX_BYTES]
    if fitting:
        raw = next((a for a in fitting if a.name == "raw"), None)
        best = raw if raw is not None else max(fitting, key=lambda a: len(a.payload))
        reversible = best.name in {"raw", "zlib-9", "bz2-9", "lzma-9"}
        mode = MODE_RAW if best.name == "raw" else MODE_TRANSCODED
        container = _build_container(mode, {
            "algo": best.name,
            "mime": mime,
            "sha256": sha,
            "lossless": reversible,
            "reversible": reversible,
            "notes": best.notes,
        }, best.payload)
        if len(container) > MAX_BYTES:
            # Fall through to oracle if metadata overhead pushed it over.
            fitting = []
        else:
            output_path.write_bytes(container)
            return {
                "status": "payload",
                "mode": mode,
                "algo": best.name,
                "size": len(container),
                "sha256": sha,
            }

    oracle_meta = {
        "algo": "oracle-token-v1",
        "mime": mime,
        "sha256": sha,
        "original_size": len(data),
        "source_hint": source_hint or str(input_path.name),
        "lossless": False,
        "requires_oracle": True,
    }
    container = _build_container(MODE_ORACLE, oracle_meta, b"")
    if len(container) > MAX_BYTES:
        raise CodecError("oracle token exceeded 1KB budget")

    output_path.write_bytes(container)
    return {
        "status": "oracle",
        "mode": MODE_ORACLE,
        "algo": "oracle-token-v1",
        "size": len(container),
        "sha256": sha,
    }


def decompress_from_anymeans(input_path: Path, output_path: Path, oracle_dir: Path | None = None) -> dict:
    mode, metadata, payload = _parse_container(input_path.read_bytes())

    if mode in (MODE_RAW, MODE_TRANSCODED):
        algo = metadata.get("algo", "raw")
        if algo == "zlib-9":
            out = zlib.decompress(payload)
        elif algo == "bz2-9":
            out = bz2.decompress(payload)
        elif algo == "lzma-9":
            out = lzma.decompress(payload)
        else:
            out = payload
        output_path.write_bytes(out)
        return {"status": "ok", "mode": mode, "bytes": len(out), "metadata": metadata}

    if mode != MODE_ORACLE:
        raise CodecError(f"unsupported mode: {mode}")

    oracle_root = oracle_dir or (Path.home() / ".anymeans_oracle")
    target = oracle_root / metadata["sha256"]
    if not target.exists():
        raise CodecError(
            "oracle token cannot be materialized locally; "
            f"missing blob {metadata['sha256']} in {oracle_root}"
        )
    output_path.write_bytes(target.read_bytes())
    return {"status": "oracle-restored", "mode": mode, "bytes": output_path.stat().st_size}


def _main() -> None:
    parser = argparse.ArgumentParser(description="ANYMEANS <=1KB media codec")
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compress")
    c.add_argument("input", type=Path)
    c.add_argument("output", type=Path)
    c.add_argument("--source-hint", default=None)

    d = sub.add_parser("decompress")
    d.add_argument("input", type=Path)
    d.add_argument("output", type=Path)
    d.add_argument("--oracle-dir", type=Path, default=None)

    args = parser.parse_args()

    if args.cmd == "compress":
        result = compress_to_anymeans(args.input, args.output, args.source_hint)
    else:
        result = decompress_from_anymeans(args.input, args.output, args.oracle_dir)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
