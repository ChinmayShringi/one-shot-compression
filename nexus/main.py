#!/usr/bin/env python3
"""NEXUS codec CLI - compress and decompress media files.

Usage:
    python -m nexus.main compress vid.mp4 output.nxv [--frames N] [--scale S]
    python -m nexus.main decompress output.nxv reconstructed/
    python -m nexus.main verify vid.mp4 output.nxv [--frames N] [--scale S]
    python -m nexus.main benchmark vid.mp4 [--frames N] [--scale S]
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .codec import encode_video, decode_video


def extract_frames(video_path: str, max_frames: int | None = None,
                   scale: float = 1.0) -> tuple[list[np.ndarray], dict]:
    """Extract raw frames from a video file using ffmpeg.

    Note: ffmpeg is used for I/O only (demuxing/decoding the container).
    All NEXUS compression is from scratch.
    """
    # Get video info
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", video_path],
        capture_output=True, text=True,
    )
    import json
    info = json.loads(probe.stdout)

    video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    nb_frames = int(video_stream.get("nb_frames", 0))
    fps_str = video_stream.get("r_frame_rate", "25/1")
    fps_parts = fps_str.split("/")
    fps_num, fps_den = int(fps_parts[0]), int(fps_parts[1]) if len(fps_parts) > 1 else 1

    # Scale dimensions
    new_w = int(width * scale) & ~1  # ensure even
    new_h = int(height * scale) & ~1

    # Extract frames as raw grayscale (Y channel)
    n_frames = min(max_frames, nb_frames) if max_frames else nb_frames
    cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", video_path,
        "-vf", f"scale={new_w}:{new_h}",
        "-pix_fmt", "gray",
        "-frames:v", str(n_frames),
        "-f", "rawvideo", "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    raw = result.stdout

    frame_size = new_w * new_h
    actual_frames = len(raw) // frame_size
    frames = []
    for i in range(actual_frames):
        frame_data = raw[i * frame_size:(i + 1) * frame_size]
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(new_h, new_w)
        frames.append(frame)

    # Extract audio
    audio_cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", video_path,
        "-ar", "16000",  # downsample to 16kHz for smaller size
        "-ac", "1",       # mono
        "-f", "s16le",
        "-t", str(n_frames / (fps_num / fps_den)),  # match video duration
        "-",
    ]
    audio_result = subprocess.run(audio_cmd, capture_output=True)
    audio_raw = audio_result.stdout
    audio_samples = np.frombuffer(audio_raw, dtype=np.int16) if audio_raw else None

    metadata = {
        "original_width": width, "original_height": height,
        "scaled_width": new_w, "scaled_height": new_h,
        "fps": (fps_num, fps_den),
        "n_frames": actual_frames,
        "audio_rate": 16000,
        "audio_channels": 1,
    }
    return frames, metadata, audio_samples


def cmd_compress(args: argparse.Namespace) -> None:
    """Compress a video file to NEXUS format."""
    print(f"NEXUS Codec - Compressing {args.input}")
    print(f"  Max frames: {args.frames or 'all'}")
    print(f"  Scale: {args.scale}")

    t0 = time.time()
    frames, meta, audio = extract_frames(args.input, args.frames, args.scale)
    t_extract = time.time() - t0
    print(f"  Extracted {len(frames)} frames at {meta['scaled_width']}x"
          f"{meta['scaled_height']} in {t_extract:.1f}s")

    raw_size = sum(f.nbytes for f in frames)
    if audio is not None:
        raw_size += audio.nbytes
        print(f"  Audio: {len(audio):,} samples at 16kHz")
    print(f"  Raw data: {raw_size:,} bytes ({raw_size / 1024 / 1024:.1f} MB)")

    def progress(i, total):
        if i % 10 == 0 or i == total:
            elapsed = time.time() - t0
            fps_val = i / max(elapsed - t_extract, 0.01)
            print(f"  Encoding frame {i}/{total} ({fps_val:.1f} fps)", end="\r")

    t1 = time.time()
    encoded = encode_video(
        frames, meta["scaled_width"], meta["scaled_height"],
        fps=meta["fps"],
        audio_samples=audio,
        audio_rate=meta["audio_rate"],
        audio_channels=meta["audio_channels"],
        progress_fn=progress,
    )
    t_encode = time.time() - t1
    print()

    with open(args.output, "wb") as f:
        f.write(encoded)

    original_size = os.path.getsize(args.input)
    nexus_size = len(encoded)

    print(f"\n  Results:")
    print(f"    Original (H.264):  {original_size:>12,} bytes ({original_size / 1024 / 1024:.1f} MB)")
    print(f"    Raw pixels+audio:  {raw_size:>12,} bytes ({raw_size / 1024 / 1024:.1f} MB)")
    print(f"    NEXUS compressed:  {nexus_size:>12,} bytes ({nexus_size / 1024 / 1024:.1f} MB)")
    print(f"    Ratio (vs raw):    {raw_size / nexus_size:.2f}x")
    print(f"    Ratio (vs H.264):  {'better' if nexus_size < original_size else 'worse'}"
          f" ({nexus_size / original_size * 100:.1f}% of H.264)")
    print(f"    Encode time:       {t_encode:.1f}s ({len(frames) / t_encode:.1f} fps)")
    print(f"\n  Output: {args.output}")


def cmd_decompress(args: argparse.Namespace) -> None:
    """Decompress NEXUS file back to frames."""
    print(f"NEXUS Codec - Decompressing {args.input}")

    with open(args.input, "rb") as f:
        data = f.read()

    t0 = time.time()

    def progress(i, total):
        if i % 10 == 0 or i == total:
            print(f"  Decoding frame {i}/{total}", end="\r")

    frames, metadata = decode_video(data, progress_fn=progress)
    t_decode = time.time() - t0
    print()
    print(f"  Decoded {len(frames)} frames in {t_decode:.1f}s")

    # Save frames as images
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame, "L").save(out_dir / f"frame_{i:05d}.png")

    print(f"  Saved to {out_dir}/")


def cmd_verify(args: argparse.Namespace) -> None:
    """Compress, decompress, and verify lossless round-trip."""
    print(f"NEXUS Codec - Verify Lossless Round-Trip")
    print(f"  Input: {args.input}")

    frames, meta, audio = extract_frames(args.input, args.frames, args.scale)
    print(f"  Extracted {len(frames)} frames at {meta['scaled_width']}x"
          f"{meta['scaled_height']}")

    t0 = time.time()
    encoded = encode_video(
        frames, meta["scaled_width"], meta["scaled_height"],
        fps=meta["fps"],
        audio_samples=audio,
        audio_rate=meta["audio_rate"],
        audio_channels=meta["audio_channels"],
    )
    t_encode = time.time() - t0

    t1 = time.time()
    decoded_frames, decoded_meta = decode_video(encoded)
    t_decode = time.time() - t1

    # Verify each frame
    all_match = True
    max_diff = 0
    for i, (orig, dec) in enumerate(zip(frames, decoded_frames)):
        if not np.array_equal(orig, dec):
            diff = np.abs(orig.astype(int) - dec.astype(int))
            frame_max = diff.max()
            max_diff = max(max_diff, frame_max)
            n_diff = np.count_nonzero(diff)
            print(f"  Frame {i}: MISMATCH - {n_diff} pixels differ, "
                  f"max diff = {frame_max}")
            all_match = False

    raw_size = sum(f.nbytes for f in frames)
    if audio is not None:
        raw_size += audio.nbytes
    nexus_size = len(encoded)
    original_size = os.path.getsize(args.input)

    print(f"\n  Verification: {'LOSSLESS - all frames match!' if all_match else f'LOSSY - max pixel diff = {max_diff}'}")
    print(f"  Encode: {t_encode:.1f}s | Decode: {t_decode:.1f}s")
    print(f"  Raw: {raw_size:,} bytes | NEXUS: {nexus_size:,} bytes | "
          f"Ratio: {raw_size / nexus_size:.2f}x")
    print(f"  vs H.264 ({original_size:,} bytes): "
          f"{nexus_size / original_size * 100:.1f}%")

    # Verify audio
    if audio is not None and "audio_samples" in decoded_meta:
        audio_match = np.array_equal(audio[:len(decoded_meta["audio_samples"])],
                                      decoded_meta["audio_samples"])
        print(f"  Audio: {'LOSSLESS MATCH' if audio_match else 'MISMATCH'}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run benchmark: compress at multiple scales, report ratios."""
    print(f"NEXUS Codec - Benchmark")
    print(f"  Input: {args.input}")

    scales = [0.125, 0.25, 0.5]
    frame_counts = [args.frames or 50]

    print(f"\n  {'Scale':>6} {'Dims':>12} {'Frames':>7} {'Raw':>12} "
          f"{'NEXUS':>12} {'Ratio':>8} {'Enc fps':>8}")
    print("  " + "-" * 75)

    for scale in scales:
        for n_frames in frame_counts:
            frames, meta, audio = extract_frames(args.input, n_frames, scale)
            w, h = meta["scaled_width"], meta["scaled_height"]
            raw = sum(f.nbytes for f in frames)
            if audio is not None:
                raw += audio.nbytes

            t0 = time.time()
            encoded = encode_video(
                frames, w, h, fps=meta["fps"],
                audio_samples=audio,
                audio_rate=meta["audio_rate"],
                audio_channels=meta["audio_channels"],
            )
            t_enc = time.time() - t0
            nxv = len(encoded)
            enc_fps = len(frames) / max(t_enc, 0.01)

            print(f"  {scale:>6.3f} {w:>5}x{h:<5} {len(frames):>7} "
                  f"{raw:>12,} {nxv:>12,} {raw / nxv:>7.2f}x {enc_fps:>7.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NEXUS Codec - Novel from-scratch media compressor")
    sub = parser.add_subparsers(dest="command", required=True)

    p_comp = sub.add_parser("compress", help="Compress video to .nxv")
    p_comp.add_argument("input", help="Input video file")
    p_comp.add_argument("output", help="Output .nxv file")
    p_comp.add_argument("--frames", type=int, help="Max frames to encode")
    p_comp.add_argument("--scale", type=float, default=0.25, help="Resolution scale")

    p_dec = sub.add_parser("decompress", help="Decompress .nxv to frames")
    p_dec.add_argument("input", help="Input .nxv file")
    p_dec.add_argument("output", help="Output directory for frames")

    p_ver = sub.add_parser("verify", help="Verify lossless round-trip")
    p_ver.add_argument("input", help="Input video file")
    p_ver.add_argument("--frames", type=int, default=30, help="Frames to test")
    p_ver.add_argument("--scale", type=float, default=0.25, help="Resolution scale")

    p_bench = sub.add_parser("benchmark", help="Run compression benchmark")
    p_bench.add_argument("input", help="Input video file")
    p_bench.add_argument("--frames", type=int, default=50, help="Frames per test")
    p_bench.add_argument("--scale", type=float, default=0.25, help="Resolution scale")

    args = parser.parse_args()
    {"compress": cmd_compress, "decompress": cmd_decompress,
     "verify": cmd_verify, "benchmark": cmd_benchmark}[args.command](args)


if __name__ == "__main__":
    main()
