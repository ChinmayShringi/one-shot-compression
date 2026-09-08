#!/usr/bin/env python3
"""Independent exactness checker for the frozen shape family.

Run this from a checkout of commit d287e17a10bdb5c2613fc72bc789a029d3f35a56
(PR 5, branch exp/shape-family-abc). It does not read results.json and does
not trust the experiment runner. It regenerates the 24 images from the
recorded seed, encodes and decodes with the codecs in this tree, and compares
pixels itself.

Exit 0 only if A and B decoded pixels match the regenerated source for every
image. That is this process's count, not a claim copied from SUMMARY.md.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# This file is meant to live next to the codecs, or one directory up.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from codec_a import decode as decode_a
from codec_a import encode as encode_a
from codec_b import decode as decode_b
from codec_b import encode as encode_b
from generator import FIT_COUNT, MASTER_SEED, N_IMAGES, generate_scene, render

TARGET_COMMIT = "d287e17a10bdb5c2613fc72bc789a029d3f35a56"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mismatch_count(a: bytes, b: bytes) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    n = 0
    for i in range(0, len(a), 3):
        if a[i : i + 3] != b[i : i + 3]:
            n += 1
    return n


def main() -> int:
    print("target_commit", TARGET_COMMIT)
    print("seed", MASTER_SEED)
    print("note does_not_read results.json")
    failed = 0
    for index in range(N_IMAGES):
        scene = generate_scene(index, MASTER_SEED)
        source = render(scene)
        src_hash = sha256_bytes(source)
        split = "fit" if index < FIT_COUNT else "heldout"

        a_payload = encode_a(scene)
        a_out = decode_a(a_payload)
        a_mis = mismatch_count(source, a_out)

        b_payload, _info = encode_b(source, scene["width"], scene["height"])
        b_out = decode_b(b_payload)
        b_mis = mismatch_count(source, b_out)

        if a_mis or b_mis:
            failed += 1
        print(
            "image",
            index,
            split,
            "src_sha256",
            src_hash,
            "a_mismatch_px",
            a_mis,
            "b_mismatch_px",
            b_mis,
            "a_payload",
            len(a_payload),
            "b_payload",
            len(b_payload),
        )
    print("failed_images", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
