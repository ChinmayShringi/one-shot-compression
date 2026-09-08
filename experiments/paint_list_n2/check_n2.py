#!/usr/bin/env python3
"""Family N2 checker. Not an answer key.

Hashes committed fixture bytes. Decodes committed programs only. Does not
re-encode to prove exactness. Does not read results.json. Does not hardcode
a payload byte table. Counts program, PNG, and JPEG XL effort 9 bytes itself.

    python3 experiments/paint_list_n2/check_n2.py
"""

from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import baselines
import codec
from raster import HEIGHT, WIDTH

N_IMAGES = 24
FIT_COUNT = 16
FIXTURE_BYTES = WIDTH * HEIGHT * 3


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def fixture_path(index):
    return os.path.join(HERE, "fixtures", "img_%02d.rgb" % index)


def program_path(index):
    return os.path.join(HERE, "programs", "img_%02d.pln" % index)


def split_name(index):
    return "fit" if index < FIT_COUNT else "heldout"


def main():
    print("checker does_not_read results.json")
    print("checker does_not_reencode")
    print("checker hashes committed image bytes")
    print("jpegxl_effort", baselines.JXL_EFFORT)
    if baselines.JXL_EFFORT != 9:
        print("FAIL jpegxl effort is not 9")
        return 1

    decoder = 0
    for name in ("raster.py", "codec.py"):
        n = os.path.getsize(os.path.join(HERE, name))
        print("decoder_file", name, n)
        decoder += n
    print("shared_decoder_bytes", decoder)
    print("shared_decoder_not_in_payload", 1)

    fail = []
    rows = []
    for index in range(N_IMAGES):
        split = split_name(index)
        fpath = fixture_path(index)
        if not os.path.isfile(fpath):
            print("image", index, split, "FAIL missing fixture")
            fail.append("missing fixture %d" % index)
            continue
        image = open(fpath, "rb").read()
        image_hash = sha256_bytes(image)
        if len(image) != FIXTURE_BYTES:
            print("image", index, split, "src_sha256", image_hash, "FAIL fixture size", len(image))
            fail.append("fixture size %d" % index)
            continue
        ppath = program_path(index)
        if not os.path.isfile(ppath):
            print(
                "image",
                index,
                split,
                "src_sha256",
                image_hash,
                "redraw_sha256",
                "missing",
                "hash_match",
                0,
                "program_bytes",
                "missing",
            )
            fail.append("missing program %d" % index)
            continue
        payload = open(ppath, "rb").read()
        try:
            redraw = codec.decode(payload)
        except (ValueError, IndexError) as exc:
            print(
                "image",
                index,
                split,
                "src_sha256",
                image_hash,
                "redraw_sha256",
                "decode_error",
                "hash_match",
                0,
                "program_bytes",
                len(payload),
            )
            fail.append("decode %d %s" % (index, type(exc).__name__))
            continue
        redraw_bytes = bytes(redraw)
        redraw_hash = sha256_bytes(redraw_bytes)
        match = redraw_bytes == image and redraw_hash == image_hash
        png = baselines.encode_png(image, WIDTH, HEIGHT)
        png_px, w, h = baselines.decode_png(png)
        if w != WIDTH or h != HEIGHT or png_px != image:
            fail.append("png %d" % index)
        jxl = baselines.encode_jxl(image, WIDTH, HEIGHT)
        jxl_px, w, h = baselines.decode_jxl(jxl)
        if w != WIDTH or h != HEIGHT or jxl_px != image:
            fail.append("jxl %d" % index)
        scene = codec.scene_from_program(payload)
        counts = {}
        for _typ, color_index, _geom in scene["shapes"]:
            counts[color_index] = counts.get(color_index, 0) + 1
        reuse = any(n >= 2 for n in counts.values())
        print(
            "image",
            index,
            split,
            "src_sha256",
            image_hash,
            "redraw_sha256",
            redraw_hash,
            "hash_match",
            int(match),
            "program_bytes",
            len(payload),
            "png_bytes",
            len(png),
            "jxl_e9_bytes",
            len(jxl),
            "reuse",
            int(reuse),
            "shapes",
            len(scene["shapes"]),
            "palette",
            len(scene["palette"]),
        )
        if not match:
            fail.append("hash mismatch %d" % index)
        rows.append(
            {
                "index": index,
                "split": split,
                "match": match,
                "program": len(payload),
                "png": len(png),
                "jxl": len(jxl),
                "reuse": reuse,
            }
        )

    def rollup(split):
        sub = [r for r in rows if r["split"] == split]
        return {
            "n": len(sub),
            "exact": sum(1 for r in sub if r["match"]),
            "program": sum(r["program"] for r in sub),
            "png": sum(r["png"] for r in sub),
            "jxl": sum(r["jxl"] for r in sub),
            "reuse": sum(1 for r in sub if r["reuse"]),
        }

    held = rollup("heldout")
    fit = rollup("fit")
    print(
        "heldout_table",
        "program",
        held["program"],
        "png",
        held["png"],
        "jxl_e9",
        held["jxl"],
        "exact",
        "%d/%d" % (held["exact"], held["n"]),
        "reuse_images",
        held["reuse"],
    )
    print(
        "fit_table",
        "program",
        fit["program"],
        "png",
        fit["png"],
        "jxl_e9",
        fit["jxl"],
        "exact",
        "%d/%d" % (fit["exact"], fit["n"]),
        "reuse_images",
        fit["reuse"],
    )
    smaller = held["program"] < held["jxl"] and held["n"] == 8
    all_exact = held["exact"] == 8 and held["n"] == 8
    hash_ok = not fail and all(r["match"] for r in rows) and len(rows) == N_IMAGES
    advance = "pass" if (all_exact and smaller and hash_ok) else "fail"
    print("heldout_exact", "%d/8" % held["exact"])
    print("heldout_program_bytes", held["program"])
    print("heldout_png_bytes", held["png"])
    print("heldout_jxl_e9_bytes", held["jxl"])
    print("heldout_reuse_images", held["reuse"])
    print("program_smaller_than_jxl_e9", int(smaller))
    print("checker_hash_match", int(hash_ok))
    print("advance", advance)
    if fail or not hash_ok:
        print("FAIL")
        for item in fail:
            print("error", item)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
