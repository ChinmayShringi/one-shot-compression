#!/usr/bin/env python3
"""Second checker. Does not read results.json.

Rebuilds each source image, prints sha256 of its RGB bytes, encodes a paint
list, PNG, and JPEG XL effort 9 itself, and compares the held-out table to the
numbers published in the PR body. Exit is non-zero if they disagree.

    python3 experiments/paint_list/check_published.py
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
import generator_n
import generator_o
from raster import HEIGHT, WIDTH

# Same numbers as the PR body. Not loaded from results.json.
PUBLISHED = {
    "family_n_heldout_program_bytes": 364,
    "family_n_heldout_png_bytes": 2614,
    "family_n_heldout_jxl_bytes": 938,
    "family_n_heldout_exact": 8,
    "family_n_heldout_program_missing": 0,
    "family_o_heldout_program_bytes": 39,
    "family_o_heldout_png_bytes": 2294,
    "family_o_heldout_jxl_bytes": 791,
    "family_o_heldout_exact": 2,
    "family_o_heldout_program_missing": 6,
    "advance_bar": "pass",
    "shared_decoder_bytes": 18833,
    "jpegxl_effort": 9,
    "family_n_seed": 20260908,
    "family_o_seed": 20260908,
}

# Held-out program/png/jxl byte lengths. program is None when no exact program is stored.
# These cells are the PR held-out table, one row per image 16-23.
PUBLISHED_ROWS = {
    "N": {
        16: (64, 384, 149),
        17: (33, 263, 91),
        18: (24, 166, 45),
        19: (53, 361, 131),
        20: (44, 455, 153),
        21: (71, 432, 169),
        22: (42, 362, 136),
        23: (33, 191, 64),
    },
    "O": {
        16: (None, 309, 120),
        17: (None, 414, 151),
        18: (None, 173, 51),
        19: (15, 139, 38),
        20: (None, 466, 168),
        21: (None, 348, 114),
        22: (None, 281, 99),
        23: (24, 164, 50),
    },
}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def measure_one(px):
    payload = codec.encode(px, WIDTH, HEIGHT)
    if payload is None:
        prog_len = None
        exact = False
    else:
        decoded = codec.decode(payload)
        exact = decoded == px
        prog_len = len(payload) if exact else None
        if not exact:
            # A stored inexact program is not counted. Record the disagreement.
            exact = False
            prog_len = None
    png = baselines.encode_png(px, WIDTH, HEIGHT)
    png_px, w, h = baselines.decode_png(png)
    if w != WIDTH or h != HEIGHT or png_px != px:
        raise RuntimeError("PNG baseline not exact")
    jxl = baselines.encode_jxl(px, WIDTH, HEIGHT)
    jxl_px, w, h = baselines.decode_jxl(jxl)
    if w != WIDTH or h != HEIGHT or jxl_px != px:
        raise RuntimeError("JPEG XL effort 9 baseline not exact")
    return prog_len, exact, len(png), len(jxl)


def main():
    print("checker does_not_read results.json")
    print("jpegxl_effort", baselines.JXL_EFFORT)
    if baselines.JXL_EFFORT != 9:
        print("FAIL jpegxl effort is not 9")
        return 1
    disagree = []
    if baselines.JXL_EFFORT != PUBLISHED["jpegxl_effort"]:
        disagree.append("jpegxl_effort")

    decoder = 0
    for name in ("raster.py", "codec.py"):
        n = os.path.getsize(os.path.join(HERE, name))
        print("decoder_file", name, n)
        decoder += n
    print("shared_decoder_bytes", decoder)
    if decoder != PUBLISHED["shared_decoder_bytes"]:
        disagree.append("shared_decoder_bytes measured=%s published=%s" % (decoder, PUBLISHED["shared_decoder_bytes"]))

    measured = {}
    for fam, gen in (("N", generator_n), ("O", generator_o)):
        print("family", fam, "seed", gen.MASTER_SEED)
        if gen.MASTER_SEED != PUBLISHED["family_%s_seed" % fam.lower()]:
            disagree.append("seed %s" % fam)
        held_prog = 0
        held_png = 0
        held_jxl = 0
        held_exact = 0
        held_missing = 0
        for index in range(gen.N_IMAGES):
            scene = gen.generate_scene(index, gen.MASTER_SEED)
            px = gen.render(scene)
            digest = sha256_bytes(px)
            prog_len, exact, png_len, jxl_len = measure_one(px)
            split = gen.split_name(index)
            print(
                "image",
                fam,
                index,
                split,
                "src_sha256",
                digest,
                "program_bytes",
                "missing" if prog_len is None else prog_len,
                "exact",
                int(exact),
                "png_bytes",
                png_len,
                "jxl_e9_bytes",
                jxl_len,
            )
            if split != "heldout":
                continue
            if prog_len is None:
                held_missing += 1
            else:
                held_prog += prog_len
            if exact:
                held_exact += 1
            held_png += png_len
            held_jxl += jxl_len
            expected = PUBLISHED_ROWS[fam][index]
            got = (prog_len, png_len, jxl_len)
            if got != expected:
                disagree.append("row %s %s measured=%s published=%s" % (fam, index, got, expected))
        measured[fam] = {
            "program_bytes": held_prog,
            "png_bytes": held_png,
            "jxl_bytes": held_jxl,
            "exact": held_exact,
            "missing": held_missing,
        }
        print(
            "heldout_table",
            fam,
            "program",
            held_prog,
            "png",
            held_png,
            "jxl_e9",
            held_jxl,
            "exact",
            "%d/8" % held_exact,
            "program_missing",
            held_missing,
        )

    checks = [
        ("family_n_heldout_program_bytes", measured["N"]["program_bytes"]),
        ("family_n_heldout_png_bytes", measured["N"]["png_bytes"]),
        ("family_n_heldout_jxl_bytes", measured["N"]["jxl_bytes"]),
        ("family_n_heldout_exact", measured["N"]["exact"]),
        ("family_n_heldout_program_missing", measured["N"]["missing"]),
        ("family_o_heldout_program_bytes", measured["O"]["program_bytes"]),
        ("family_o_heldout_png_bytes", measured["O"]["png_bytes"]),
        ("family_o_heldout_jxl_bytes", measured["O"]["jxl_bytes"]),
        ("family_o_heldout_exact", measured["O"]["exact"]),
        ("family_o_heldout_program_missing", measured["O"]["missing"]),
    ]
    for key, value in checks:
        print("compare", key, "measured", value, "published", PUBLISHED[key])
        if value != PUBLISHED[key]:
            disagree.append("%s measured=%s published=%s" % (key, value, PUBLISHED[key]))

    n_pass = (
        measured["N"]["exact"] == 8
        and measured["N"]["missing"] == 0
        and measured["N"]["program_bytes"] < measured["N"]["jxl_bytes"]
    )
    bar = "pass" if n_pass else "fail"
    print("advance_bar", bar)
    print("published_advance_bar", PUBLISHED["advance_bar"])
    if bar != PUBLISHED["advance_bar"]:
        disagree.append("advance_bar measured=%s published=%s" % (bar, PUBLISHED["advance_bar"]))

    if disagree:
        print("FAIL")
        for item in disagree:
            print("disagree", item)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
