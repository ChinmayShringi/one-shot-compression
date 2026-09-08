#!/usr/bin/env python3
"""Pure-encode checker for committed Family N2 rasters.

Reads experiments/paint_list_n2/fixtures/img_XX.rgb only. Does not import a
generator. Does not read a seed. Does not hardcode a payload byte table.
Does not regenerate fixtures.

Prints committed fixture hashes and redraw hashes. Exits non-zero if they
differ, if encode cannot recover a program, or if the encode module's
import graph names a generator.

    python3 experiments/paint_list_n2_encode/check_encode.py
"""

from __future__ import annotations

import ast
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIXTURE_DIR = os.path.join(REPO, "experiments", "paint_list_n2", "fixtures")
PROGRAM_DIR = os.path.join(HERE, "programs")
N_IMAGES = 24
FIT_COUNT = 16
WIDTH = 64
HEIGHT = 64
FIXTURE_BYTES = WIDTH * HEIGHT * 3
FIXTURE_COMMIT = "b59041517f5756bb0950f9d1441760b72249fc52"

if HERE not in sys.path:
    sys.path.insert(0, HERE)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def fixture_path(index):
    return os.path.join(FIXTURE_DIR, "img_%02d.rgb" % index)


def program_path(index):
    return os.path.join(PROGRAM_DIR, "img_%02d.pln" % index)


def split_name(index):
    return "fit" if index < FIT_COUNT else "heldout"


def _module_names(node):
    names = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            names.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
        if node.module:
            names.append(node.module)
        for alias in node.names:
            names.append(alias.name)
    return names


def forbidden_import(name):
    leaf = name.split(".")[-1]
    if leaf == "generator_n2" or name.endswith(".generator_n2"):
        return True
    if leaf == "generator" or name.endswith(".generator") or ".generator." in name:
        return True
    if leaf.startswith("generator_") or name.startswith("generator_"):
        return True
    return False


def scan_source(path, seen, hits):
    rel = os.path.relpath(path, REPO)
    if rel in seen:
        return
    seen.add(rel)
    text = open(path, "r", encoding="utf-8").read()
    # Source-text scan only. This function does not import the scanned module.
    if "generator_n2" in text:
        hits.append("%s contains generator_n2" % rel)
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as exc:
        hits.append("%s syntax %s" % (rel, exc))
        return
    local_deps = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for name in _module_names(node):
            if forbidden_import(name):
                hits.append("%s imports %s" % (rel, name))
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            leaf = node.module.split(".")[-1]
            sibling = os.path.join(os.path.dirname(path), leaf + ".py")
            if os.path.isfile(sibling):
                local_deps.append(sibling)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                leaf = alias.name.split(".")[-1]
                sibling = os.path.join(os.path.dirname(path), leaf + ".py")
                if os.path.isfile(sibling):
                    local_deps.append(sibling)
    for dep in local_deps:
        scan_source(dep, seen, hits)


def import_graph_scan():
    encode_py = os.path.join(HERE, "encode.py")
    seen = set()
    hits = []
    scan_source(encode_py, seen, hits)
    return seen, hits


def main():
    print("checker fixture_commit", FIXTURE_COMMIT)
    print("checker fixture_dir", os.path.relpath(FIXTURE_DIR, REPO))
    print("checker does_not_import_generator")
    print("checker does_not_hardcode_byte_table")
    print("checker encodes_from_committed_rasters_only")

    seen, hits = import_graph_scan()
    print("import_graph_modules", " ".join(sorted(seen)))
    if hits:
        print("import_graph", "fail")
        for item in hits:
            print("import_graph_hit", item)
        print("representation_stops", 1)
        print("reason", "encode module imports a generator")
        print("FAIL")
        return 1
    print("import_graph", "clean")
    print("import_graph_result", "encode.py does not import generator_n2 or any generator")

    import baselines
    import encode

    if encode.encode.__code__.co_argcount != 3:
        print("FAIL encode arity", encode.encode.__code__.co_varnames[: encode.encode.__code__.co_argcount])
        return 1
    if tuple(encode.encode.__code__.co_varnames[:3]) != ("px", "width", "height"):
        print("FAIL encode signature", encode.encode.__code__.co_varnames[:3])
        return 1
    frozen = {
        "ELLIPSE_MARGIN": 4,
        "TRI_LO": -2,
        "TRI_HI": 6,
        "LINE_RADIUS": 8,
    }
    for name, value in frozen.items():
        if getattr(encode, name) != value:
            print("FAIL frozen constant", name)
            return 1
    print("jpegxl_effort", baselines.JXL_EFFORT)
    if baselines.JXL_EFFORT != 9:
        print("FAIL jpegxl effort is not 9")
        return 1

    os.makedirs(PROGRAM_DIR, exist_ok=True)
    fail = []
    rows = []
    for index in range(N_IMAGES):
        split = split_name(index)
        fpath = fixture_path(index)
        if not os.path.isfile(fpath):
            print("image", index, split, "FAIL missing committed fixture")
            fail.append("missing fixture %d" % index)
            continue
        image = open(fpath, "rb").read()
        image_hash = sha256_bytes(image)
        if len(image) != FIXTURE_BYTES:
            print(
                "image",
                index,
                split,
                "src_sha256",
                image_hash,
                "FAIL fixture size",
                len(image),
            )
            fail.append("fixture size %d" % index)
            continue
        payload = encode.encode(image, WIDTH, HEIGHT)
        if payload is None:
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
                "recover",
                0,
            )
            fail.append("unrecovered %d" % index)
            ppath = program_path(index)
            if os.path.exists(ppath):
                os.remove(ppath)
            continue
        ppath = program_path(index)
        with open(ppath, "wb") as fh:
            fh.write(payload)
        stored = open(ppath, "rb").read()
        if stored != payload:
            fail.append("stored mismatch %d" % index)
        try:
            redraw = encode.decode(stored)
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
                len(stored),
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
            len(stored),
            "png_bytes",
            len(png),
            "jxl_e9_bytes",
            len(jxl),
        )
        if not match:
            fail.append("hash mismatch %d" % index)
        rows.append(
            {
                "index": index,
                "split": split,
                "match": match,
                "program": len(stored),
                "png": len(png),
                "jxl": len(jxl),
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
    )
    smaller = held["n"] == 8 and held["program"] < held["jxl"]
    all_exact = held["exact"] == 8 and held["n"] == 8
    hash_ok = not fail and all(r["match"] for r in rows) and len(rows) == N_IMAGES
    recovered = held["n"] == 8 and fit["n"] == 16
    if not recovered:
        print("representation", "recipe_replay")
        print("representation_stops", 1)
        print("reason", "encoder cannot recover a program without the generator")
    advance = "pass" if (all_exact and smaller and hash_ok and recovered) else "fail"
    print("heldout_exact", "%d/8" % held["exact"])
    print("heldout_program_bytes", held["program"])
    print("heldout_png_bytes", held["png"])
    print("heldout_jxl_e9_bytes", held["jxl"])
    print("program_smaller_than_jxl_e9", int(smaller))
    print("checker_hash_match", int(hash_ok))
    print("advance", advance)
    if advance != "pass":
        print("representation_stops", 1)
    else:
        print("representation_stops", 0)
    if fail or not hash_ok:
        print("FAIL")
        for item in fail:
            print("error", item)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
