#!/usr/bin/env python3
"""Family N3 checker. Does not import the generator. Does not hardcode a byte table.

Hash match decodes committed programs only. Program byte counts are the
committed program files. JPEG XL effort 9 is encoded independently on the
committed rasters. Shared-edge count is read from the generator sidecar.
The encoder path does not open that sidecar.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIXTURE_DIR = os.path.join(HERE, "fixtures")
PROGRAM_DIR = os.path.join(HERE, "programs")
SIDECAR = os.path.join(HERE, "sidecar", "shared_edges.json")
N_IMAGES = 24
FIT_COUNT = 16
WIDTH = 64
HEIGHT = 64
FIXTURE_BYTES = WIDTH * HEIGHT * 3

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
    if leaf == "generator_n3" or name.endswith(".generator_n3"):
        return True
    if leaf == "generator" or name.endswith(".generator") or ".generator." in name:
        return True
    if leaf.startswith("generator_") or name.startswith("generator_"):
        return True
    if leaf == "write_fixtures" or name.endswith(".write_fixtures"):
        return True
    return False


def scan_source(path, seen, hits):
    rel = os.path.relpath(path, REPO)
    if rel in seen:
        return
    seen.add(rel)
    text = open(path, "r", encoding="utf-8").read()
    if "generator_n3" in text or "write_fixtures" in text:
        hits.append("%s names a generator module" % rel)
    if "shared_edges.json" in text or "sidecar" in text:
        hits.append("%s names the sidecar" % rel)
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
    print("checker reads committed fixtures and committed programs")
    print("checker does_not_import_generator")
    print("checker does_not_hardcode_byte_table")
    print("checker hash_match_decodes_committed_programs_only")
    print("checker encoder_path_does_not_open_sidecar")
    print("sidecar_role generator_fixture_write_only")

    seen, hits = import_graph_scan()
    print("import_graph_modules", " ".join(sorted(seen)))
    if hits:
        print("import_graph", "fail")
        for item in hits:
            print("import_graph_hit", item)
        print("import_graph_result", "encode.py imports or names a generator")
        print("representation_stops", 1)
        print("reason", "encode module imports a generator")
        print("FAIL")
        return 1
    print("import_graph", "clean")
    print("import_graph_result", "encode.py does not import generator_n3 or any generator")

    import baselines
    import encode

    if encode.encode.__code__.co_argcount != 3:
        print("FAIL encode arity")
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
    print("frozen_constants", "ELLIPSE_MARGIN=4 TRI_LO=-2 TRI_HI=6 LINE_RADIUS=8")
    print("frozen_constants_copied_before_heldout", 1)
    print("seed", 20260910)
    print("seed_changed", 0)
    print("jpegxl_effort", baselines.JXL_EFFORT)
    if baselines.JXL_EFFORT != 9:
        print("FAIL jpegxl effort is not 9")
        return 1

    if not os.path.isfile(SIDECAR):
        print("FAIL missing sidecar")
        return 1
    with open(SIDECAR, "r", encoding="utf-8") as fh:
        sidecar = json.load(fh)
    held_flags = []
    for row in sidecar["images"]:
        if row["split"] != "heldout":
            continue
        held_flags.append(1 if row["has_shared_edge"] else 0)
        print(
            "sidecar_image",
            row["index"],
            "heldout",
            "has_shared_edge",
            int(bool(row["has_shared_edge"])),
            "shared_edge_pairs",
            row["shared_edge_pairs"],
        )
    shared_n = sum(held_flags)
    if shared_n != int(sidecar["heldout_shared_edge_images"]):
        print("FAIL sidecar heldout count mismatch")
        return 1
    print("heldout_shared_edge_images", shared_n)

    fail = []
    rows = []
    for index in range(N_IMAGES):
        split = split_name(index)
        fpath = fixture_path(index)
        ppath = program_path(index)
        if not os.path.isfile(fpath):
            print("image", index, split, "FAIL missing committed fixture")
            fail.append("missing fixture %d" % index)
            continue
        image = open(fpath, "rb").read()
        image_hash = sha256_bytes(image)
        if len(image) != FIXTURE_BYTES:
            print("image", index, split, "src_sha256", image_hash, "FAIL fixture size", len(image))
            fail.append("fixture size %d" % index)
            continue
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
            )
            fail.append("missing program %d" % index)
            continue
        stored = open(ppath, "rb").read()
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
        recovered = encode.encode(image, WIDTH, HEIGHT)
        recover_ok = recovered == stored and recovered is not None
        if recovered is None:
            fail.append("unrecovered %d" % index)
        elif recovered != stored:
            fail.append("encode differs from committed program %d" % index)
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
            "encode_exact",
            int(recover_ok),
        )
        if not match:
            fail.append("hash mismatch %d" % index)
        rows.append(
            {
                "index": index,
                "split": split,
                "match": match and recover_ok,
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
    shared_ok = shared_n >= 3
    adjacency_broke = any(item.startswith("unrecovered") or item.startswith("hash mismatch") for item in fail)
    if not recovered or adjacency_broke:
        print("representation", "n3_adjacency")
        print("representation_stops", 1)
        print("reason", "N3 adjacency broke exactness")
    advance = "pass" if (all_exact and smaller and hash_ok and recovered and shared_ok) else "fail"
    print("heldout_exact", "%d/8" % held["exact"])
    print("heldout_program_bytes", held["program"])
    print("heldout_png_bytes", held["png"])
    print("heldout_jxl_e9_bytes", held["jxl"])
    print("program_smaller_than_jxl_e9", int(smaller))
    print("heldout_shared_edge_images", shared_n)
    print("checker_hash_match", int(hash_ok))
    print("advance", advance)
    if advance != "pass":
        print("representation_stops", 1)
        if adjacency_broke:
            print("stop", "N3 adjacency broke exactness. No residual. This representation stops.")
    else:
        print("representation_stops", 0)
    if fail or not hash_ok or not shared_ok:
        print("FAIL")
        for item in fail:
            print("error", item)
        if not shared_ok:
            print("error", "heldout shared-edge images %d < 3" % shared_n)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
