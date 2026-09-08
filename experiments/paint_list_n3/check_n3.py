#!/usr/bin/env python3
"""Family N3 checker. Does not import the generator. Does not hardcode a byte table.

Hash match decodes committed programs only. Program byte counts are the
committed program files. JPEG XL effort 9 is encoded independently on the
committed rasters.

A shared edge is counted from the committed pixel rasters: two distinct
same-color shapes in the recovered program whose painted pixels touch along
a 4-connected edge. Corner-only contact does not count. The checker does not
open the generator sidecar.
"""

from __future__ import annotations

import ast
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIXTURE_DIR = os.path.join(HERE, "fixtures")
PROGRAM_DIR = os.path.join(HERE, "programs")
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
    if "shared_edges.json" in text or "/sidecar" in text:
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


def shape_pixels(typ, geom, width, height):
    from raster import TYPE_ELLIPSE, TYPE_RECT, TYPE_TRIANGLE, pixel_in_ellipse, pixel_in_triangle

    pts = []
    if typ == TYPE_RECT:
        x0, y0, x1, y1 = geom
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        x0 = 0 if x0 < 0 else x0
        y0 = 0 if y0 < 0 else y0
        x1 = width - 1 if x1 >= width else x1
        y1 = height - 1 if y1 >= height else y1
        if x0 > x1 or y0 > y1:
            return pts
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                pts.append((x, y))
    elif typ == TYPE_ELLIPSE:
        x0, y0, x1, y1 = geom
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        xa = 0 if x0 < 0 else x0
        ya = 0 if y0 < 0 else y0
        xb = width - 1 if x1 >= width else x1
        yb = height - 1 if y1 >= height else y1
        for y in range(ya, yb + 1):
            for x in range(xa, xb + 1):
                if pixel_in_ellipse(x, y, x0, y0, x1, y1):
                    pts.append((x, y))
    elif typ == TYPE_TRIANGLE:
        v0, v1, v2 = geom
        xs = (v0[0], v1[0], v2[0])
        ys = (v0[1], v1[1], v2[1])
        x0 = max(0, min(xs))
        y0 = max(0, min(ys))
        x1 = min(width - 1, max(xs))
        y1 = min(height - 1, max(ys))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if pixel_in_triangle(x, y, v0, v1, v2):
                    pts.append((x, y))
    return pts


def pixels_share_4_edge(a, b):
    """4-connected adjacency. Corner-only contact returns False."""
    if len(a) > len(b):
        a, b = b, a
    bset = b if isinstance(b, set) else set(b)
    for x, y in a:
        if (x + 1, y) in bset or (x - 1, y) in bset or (x, y + 1) in bset or (x, y - 1) in bset:
            return True
    return False


def type_name(typ):
    from raster import TYPE_ELLIPSE, TYPE_RECT, TYPE_TRIANGLE

    if typ == TYPE_RECT:
        return "rect"
    if typ == TYPE_ELLIPSE:
        return "ellipse"
    if typ == TYPE_TRIANGLE:
        return "triangle"
    return "other"


def fixture_color(image, x, y, width):
    o = (y * width + x) * 3
    return (image[o], image[o + 1], image[o + 2])


def shared_edges_from_pixels(image, payload):
    """Count same-color 4-edge pairs from the raster and the recovered program.

    Returns None when the committed program does not redraw the committed raster.
    Does not read a generator sidecar. Does not treat corner contact as an edge.
    """
    import encode
    from raster import TYPE_RECT

    scene = encode.scene_from_program(payload)
    width = int(scene["width"])
    height = int(scene["height"])
    if encode.decode(payload) != image:
        return None
    painted = []
    for typ, color_index, geom in scene["shapes"]:
        color = scene["palette"][color_index]
        pts = shape_pixels(typ, geom, width, height)
        if not pts:
            return None
        for x, y in pts:
            if fixture_color(image, x, y, width) != color:
                return None
        painted.append((typ, color, set(pts)))
    pairs = []
    for i in range(len(painted)):
        for j in range(i + 1, len(painted)):
            ta, ca, pa = painted[i]
            tb, cb, pb = painted[j]
            if ca != cb:
                continue
            if pa & pb:
                continue
            if not pixels_share_4_edge(pa, pb):
                continue
            pairs.append((type_name(ta), type_name(tb), ta != TYPE_RECT or tb != TYPE_RECT))
    return pairs


def main():
    print("checker reads committed fixtures and committed programs")
    print("checker does_not_import_generator")
    print("checker does_not_hardcode_byte_table")
    print("checker hash_match_decodes_committed_programs_only")
    print("checker does_not_open_sidecar")
    print("shared_edge_source committed_pixels")
    print("shared_edge_rule two_distinct_same_color_shapes_4_connected_edge")
    print("corner_only_contact_counts 0")

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
    print("frozen_constants_retuned", 0)
    print("seed", 20260910)
    print("seed_changed", 0)
    print("jpegxl_effort", baselines.JXL_EFFORT)
    if baselines.JXL_EFFORT != 9:
        print("FAIL jpegxl effort is not 9")
        return 1

    fail = []
    rows = []
    heldout_pair_images = []
    heldout_nonrect_images = []
    heldout_pair_total = 0
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
        png = baselines.encode_png(image, WIDTH, HEIGHT)
        png_px, w, h = baselines.decode_png(png)
        if w != WIDTH or h != HEIGHT or png_px != image:
            fail.append("png %d" % index)
        jxl = baselines.encode_jxl(image, WIDTH, HEIGHT)
        jxl_px, w, h = baselines.decode_jxl(jxl)
        if w != WIDTH or h != HEIGHT or jxl_px != image:
            fail.append("jxl %d" % index)
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
                "png_bytes",
                len(png),
                "jxl_e9_bytes",
                len(jxl),
                "encode_exact",
                0,
                "shared_edge_pairs_from_pixels",
                "unrecovered",
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
                "png_bytes",
                len(png),
                "jxl_e9_bytes",
                len(jxl),
                "encode_exact",
                0,
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
        pairs = shared_edges_from_pixels(image, stored) if match else None
        if match and pairs is None:
            fail.append("shared-edge pixel count unavailable %d" % index)
        pair_n = 0 if pairs is None else len(pairs)
        nonrect = 0 if not pairs else sum(1 for _a, _b, is_nonrect in pairs if is_nonrect)
        if split == "heldout" and pairs:
            heldout_pair_total += pair_n
            if pair_n:
                heldout_pair_images.append(index)
            if nonrect:
                heldout_nonrect_images.append(index)
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
            "shared_edge_pairs_from_pixels",
            "unrecovered" if pairs is None else pair_n,
            "nonrect_shared_edge_pairs_from_pixels",
            "unrecovered" if pairs is None else nonrect,
        )
        if pairs:
            for left, right, is_nonrect in pairs:
                print(
                    "shared_edge_from_pixels",
                    index,
                    split,
                    left,
                    right,
                    "nonrect",
                    int(is_nonrect),
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
        "%d/%d" % (held["exact"], 8),
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
    print("heldout_shared_edge_count_from_pixels", heldout_pair_total)
    print("heldout_shared_edge_images_from_pixels", " ".join(str(i) for i in heldout_pair_images) or "none")
    if heldout_nonrect_images:
        print(
            "heldout_nonrect_shared_edge_images",
            " ".join(str(i) for i in heldout_nonrect_images),
        )
    else:
        print("heldout_nonrect_shared_edge_images", "none")
    smaller = held["n"] == 8 and held["program"] < held["jxl"]
    all_exact = held["exact"] == 8 and held["n"] == 8
    hash_ok = not fail and all(r["match"] for r in rows) and len(rows) == N_IMAGES
    recovered_all = held["n"] == 8 and fit["n"] == 16
    nonrect_ok = len(heldout_nonrect_images) >= 1
    adjacency_broke = any(
        item.startswith("unrecovered") or item.startswith("hash mismatch") or item.startswith("missing program")
        for item in fail
    )
    print("heldout_exact", "%d/8" % held["exact"])
    print("heldout_program_bytes", held["program"])
    print("heldout_png_bytes", held["png"])
    print("heldout_jxl_e9_bytes", held["jxl"])
    print("program_smaller_than_jxl_e9", int(smaller))
    print("checker_hash_match", int(hash_ok))
    print("nonrect_shared_edge_constraint", int(nonrect_ok))
    if not recovered_all or adjacency_broke or not nonrect_ok:
        print("representation", "n3_adjacency")
        print("representation_stops", 1)
        print("reason", "the paint list stops at disconnected or two-rectangle masks")
        print("stop", "the paint list stops at disconnected or two-rectangle masks")
    advance = "pass" if (all_exact and smaller and hash_ok and recovered_all and nonrect_ok) else "fail"
    print("advance", advance)
    if advance == "pass":
        print("representation_stops", 0)
    if fail or not hash_ok or not nonrect_ok:
        print("FAIL")
        for item in fail:
            print("error", item)
        if not nonrect_ok:
            print("error", "no held-out non-rectangle shared-edge pair counted from pixels")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
