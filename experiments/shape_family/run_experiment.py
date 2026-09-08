#!/usr/bin/env python3
"""Frozen shape-family A/B/C experiment runner.

One command, from the repository root:
    python3 experiments/shape_family/run_experiment.py

Regenerates images from seed 20260908. Does not tune on held-out.
Does not drop images. Does not use the network at encode or decode.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import baselines
import codec_a
import codec_b
import codec_c
from generator import FIT_COUNT, MASTER_SEED, N_IMAGES, SETTINGS_ID, generate_scene, render, split_name
from raster import HEIGHT, WIDTH, mismatch_count, pixels_equal

try:
    from zoneinfo import ZoneInfo

    ET = ZoneInfo("America/New_York")
except Exception:
    ET = timezone(timedelta(hours=-4))

REGEN_COMMAND = "python3 experiments/shape_family/run_experiment.py"
DECODER_FILES = ("raster.py", "codec_a.py", "codec_b.py", "codec_c.py")
# generator.py synthesizes scenes. Stored A payloads already contain geometry,
# so decode does not import generator.py. Counted separately, not per image.
GENERATOR_FILE = "generator.py"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def now_utc():
    return datetime.now(timezone.utc)


def fmt_et(dt):
    return dt.astimezone(ET).strftime("%Y-%m-%d %H:%M:%S %Z")


def file_size(name):
    return os.path.getsize(os.path.join(HERE, name))


def shared_decoder_size():
    files = []
    total = 0
    for name in DECODER_FILES:
        n = file_size(name)
        files.append({"path": "experiments/shape_family/" + name, "bytes": n})
        total += n
    gen = file_size(GENERATOR_FILE)
    return {
        "bytes": total,
        "included": files,
        "excluded": [
            "experiments/shape_family/generator.py (scene synthesis; not used to decode stored payloads)",
            "experiments/shape_family/baselines.py",
            "experiments/shape_family/run_experiment.py",
            "experiments/shape_family/results.json",
            "experiments/shape_family/SUMMARY.md",
        ],
        "generator_source_bytes_not_in_decoder": gen,
        "note": (
            "Shared decoder size is the byte size of decoder source files, "
            "not stored in any per-image payload. Per-image payloads still "
            "include prompts/outlines/palettes/seeds/settings/correction where those fields exist."
        ),
    }


def timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - t0


def empty_codec():
    return {
        "exact": False,
        "mismatch_count": None,
        "payload_bytes": None,
        "encode_s": None,
        "decode_s": None,
        "error": None,
    }


def run_a(scene, px):
    payload, enc_s = timed(lambda: codec_a.encode(scene))
    decoded, dec_s = timed(lambda: codec_a.decode(payload))
    mis = mismatch_count(px, decoded)
    return {
        "exact": pixels_equal(px, decoded),
        "mismatch_count": mis,
        "payload_bytes": len(payload),
        "encode_s": enc_s,
        "decode_s": dec_s,
        "error": None,
        "correction_bytes": 0,
        "xor_corrected": False,
    }


def run_b(px):
    payload, enc_s = timed(lambda: codec_b.encode_with_breakdown(px, WIDTH, HEIGHT))
    blob, info = payload
    decoded, dec_s = timed(lambda: codec_b.decode(blob))
    mis = mismatch_count(px, decoded)
    return {
        "exact": pixels_equal(px, decoded),
        "mismatch_count": mis,
        "payload_bytes": len(blob),
        "encode_s": enc_s,
        "decode_s": dec_s,
        "error": None,
        "description_bytes": info["description_bytes"],
        "correction_bytes": info["correction_bytes"],
        "xor_corrected": True,
        "regions": info["n_regions"],
        "palette": info["n_palette"],
        "mismatch_pixels_before_correction": info["mismatch_pixels_before_correction"],
        "correction_method": info["correction_method"],
    }


def run_c(px):
    # Time the middle decode and the second encode separately. One cycle only.
    t0 = time.perf_counter()
    first, info1 = codec_b.encode_with_breakdown(px, WIDTH, HEIGHT)
    enc1_s = time.perf_counter() - t0
    decoded, dec_s = timed(lambda: codec_b.decode(first))
    second, enc2_s = timed(lambda: codec_b.encode_with_breakdown(decoded, WIDTH, HEIGHT))
    blob2, info2 = second
    mis = mismatch_count(px, decoded)
    return {
        "exact": pixels_equal(px, decoded),
        "mismatch_count": mis,
        "payload_bytes": len(blob2),
        "encode_s": enc2_s,
        "decode_s": dec_s,
        "error": None,
        "description_byte_identical": first == blob2,
        "first_payload_bytes": len(first),
        "second_payload_bytes": len(blob2),
        "first_encode_s": enc1_s,
        "correction_bytes": info2["correction_bytes"],
        "description_bytes": info2["description_bytes"],
        "xor_corrected": True,
    }


def run_png(px):
    payload, enc_s = timed(lambda: baselines.encode_png(px, WIDTH, HEIGHT))
    decoded_pack, dec_s = timed(lambda: baselines.decode_png(payload))
    decoded, w, h = decoded_pack
    if w != WIDTH or h != HEIGHT:
        raise ValueError("PNG size mismatch")
    mis = mismatch_count(px, decoded)
    return {
        "exact": pixels_equal(px, decoded),
        "mismatch_count": mis,
        "payload_bytes": len(payload),
        "encode_s": enc_s,
        "decode_s": dec_s,
        "error": None,
    }


def run_jxl(px):
    payload, enc_s = timed(lambda: baselines.encode_jxl(px, WIDTH, HEIGHT))
    decoded_pack, dec_s = timed(lambda: baselines.decode_jxl(payload))
    decoded, w, h = decoded_pack
    if w != WIDTH or h != HEIGHT:
        raise ValueError("JXL size mismatch")
    mis = mismatch_count(px, decoded)
    return {
        "exact": pixels_equal(px, decoded),
        "mismatch_count": mis,
        "payload_bytes": len(payload),
        "encode_s": enc_s,
        "decode_s": dec_s,
        "error": None,
    }


def fail_codec(exc):
    rec = empty_codec()
    rec["error"] = "%s: %s" % (type(exc).__name__, exc)
    return rec


def aggregate(rows, split, key):
    subset = [r for r in rows if r["split"] == split]
    out = {
        "n": len(subset),
        "payload_bytes": 0,
        "exact_count": 0,
        "mismatch_pixels": 0,
        "encode_s": 0.0,
        "decode_s": 0.0,
        "missing": 0,
    }
    for r in subset:
        block = r[key]
        if block.get("payload_bytes") is None:
            out["missing"] += 1
            continue
        out["payload_bytes"] += block["payload_bytes"]
        out["exact_count"] += 1 if block.get("exact") else 0
        out["mismatch_pixels"] += block.get("mismatch_count") or 0
        out["encode_s"] += block.get("encode_s") or 0.0
        out["decode_s"] += block.get("decode_s") or 0.0
    out["all_exact"] = out["exact_count"] == out["n"] and out["missing"] == 0
    return out


def write_summary(results):
    agg = results["aggregates"]
    bar = results["advance_bar"]
    h = agg["heldout"]
    lines = []
    lines.append("# Frozen shape-family A/B/C")
    lines.append("")
    lines.append("Seed %d. Images 0-15 fit, 16-23 held-out. 64x64 RGB, filled shapes, at most 6 shapes, palette at most 8, hard edges, integer coordinates. No antialiasing. Not tuned on held-out. No images dropped." % results["seed"])
    lines.append("")
    lines.append("Regenerate: `%s`" % results["command"])
    lines.append("")
    lines.append("## Held-out totals (8 images)")
    lines.append("")
    lines.append("| codec | payload bytes | exact | mismatch pixels | encode s | decode s |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name in ("A", "B", "C", "PNG", "JPEG_XL"):
        block = h[name]
        lines.append(
            "| %s | %s | %d/%d | %s | %.6f | %.6f |"
            % (
                name,
                block["payload_bytes"] if block["missing"] == 0 else "n/a",
                block["exact_count"],
                block["n"],
                block["mismatch_pixels"] if block["missing"] == 0 else "n/a",
                block["encode_s"],
                block["decode_s"],
            )
        )
    lines.append("")
    lines.append("## Fit totals (16 images)")
    lines.append("")
    lines.append("| codec | payload bytes | exact | mismatch pixels | encode s | decode s |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name in ("A", "B", "C", "PNG", "JPEG_XL"):
        block = agg["fit"][name]
        lines.append(
            "| %s | %s | %d/%d | %s | %.6f | %.6f |"
            % (
                name,
                block["payload_bytes"] if block["missing"] == 0 else "n/a",
                block["exact_count"],
                block["n"],
                block["mismatch_pixels"] if block["missing"] == 0 else "n/a",
                block["encode_s"],
                block["decode_s"],
            )
        )
    lines.append("")
    lines.append("## Advance bar")
    lines.append("")
    lines.append("- Bar 1 (A exact on every held-out image AND A held-out payload smaller than JPEG XL lossless): %s" % ("pass" if bar["bar1_pass"] else "fail"))
    lines.append("- Bar 2 (B plus correction beats JPEG XL lossless on the held-out 8): %s" % ("pass" if bar["bar2_pass"] else "fail"))
    lines.append("- A held-out exact: %s" % bar["a_heldout_exact"])
    lines.append("- A held-out bytes %s; JPEG XL held-out bytes %s" % (bar["a_heldout_bytes"], bar["jxl_heldout_bytes"]))
    lines.append("- B held-out bytes %s (description %s, correction %s); JPEG XL held-out bytes %s" % (
        bar["b_heldout_bytes"], bar["b_heldout_description_bytes"], bar["b_heldout_correction_bytes"], bar["jxl_heldout_bytes"]
    ))
    if bar["xor_correction_as_costly_as_jxl"] is True:
        lines.append("- Held-out XOR correction bytes are greater than or equal to JPEG XL lossless. Outline-then-color failed.")
    elif bar["xor_correction_as_costly_as_jxl"] is False:
        lines.append("- Held-out XOR correction bytes are smaller than JPEG XL lossless.")
    lines.append("- C held-out description byte-identical: %d/%d" % (bar["c_heldout_identical"], 8))
    lines.append("")
    lines.append("## Payload contents")
    lines.append("")
    lines.append("- A: seed, image index, settings id, max shapes, max palette, width, height, palette RGB, shape type, geometry, palette index. No XOR correction. No LLM prompt.")
    lines.append("- B: width, height, palette RGB, background index, region type, geometry bounds, palette index, XOR correction (method byte, length, zlib blob). No generator seed. Encoder sees pixels only.")
    lines.append("- C: same stored description as a second B encode after one decode. Byte-identical means the second payload equals the first. One cycle only.")
    lines.append("- PNG: full lossless PNG file. JPEG XL: full lossless `.jxl` file from `cjxl -d 0 -e 7`.")
    lines.append("- Shared decoder size: %d bytes from %s. Not added to per-image payloads. generator.py is %d bytes and is not required to decode stored payloads." % (
        results["shared_decoder"]["bytes"],
        ", ".join(results["shared_decoder"]["included"][i]["path"] for i in range(len(results["shared_decoder"]["included"]))),
        results["shared_decoder"]["generator_source_bytes_not_in_decoder"],
    ))
    lines.append("")
    lines.append("## Versions and commands")
    lines.append("")
    lines.append("- Python: %s" % results["versions"]["python"])
    lines.append("- zlib: %s" % results["versions"]["zlib"])
    lines.append("- cjxl: %s" % results["versions"]["cjxl"])
    lines.append("- djxl: %s" % results["versions"]["djxl"])
    lines.append("- JPEG XL install failure class: %s" % results["versions"]["jpegxl_install_failure_class"])
    lines.append("- PNG command: stdlib PNG writer in baselines.py (8-bit RGB, filter 0-4, zlib level 9)")
    lines.append("- JPEG XL encode: `cjxl INPUT.ppm OUTPUT.jxl -d 0 -e 7 --quiet`")
    lines.append("- JPEG XL decode: `djxl INPUT.jxl OUTPUT.ppm --quiet`")
    lines.append("- Wall clock: %.6f s" % results["wall_clock_s"])
    lines.append("- Started: %s" % results["started_et"])
    lines.append("- Finished: %s" % results["finished_et"])
    lines.append("- Output: experiments/shape_family/results.json and experiments/shape_family/SUMMARY.md")
    lines.append("- Machine-readable results: experiments/shape_family/results.json")
    if results["failures"]:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for item in results["failures"]:
            lines.append("- %s" % item)
    else:
        lines.append("")
        lines.append("No runner failures.")
    lines.append("")
    path = os.path.join(HERE, "SUMMARY.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    started = now_utc()
    t_wall = time.perf_counter()
    failures = []
    rows = []

    cjxl_text, cjxl_err = baselines.cjxl_version()
    djxl_text, djxl_err = baselines.djxl_version()
    jxl_failure = None
    if cjxl_err or djxl_err:
        jxl_failure = cjxl_err or djxl_err

    for image_index in range(N_IMAGES):
        scene = generate_scene(image_index, MASTER_SEED)
        px = render(scene)
        row = {
            "image_index": image_index,
            "split": split_name(image_index),
            "width": WIDTH,
            "height": HEIGHT,
            "n_shapes": len(scene["shapes"]),
            "n_palette_generator": len(scene["palette"]),
            "pixel_sha256": sha256_bytes(px),
            "raw_rgb_bytes": len(px),
        }
        try:
            row["A"] = run_a(scene, px)
            if not row["A"]["exact"]:
                failures.append("A mismatch image %d count %s" % (image_index, row["A"]["mismatch_count"]))
        except Exception as exc:
            row["A"] = fail_codec(exc)
            failures.append("A image %d %s" % (image_index, type(exc).__name__))
        try:
            row["B"] = run_b(px)
            if not row["B"]["exact"]:
                failures.append("B mismatch image %d count %s" % (image_index, row["B"]["mismatch_count"]))
        except Exception as exc:
            row["B"] = fail_codec(exc)
            failures.append("B image %d %s" % (image_index, type(exc).__name__))
        try:
            row["C"] = run_c(px)
            if not row["C"]["exact"]:
                failures.append("C mismatch image %d count %s" % (image_index, row["C"]["mismatch_count"]))
        except Exception as exc:
            row["C"] = fail_codec(exc)
            failures.append("C image %d %s" % (image_index, type(exc).__name__))
        try:
            row["PNG"] = run_png(px)
            if not row["PNG"]["exact"]:
                failures.append("PNG mismatch image %d count %s" % (image_index, row["PNG"]["mismatch_count"]))
        except Exception as exc:
            row["PNG"] = fail_codec(exc)
            failures.append("PNG image %d %s" % (image_index, type(exc).__name__))
        if jxl_failure:
            row["JPEG_XL"] = empty_codec()
            row["JPEG_XL"]["error"] = jxl_failure
        else:
            try:
                row["JPEG_XL"] = run_jxl(px)
                if not row["JPEG_XL"]["exact"]:
                    failures.append(
                        "JPEG XL mismatch image %d count %s" % (image_index, row["JPEG_XL"]["mismatch_count"])
                    )
            except Exception as exc:
                row["JPEG_XL"] = fail_codec(exc)
                failures.append("JPEG XL image %d %s" % (image_index, type(exc).__name__))
        rows.append(row)
        print(
            "image %02d %s A=%s B=%s C=%s PNG=%s JXL=%s C_ident=%s"
            % (
                image_index,
                row["split"],
                row["A"].get("payload_bytes"),
                row["B"].get("payload_bytes"),
                row["C"].get("payload_bytes"),
                row["PNG"].get("payload_bytes"),
                row["JPEG_XL"].get("payload_bytes"),
                row["C"].get("description_byte_identical"),
            ),
            flush=True,
        )

    codecs = ("A", "B", "C", "PNG", "JPEG_XL")
    aggregates = {
        "fit": {name: aggregate(rows, "fit", name) for name in codecs},
        "heldout": {name: aggregate(rows, "heldout", name) for name in codecs},
    }

    def sum_field(split, name, field):
        total = 0
        for r in rows:
            if r["split"] != split:
                continue
            block = r[name]
            val = block.get(field)
            if val is None:
                return None
            total += val
        return total

    a_exact = aggregates["heldout"]["A"]["all_exact"]
    a_bytes = aggregates["heldout"]["A"]["payload_bytes"] if aggregates["heldout"]["A"]["missing"] == 0 else None
    b_bytes = aggregates["heldout"]["B"]["payload_bytes"] if aggregates["heldout"]["B"]["missing"] == 0 else None
    jxl_bytes = aggregates["heldout"]["JPEG_XL"]["payload_bytes"] if aggregates["heldout"]["JPEG_XL"]["missing"] == 0 else None
    b_corr = sum_field("heldout", "B", "correction_bytes")
    b_desc = sum_field("heldout", "B", "description_bytes")
    c_ident = 0
    for r in rows:
        if r["split"] == "heldout" and r["C"].get("description_byte_identical") is True:
            c_ident += 1

    if jxl_bytes is None:
        bar1 = False
        bar2 = False
        xor_costly = None
    else:
        bar1 = bool(a_exact and a_bytes is not None and a_bytes < jxl_bytes)
        bar2 = bool(b_bytes is not None and b_bytes < jxl_bytes)
        xor_costly = None if b_corr is None else bool(b_corr >= jxl_bytes)

    finished = now_utc()
    wall = time.perf_counter() - t_wall
    results = {
        "experiment": SETTINGS_ID,
        "seed": MASTER_SEED,
        "n_images": N_IMAGES,
        "fit_indices": "0-15",
        "heldout_indices": "16-23",
        "width": WIDTH,
        "height": HEIGHT,
        "command": REGEN_COMMAND,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "started_et": fmt_et(started),
        "finished_et": fmt_et(finished),
        "wall_clock_s": wall,
        "versions": {
            "python": sys.version.split()[0],
            "python_full": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "zlib": baselines.zlib_version(),
            "cjxl": cjxl_text,
            "djxl": djxl_text,
            "jpegxl_install_failure_class": jxl_failure,
        },
        "commands": {
            "regenerate": REGEN_COMMAND,
            "png": "baselines.encode_png / decode_png (stdlib, zlib level 9, PNG filters 0-4)",
            "jpegxl_encode": "cjxl INPUT.ppm OUTPUT.jxl -d 0 -e 7 --quiet",
            "jpegxl_decode": "djxl INPUT.jxl OUTPUT.ppm --quiet",
        },
        "shared_decoder": shared_decoder_size(),
        "payload_included": {
            "A": [
                "magic/version",
                "seed",
                "image_index",
                "width",
                "height",
                "settings_id",
                "max_shapes",
                "max_palette",
                "palette RGB",
                "shape type",
                "geometry",
                "palette index",
            ],
            "B": [
                "width",
                "height",
                "palette RGB",
                "background index",
                "region type",
                "region bounds/geometry",
                "palette index",
                "XOR correction method, length, and zlib bytes",
            ],
            "C": [
                "second B payload after one encode/decode/encode cycle",
                "byte-identical flag compares first and second stored descriptions",
            ],
        },
        "notes": [
            "A does not XOR-correct. Exact means redrawn pixels match.",
            "B encoder sees pixels only, not generator args or seed.",
            "C is one cycle only, not indefinite reversibility.",
            "Binary PNGs are not committed. Images regenerate from the seed.",
            "Fit split was not used to change the codec. Held-out was not used to tune.",
        ],
        "images": rows,
        "aggregates": aggregates,
        "advance_bar": {
            "bar1_pass": bar1,
            "bar2_pass": bar2,
            "a_heldout_exact": a_exact,
            "a_heldout_bytes": a_bytes,
            "b_heldout_bytes": b_bytes,
            "b_heldout_description_bytes": b_desc,
            "b_heldout_correction_bytes": b_corr,
            "c_heldout_bytes": aggregates["heldout"]["C"]["payload_bytes"] if aggregates["heldout"]["C"]["missing"] == 0 else None,
            "png_heldout_bytes": aggregates["heldout"]["PNG"]["payload_bytes"] if aggregates["heldout"]["PNG"]["missing"] == 0 else None,
            "jxl_heldout_bytes": jxl_bytes,
            "c_heldout_identical": c_ident,
            "xor_correction_as_costly_as_jxl": xor_costly,
            "rule": "strictly smaller payload wins. Equality is fail. B includes correction bytes.",
        },
        "failures": failures,
        "output_paths": {
            "results_json": "experiments/shape_family/results.json",
            "summary_md": "experiments/shape_family/SUMMARY.md",
            "local_results_json": os.path.join(HERE, "results.json"),
            "local_summary_md": os.path.join(HERE, "SUMMARY.md"),
        },
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    write_summary(results)
    print("wall_clock_s", round(wall, 6))
    print("heldout A", a_bytes, "B", b_bytes, "JXL", jxl_bytes)
    print("bar1", bar1, "bar2", bar2, "xor_costly", xor_costly)
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
