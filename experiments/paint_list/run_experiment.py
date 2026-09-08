#!/usr/bin/env python3
"""Paint-list recovery experiment. No correction bytes.

From this directory, or the repository root after publish:
    python3 experiments/paint_list/run_experiment.py

Family N seed 20260908, non-overlapping. Family O is the frozen overlapping
generator from experiments/shape_family/generator.py at d287e17 (MASTER_SEED
20260908, 24 images). Images 0-15 fit, 16-23 held-out. Not tuned on held-out.
No images dropped. No residual. No network at encode or decode.
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
import codec
import generator_n
import generator_o
from raster import HEIGHT, WIDTH, mismatch_count, pixels_equal

try:
    from zoneinfo import ZoneInfo

    ET = ZoneInfo("America/New_York")
except Exception:
    ET = timezone(timedelta(hours=-4))

REGEN_COMMAND = "python3 experiments/paint_list/run_experiment.py"
DECODER_FILES = ("raster.py", "codec.py")
FAMILIES = (
    ("N", generator_n),
    ("O", generator_o),
)


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
        files.append({"path": "experiments/paint_list/" + name, "bytes": n})
        total += n
    return {
        "bytes": total,
        "included": files,
        "excluded": [
            "experiments/paint_list/generator_n.py",
            "experiments/paint_list/generator_o.py",
            "experiments/paint_list/baselines.py",
            "experiments/paint_list/run_experiment.py",
            "experiments/paint_list/check_published.py",
            "experiments/paint_list/results.json",
            "experiments/paint_list/SUMMARY.md",
        ],
        "note": (
            "Shared decoder size is the byte size of raster.py and codec.py. "
            "It is not added to per-image payloads. Decode redraws the stored "
            "program and does not need the generators."
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
        "stored": False,
    }


def run_program(px):
    payload, enc_s = timed(lambda: codec.encode(px, WIDTH, HEIGHT))
    if payload is None:
        rec = empty_codec()
        rec["encode_s"] = enc_s
        rec["exact"] = False
        rec["mismatch_count"] = None
        rec["stored"] = False
        rec["error"] = "no exact paint list"
        return rec
    decoded, dec_s = timed(lambda: codec.decode(payload))
    mis = mismatch_count(px, decoded)
    return {
        "exact": pixels_equal(px, decoded),
        "mismatch_count": mis,
        "payload_bytes": len(payload),
        "encode_s": enc_s,
        "decode_s": dec_s,
        "error": None,
        "stored": True,
        "correction_bytes": 0,
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
    lines = []
    lines.append("# Paint-list recovery, no correction")
    lines.append("")
    lines.append(
        "Family N seed %s. Family O seed %s, same generator behavior as "
        "experiments/shape_family/generator.py at d287e17. Images 0-15 fit, 16-23 held-out. "
        "64x64 RGB, at most 6 filled shapes, palette at most 8, hard edges, integer coordinates. "
        "No antialiasing. Encoder sees pixels only. Stored program is palette plus ordered shapes. "
        "No XOR, residual, or correction bytes. Not tuned on held-out. No images dropped."
        % (results["family_n_seed"], results["family_o_seed"])
    )
    lines.append("")
    lines.append("Regenerate: `%s`" % results["command"])
    lines.append("")
    lines.append(results["family_n_seed_note"])
    lines.append("")
    for fam in ("N", "O"):
        lines.append("## Family %s held-out totals (8 images)" % fam)
        lines.append("")
        lines.append("| codec | payload bytes | exact | missing payloads |")
        lines.append("| --- | ---: | ---: | ---: |")
        h = results["aggregates"][fam]["heldout"]
        for name in ("program", "PNG", "JPEG_XL"):
            block = h[name]
            lines.append(
                "| %s | %s | %d/%d | %d |"
                % (
                    name,
                    block["payload_bytes"] if block["missing"] == 0 else block["payload_bytes"],
                    block["exact_count"],
                    block["n"],
                    block["missing"],
                )
            )
        lines.append("")
        lines.append("## Family %s fit totals (16 images)" % fam)
        lines.append("")
        lines.append("| codec | payload bytes | exact | missing payloads |")
        lines.append("| --- | ---: | ---: | ---: |")
        f = results["aggregates"][fam]["fit"]
        for name in ("program", "PNG", "JPEG_XL"):
            block = f[name]
            lines.append(
                "| %s | %s | %d/%d | %d |"
                % (
                    name,
                    block["payload_bytes"],
                    block["exact_count"],
                    block["n"],
                    block["missing"],
                )
            )
        lines.append("")
    bar = results["advance_bar"]
    lines.append("## Advance bar")
    lines.append("")
    lines.append(
        "- Advance only if Family N is exact on every held-out image AND Family N held-out program payload is smaller than JPEG XL effort 9 on those same 8: %s"
        % ("pass" if bar["pass"] else "fail")
    )
    lines.append("- Family N held-out exact: %s/%s" % (bar["n_heldout_exact"], 8))
    lines.append(
        "- Family N held-out program bytes %s; PNG %s; JPEG XL effort 9 %s"
        % (bar["n_heldout_program_bytes"], bar["n_heldout_png_bytes"], bar["n_heldout_jxl_bytes"])
    )
    lines.append("- Family O held-out exact: %s/%s" % (bar["o_heldout_exact"], 8))
    lines.append(
        "- Family O held-out program bytes %s; PNG %s; JPEG XL effort 9 %s"
        % (bar["o_heldout_program_bytes"], bar["o_heldout_png_bytes"], bar["o_heldout_jxl_bytes"])
    )
    if bar["family_o_finding"]:
        lines.append("- Finding: %s" % bar["family_o_finding"])
    lines.append("")
    lines.append("## Payload contents")
    lines.append("")
    lines.append(
        "- Program: magic/version, settings code %s (%s), width, height, palette RGB, shape type, geometry, color index, recovered order. No seed (encoder sees pixels only). No XOR. No residual. No correction bytes."
        % (codec.SETTINGS_CODE, codec.SETTINGS_ID)
    )
    lines.append("- Failed images store no program. Their payload is missing, not a residual.")
    lines.append("- PNG: full lossless PNG file. JPEG XL: full lossless `.jxl` file from `cjxl -d 0 -e 9`.")
    lines.append(
        "- Shared decoder size: %d bytes from %s. Not added to per-image payloads."
        % (
            results["shared_decoder"]["bytes"],
            ", ".join(item["path"] for item in results["shared_decoder"]["included"]),
        )
    )
    lines.append("")
    lines.append("## Versions and commands")
    lines.append("")
    lines.append("- Python: %s" % results["versions"]["python"])
    lines.append("- zlib: %s" % results["versions"]["zlib"])
    lines.append("- cjxl: %s" % results["versions"]["cjxl"])
    lines.append("- djxl: %s" % results["versions"]["djxl"])
    lines.append("- JPEG XL effort: %s" % results["versions"]["jpegxl_effort"])
    lines.append("- JPEG XL install failure class: %s" % results["versions"]["jpegxl_install_failure_class"])
    lines.append("- PNG command: stdlib PNG writer in baselines.py (8-bit RGB, filter 0-4, zlib level 9)")
    lines.append("- JPEG XL encode: `cjxl INPUT.ppm OUTPUT.jxl -d 0 -e 9 --quiet`")
    lines.append("- JPEG XL decode: `djxl INPUT.jxl OUTPUT.ppm --quiet`")
    lines.append("- Checker: `%s`" % results["checker_command"])
    lines.append("- Wall clock: %.6f s" % results["wall_clock_s"])
    lines.append("- Started: %s" % results["started_et"])
    lines.append("- Finished: %s" % results["finished_et"])
    lines.append("")
    if results["failures"]:
        lines.append("## Failures")
        lines.append("")
        for item in results["failures"]:
            lines.append("- %s" % item)
    else:
        lines.append("No runner exceptions. Exactness failures are recorded as not exact.")
    lines.append("")
    path = os.path.join(HERE, "SUMMARY.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    started = now_utc()
    t_wall = time.perf_counter()
    failures = []
    by_family = {}

    cjxl_text, cjxl_err = baselines.cjxl_version()
    djxl_text, djxl_err = baselines.djxl_version()
    jxl_failure = None
    if cjxl_err or djxl_err:
        jxl_failure = cjxl_err or djxl_err
    if baselines.JXL_EFFORT != 9:
        jxl_failure = "effort_not_9"

    for fam, gen in FAMILIES:
        rows = []
        for image_index in range(gen.N_IMAGES):
            scene = gen.generate_scene(image_index, gen.MASTER_SEED)
            px = gen.render(scene)
            if fam == "N" and not generator_n.masks_disjoint(scene):
                failures.append("Family N overlap image %d" % image_index)
            row = {
                "family": fam,
                "image_index": image_index,
                "split": gen.split_name(image_index),
                "width": WIDTH,
                "height": HEIGHT,
                "n_shapes_generator": len(scene["shapes"]),
                "n_palette_generator": len(scene["palette"]),
                "pixel_sha256": sha256_bytes(px),
                "raw_rgb_bytes": len(px),
            }
            try:
                row["program"] = run_program(px)
            except Exception as exc:
                row["program"] = fail_codec(exc)
                failures.append("program %s image %d %s" % (fam, image_index, type(exc).__name__))
            try:
                row["PNG"] = run_png(px)
            except Exception as exc:
                row["PNG"] = fail_codec(exc)
                failures.append("PNG %s image %d %s" % (fam, image_index, type(exc).__name__))
            if jxl_failure:
                row["JPEG_XL"] = empty_codec()
                row["JPEG_XL"]["error"] = jxl_failure
            else:
                try:
                    row["JPEG_XL"] = run_jxl(px)
                except Exception as exc:
                    row["JPEG_XL"] = fail_codec(exc)
                    failures.append("JPEG XL %s image %d %s" % (fam, image_index, type(exc).__name__))
            rows.append(row)
            print(
                "family %s image %02d %s program=%s exact=%s PNG=%s JXL=%s"
                % (
                    fam,
                    image_index,
                    row["split"],
                    row["program"].get("payload_bytes"),
                    row["program"].get("exact"),
                    row["PNG"].get("payload_bytes"),
                    row["JPEG_XL"].get("payload_bytes"),
                ),
                flush=True,
            )
        by_family[fam] = rows

    aggregates = {}
    for fam, _gen in FAMILIES:
        rows = by_family[fam]
        aggregates[fam] = {
            "fit": {name: aggregate(rows, "fit", name) for name in ("program", "PNG", "JPEG_XL")},
            "heldout": {name: aggregate(rows, "heldout", name) for name in ("program", "PNG", "JPEG_XL")},
        }

    n_h = aggregates["N"]["heldout"]
    o_h = aggregates["O"]["heldout"]
    n_exact = n_h["program"]["exact_count"]
    n_prog = n_h["program"]["payload_bytes"]
    n_png = n_h["PNG"]["payload_bytes"] if n_h["PNG"]["missing"] == 0 else None
    n_jxl = n_h["JPEG_XL"]["payload_bytes"] if n_h["JPEG_XL"]["missing"] == 0 else None
    o_exact = o_h["program"]["exact_count"]
    o_prog = o_h["program"]["payload_bytes"]
    o_png = o_h["PNG"]["payload_bytes"] if o_h["PNG"]["missing"] == 0 else None
    o_jxl = o_h["JPEG_XL"]["payload_bytes"] if o_h["JPEG_XL"]["missing"] == 0 else None

    advance = bool(
        n_h["program"]["all_exact"]
        and n_jxl is not None
        and n_prog < n_jxl
        and n_h["program"]["missing"] == 0
    )
    finding = None
    if advance and o_exact < 8:
        finding = "A raster does not determine paint order."
    elif (not advance) and n_h["program"]["all_exact"] and n_jxl is not None and n_prog >= n_jxl:
        finding = "Shape programs do not compress even the unambiguous case."
    elif not n_h["program"]["all_exact"]:
        finding = "Shape programs do not compress even the unambiguous case."

    finished = now_utc()
    wall = time.perf_counter() - t_wall
    results = {
        "experiment": "paint-list-no-correction",
        "family_n_seed": generator_n.MASTER_SEED,
        "family_o_seed": generator_o.MASTER_SEED,
        "family_n_seed_note": generator_n.SEED_NOTE,
        "family_o_source": "experiments/shape_family/generator.py at d287e17a10bdb5c2613fc72bc789a029d3f35a56",
        "n_images": 24,
        "fit_indices": "0-15",
        "heldout_indices": "16-23",
        "width": WIDTH,
        "height": HEIGHT,
        "command": REGEN_COMMAND,
        "checker_command": "python3 experiments/paint_list/check_published.py",
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
            "jpegxl_effort": baselines.JXL_EFFORT,
            "jpegxl_install_failure_class": jxl_failure,
        },
        "shared_decoder": shared_decoder_size(),
        "families": by_family,
        "aggregates": aggregates,
        "advance_bar": {
            "pass": advance,
            "n_heldout_exact": n_exact,
            "n_heldout_program_bytes": n_prog,
            "n_heldout_png_bytes": n_png,
            "n_heldout_jxl_bytes": n_jxl,
            "o_heldout_exact": o_exact,
            "o_heldout_program_bytes": o_prog,
            "o_heldout_png_bytes": o_png,
            "o_heldout_jxl_bytes": o_jxl,
            "family_o_finding": finding,
            "rule": "Family N exact on every held-out image AND Family N held-out program bytes strictly smaller than JPEG XL effort 9 on those same 8. Equality is fail. No residual.",
        },
        "failures": failures,
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    write_summary(results)
    print("wall_clock_s", round(wall, 6))
    print("advance", advance)
    print("N heldout exact", n_exact, "program", n_prog, "png", n_png, "jxl", n_jxl)
    print("O heldout exact", o_exact, "program", o_prog, "png", o_png, "jxl", o_jxl)
    print("finding", finding)
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
