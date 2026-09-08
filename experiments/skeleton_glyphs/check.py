"""Recompute packed-bits, payload, exactness, and JPEG XL sizes from fixtures.

Does not read a results JSON as authority. Exits non-zero if the printed
table disagrees with the tables in SUMMARY.md.
"""

from __future__ import annotations

import ast
import hashlib
import os
import pathlib
import re
import subprocess
import sys
import time
import zlib

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import encode  # noqa: E402
import decode  # noqa: E402
import raster  # noqa: E402

N = 24
FIT_N = 16
HELD_N = 8
PACKED = 128
SEED_TOKEN = "20260908"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_name(i: int) -> str:
    return "fit" if i < FIT_N else "held-out"


def version_string(cmd: str) -> str:
    proc = subprocess.run([cmd, "--version"], capture_output=True, text=True, check=False)
    text = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return text.strip("\n")


def parse_pnm(data: bytes) -> list[list[int]]:
    if not (data.startswith(b"P5") or data.startswith(b"P6")):
        raise ValueError("decoded image is not P5/P6")
    magic = data[:2]
    i = 2
    tokens: list[bytes] = []

    def skip_ws_and_comments() -> None:
        nonlocal i
        while i < len(data):
            c = data[i]
            if c in b" \t\r\n":
                i += 1
                continue
            if c == ord("#"):
                while i < len(data) and data[i] not in b"\r\n":
                    i += 1
                continue
            break

    while len(tokens) < 3:
        skip_ws_and_comments()
        start = i
        while i < len(data) and data[i] not in b" \t\r\n#":
            i += 1
        if start == i:
            raise ValueError("truncated PNM header")
        tokens.append(data[start:i])
    width = int(tokens[0])
    height = int(tokens[1])
    maxval = int(tokens[2])
    if i < len(data) and data[i] in b" \t\r\n":
        i += 1
    sample_bytes = 2 if maxval > 255 else 1
    channels = 3 if magic == b"P6" else 1
    need = width * height * channels * sample_bytes
    raw = data[i : i + need]
    if len(raw) != need:
        raise ValueError("PNM payload truncated")
    img = []
    pos = 0
    for _y in range(height):
        row = []
        for _x in range(width):
            vals = []
            for _c in range(channels):
                if sample_bytes == 1:
                    vals.append(raw[pos])
                    pos += 1
                else:
                    vals.append((raw[pos] << 8) | raw[pos + 1])
                    pos += 2
            # Binary glyph: any sample above 0 is ink. Equality is on bits.
            row.append(1 if any(v != 0 for v in vals) else 0)
        img.append(row)
    return img


def write_pgm(path: pathlib.Path, packed: bytes) -> None:
    img = raster.unpack_bits(packed, 32, 32)
    body = bytearray()
    for row in img:
        for v in row:
            body.append(255 if v else 0)
    path.write_bytes(b"P5\n32 32\n255\n" + bytes(body))


def jpegxl_size(packed: bytes, work: pathlib.Path, index: int) -> tuple[int, bool, str]:
    pgm = work / f"glyph_{index:02d}.pgm"
    jxl = work / f"glyph_{index:02d}.jxl"
    dec = work / f"glyph_{index:02d}_djxl.pgm"
    write_pgm(pgm, packed)
    enc = subprocess.run(
        ["cjxl", str(pgm), "-d", "0", "-e", "9", "--quiet", str(jxl)],
        capture_output=True,
        text=True,
        check=False,
    )
    if enc.returncode != 0 or not jxl.is_file():
        err = (enc.stderr or enc.stdout or "cjxl failed").strip()
        return -1, False, err
    dec_run = subprocess.run(
        ["djxl", str(jxl), str(dec), "--quiet"],
        capture_output=True,
        text=True,
        check=False,
    )
    if dec_run.returncode != 0 or not dec.is_file():
        err = (dec_run.stderr or dec_run.stdout or "djxl failed").strip()
        return jxl.stat().st_size, False, err
    try:
        got = parse_pnm(dec.read_bytes())
    except ValueError as exc:
        return jxl.stat().st_size, False, str(exc)
    expect = raster.unpack_bits(packed, 32, 32)
    return jxl.stat().st_size, raster.images_equal(got, expect), ""


def local_modules(path: pathlib.Path) -> set[str]:
    names = set()
    for p in path.parent.glob("*.py"):
        names.add(p.stem)
    return names


def scan_imports(start: pathlib.Path) -> tuple[str, list[str], list[str]]:
    """Source-text scan of encode.py and local imports. Returns status, modules, notes."""
    notes: list[str] = []
    modules: list[str] = []
    seen: set[pathlib.Path] = set()
    bad: list[str] = []
    local_names = local_modules(start)

    def walk(path: pathlib.Path, reason: str) -> None:
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        text = path.read_text(encoding="utf-8")
        modules.append(f"{path.name} ({reason})")
        if SEED_TOKEN in text:
            bad.append(f"{path.name} contains seed token {SEED_TOKEN}")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            bad.append(f"{path.name} does not parse: {exc}")
            return
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names.append(node.module.split(".")[0])
                elif node.level:
                    # relative import: look at imported names
                    for alias in node.names:
                        names.append(alias.name.split(".")[0])
            else:
                continue
            for name in names:
                if name == "generator":
                    bad.append(f"{path.name} imports generator")
                if name in local_names and name != "generator":
                    child = path.parent / f"{name}.py"
                    if child.is_file():
                        walk(child, f"local import from {path.name}")
                else:
                    modules.append(f"{name} (imported by {path.name})")

        if path.name == "encode.py":
            fn = None
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "encode":
                    fn = node
                    break
            if fn is None:
                bad.append("encode() is missing")
            else:
                args = [a.arg for a in fn.args.args]
                if args[:1] != ["raster_bytes_128"] or "seed" in args:
                    bad.append(f"encode() signature is {args}, expected raster_bytes_128 first and no seed")

    walk(start, "encoder entry")
    # de-duplicate module lines preserving order
    uniq: list[str] = []
    seen_line = set()
    for line in modules:
        if line not in seen_line:
            seen_line.add(line)
            uniq.append(line)
    status = "clean" if not bad else "not clean"
    notes.extend(bad)
    return status, uniq, notes


def corr_pays(payload_bytes: int, corr_bytes: int, packed: int = PACKED) -> int:
    saved = packed - (payload_bytes - corr_bytes)
    return 1 if corr_bytes >= saved else 0


def evaluate() -> dict:
    t0 = time.perf_counter()
    failures: list[str] = []
    rows = []
    work = ROOT / "jxl_work"
    work.mkdir(parents=True, exist_ok=True)
    cjxl_ver = version_string("cjxl")
    djxl_ver = version_string("djxl")
    jxl_ok = bool(cjxl_ver) and shutil_which("cjxl") and shutil_which("djxl")
    if not jxl_ok:
        failures.append("install_failure: cjxl or djxl not available")

    for i in range(N):
        fix_path = ROOT / "fixtures" / f"glyph_{i:02d}.bin"
        pay_path = ROOT / "payloads" / f"glyph_{i:02d}.sg"
        fixture = fix_path.read_bytes()
        if len(fixture) != PACKED:
            failures.append(f"glyph_{i:02d} fixture is {len(fixture)} bytes, expected 128")
        payload = pay_path.read_bytes()
        reencoded = encode.encode(fixture, 32, 32)
        if reencoded != payload:
            failures.append(f"glyph_{i:02d} stored payload is not the encoder output")
        redraw_a = decode.decode(payload, 32, 32)
        redraw_b = decode.decode(payload, 32, 32)
        info = decode.inspect_payload(payload)
        exact = redraw_a == fixture and redraw_b == redraw_a and len(redraw_a) == PACKED
        if not exact:
            failures.append(f"glyph_{i:02d} redraw does not match fixture")
        jxl_bytes = -1
        jxl_exact = False
        jxl_err = ""
        if jxl_ok:
            jxl_bytes, jxl_exact, jxl_err = jpegxl_size(fixture, work, i)
            if jxl_bytes < 0 or not jxl_exact:
                failures.append(f"glyph_{i:02d} JPEG XL failure: {jxl_err or 'pixel mismatch'}")
        rows.append(
            {
                "glyph": f"{i:02d}",
                "split": split_name(i),
                "payload_bytes": len(payload),
                "packed_bits": PACKED if len(fixture) == PACKED else len(fixture),
                "jxl_bytes": jxl_bytes,
                "exact": 1 if exact else 0,
                "desc_bytes": info["desc_bytes"],
                "corr_bytes": info["corr_bytes"],
                "corr_pays": corr_pays(len(payload), info["corr_bytes"], PACKED),
                "repr": "stroke" if info["repr"] == decode.REPR_STROKE else "skel",
                "radius": info["radius"],
                "fixture_sha256": sha256(fixture),
                "redraw_sha256": sha256(redraw_a),
                "payload_sha256": sha256(payload),
                "jxl_exact": 1 if jxl_exact else 0,
                "two_runs": 1 if redraw_a == redraw_b else 0,
            }
        )

    def subset(name: str) -> list[dict]:
        return [r for r in rows if r["split"] == name]

    def totals(name: str, items: list[dict]) -> dict:
        jxl_sum = sum(r["jxl_bytes"] for r in items) if jxl_ok and all(r["jxl_bytes"] >= 0 for r in items) else -1
        exact_n = sum(r["exact"] for r in items)
        return {
            "split": name,
            "n": len(items),
            "payload_bytes": sum(r["payload_bytes"] for r in items),
            "packed_bits": sum(r["packed_bits"] for r in items),
            "jxl_bytes": jxl_sum,
            "exact": f"{exact_n}/{len(items)}",
            "exact_n": exact_n,
            "desc_bytes": sum(r["desc_bytes"] for r in items),
            "corr_bytes": sum(r["corr_bytes"] for r in items),
        }

    fit = totals("fit", subset("fit"))
    held = totals("held-out", subset("held-out"))
    all_rows = totals("all", rows)
    header_desc = held["payload_bytes"] - held["corr_bytes"]
    saved = held["packed_bits"] - header_desc
    paid = held["corr_bytes"] >= saved
    status, modules, notes = scan_imports(ROOT / "encode.py")
    import_clean = status == "clean"
    if not import_clean:
        failures.extend(notes)

    held_exact_ok = held["exact_n"] == HELD_N
    jxl_compared = held["jxl_bytes"] >= 0
    beats_packed = held["payload_bytes"] < held["packed_bits"]
    beats_jxl = jxl_compared and held["payload_bytes"] < held["jxl_bytes"]
    advance = import_clean and held_exact_ok and beats_packed and beats_jxl
    elapsed = time.perf_counter() - t0
    shared = (ROOT / "decode.py").stat().st_size + (ROOT / "raster.py").stat().st_size
    return {
        "rows": rows,
        "fit": fit,
        "held": held,
        "all": all_rows,
        "bytes_saved_vs_packed": saved,
        "correction_paid_for_skeleton": "yes" if paid else "no",
        "representation_stops": "yes" if paid else "no",
        "advance": "PASS" if advance else "FAIL",
        "import_graph": status,
        "import_modules": modules,
        "import_notes": notes,
        "shared_decoder_bytes": shared,
        "python_version": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "zlib_version": zlib.ZLIB_VERSION,
        "cjxl_version": cjxl_ver,
        "djxl_version": djxl_ver,
        "runtime_seconds": elapsed,
        "failures": failures,
        "jxl_ok": jxl_ok,
    }


def shutil_which(cmd: str) -> str | None:
    from shutil import which

    return which(cmd)


def render_tables(result: dict) -> str:
    lines = []
    lines.append("| glyph | split | payload_bytes | packed_bits | jxl_bytes | exact | desc_bytes | corr_bytes | corr_pays | repr | fixture_sha256 | redraw_sha256 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in result["rows"]:
        lines.append(
            "| {glyph} | {split} | {payload_bytes} | {packed_bits} | {jxl_bytes} | {exact} | {desc_bytes} | {corr_bytes} | {corr_pays} | {repr} | {fixture_sha256} | {redraw_sha256} |".format(
                **r
            )
        )
    lines.append("")
    lines.append("| split | n | payload_bytes | packed_bits | jxl_bytes | exact |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for block in (result["fit"], result["held"], result["all"]):
        lines.append(
            "| {split} | {n} | {payload_bytes} | {packed_bits} | {jxl_bytes} | {exact} |".format(**block)
        )
    lines.append("")
    keys = [
        ("advance", result["advance"]),
        ("representation_stops", result["representation_stops"]),
        ("correction_paid_for_skeleton", result["correction_paid_for_skeleton"]),
        ("import_graph", result["import_graph"]),
        ("held_out_payload", result["held"]["payload_bytes"]),
        ("held_out_packed", result["held"]["packed_bits"]),
        ("held_out_jxl", result["held"]["jxl_bytes"]),
        ("held_out_exact", result["held"]["exact"]),
        ("held_out_desc_bytes", result["held"]["desc_bytes"]),
        ("held_out_corr_bytes", result["held"]["corr_bytes"]),
        ("held_out_bytes_saved_vs_packed", result["bytes_saved_vs_packed"]),
        ("fit_payload", result["fit"]["payload_bytes"]),
        ("fit_packed", result["fit"]["packed_bits"]),
        ("fit_jxl", result["fit"]["jxl_bytes"]),
        ("fit_exact", result["fit"]["exact"]),
        ("all_payload", result["all"]["payload_bytes"]),
        ("all_packed", result["all"]["packed_bits"]),
        ("all_jxl", result["all"]["jxl_bytes"]),
        ("all_exact", result["all"]["exact"]),
        ("shared_decoder_bytes", result["shared_decoder_bytes"]),
    ]
    lines.append("| key | value |")
    lines.append("| --- | --- |")
    for k, v in keys:
        lines.append(f"| {k} | {v} |")
    lines.append("")
    return "\n".join(lines)


def parse_markdown_tables(text: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if cells and all(set(c) <= set("-: ") and c for c in cells):
                continue
            current.append(cells)
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)
    return tables


def table_map(table: list[list[str]]) -> dict[str, list[str]]:
    header = table[0]
    out = {}
    for row in table[1:]:
        if not row:
            continue
        key = row[0]
        out[key] = row
        # header kept for comparison by caller
    out["__header__"] = header
    return out


def find_table(tables: list[list[list[str]]], first_header: str, second: str | None = None) -> list[list[str]] | None:
    for table in tables:
        if not table:
            continue
        header = table[0]
        if header and header[0] == first_header and (second is None or (len(header) > 1 and header[1] == second)):
            return table
    return None


def compare_to_summary(result: dict, summary_text: str) -> list[str]:
    printed = render_tables(result)
    parsed_print = parse_markdown_tables(printed)
    parsed_sum = parse_markdown_tables(summary_text)
    errors: list[str] = []
    pairs = [
        ("glyph", "split"),
        ("split", "n"),
        ("key", "value"),
    ]
    for a, b in pairs:
        got = find_table(parsed_print, a, b)
        exp = find_table(parsed_sum, a, b)
        if got is None or exp is None:
            errors.append(f"missing table starting with {a}|{b}")
            continue
        if got[0] != exp[0]:
            errors.append(f"header mismatch for {a}: printed {got[0]} summary {exp[0]}")
        got_map = {row[0]: row for row in got[1:]}
        exp_map = {row[0]: row for row in exp[1:]}
        if set(got_map) != set(exp_map):
            errors.append(f"row keys differ for {a}: printed {sorted(got_map)} summary {sorted(exp_map)}")
        for key in sorted(set(got_map) & set(exp_map)):
            if got_map[key] != exp_map[key]:
                errors.append(f"row {a}={key} printed {got_map[key]} summary {exp_map[key]}")
    return errors


def write_import_graph(result: dict) -> None:
    lines = [f"status: {result['import_graph']}"]
    lines.append("modules:")
    for m in result["import_modules"]:
        lines.append(f"- {m}")
    if result["import_notes"]:
        lines.append("notes:")
        for n in result["import_notes"]:
            lines.append(f"- {n}")
    else:
        lines.append("notes:")
        lines.append("- encode.py does not import generator and does not contain seed 20260908")
        lines.append("- encode(raster_bytes_128, width=32, height=32) reads only the committed raster")
        lines.append("- local imports followed from encode.py: decode.py, raster.py")
    (ROOT / "import_graph.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    result = evaluate()
    printed = render_tables(result)
    print(printed)
    print(f"python: {result['python_full']}")
    print(f"zlib: {result['zlib_version']}")
    print("cjxl_version:")
    print(result["cjxl_version"])
    print("djxl_version:")
    print(result["djxl_version"])
    print(f"runtime_seconds: {result['runtime_seconds']:.6f}")
    print(f"advance: {result['advance']}")
    print(f"representation_stops: {result['representation_stops']}")
    print(f"correction_paid_for_skeleton: {result['correction_paid_for_skeleton']}")
    if result["failures"]:
        print("failures:")
        for f in result["failures"]:
            print(f"- {f}")
    else:
        print("failures: none")

    summary_path = ROOT / "SUMMARY.md"
    if not summary_path.is_file():
        print("ERROR: SUMMARY.md missing")
        return 1
    summary_text = summary_path.read_text(encoding="utf-8")
    errors = compare_to_summary(result, summary_text)
    graph_path = ROOT / "import_graph.txt"
    if not graph_path.is_file():
        errors.append("import_graph.txt missing")
    else:
        graph_text = graph_path.read_text(encoding="utf-8")
        m = re.search(r"^status:\s*(\S+)", graph_text, re.M)
        if not m or m.group(1) != result["import_graph"]:
            errors.append(
                f"import_graph.txt status {m.group(1) if m else 'missing'} != scanned {result['import_graph']}"
            )
    if errors:
        print("TABLE_MISMATCH")
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    print("TABLE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
