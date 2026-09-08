"""PNG lossless and JPEG XL lossless baselines.

PNG is written and decoded with the standard library (8-bit RGB, non-interlaced,
adaptive per-row PNG filters, zlib level 9). JPEG XL is cjxl/djxl, distance 0,
effort 9. Effort 7 is not used. No other codec is substituted for JPEG XL.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
import zlib

PNG_SIG = b"\x89PNG\r\n\x1a\n"
JXL_EFFORT = 9
JXL_DISTANCE = "0"


def zlib_version():
    return zlib.ZLIB_VERSION


def _tool_version(name):
    exe = shutil.which(name)
    if not exe:
        return None, "FileNotFoundError"
    try:
        proc = subprocess.run([exe, "--version"], check=False, capture_output=True, text=True)
    except OSError as exc:
        return None, type(exc).__name__
    text = (proc.stdout or proc.stderr or "").strip()
    if proc.returncode != 0 and not text:
        return None, "CalledProcessError"
    return text, None


def cjxl_version():
    return _tool_version("cjxl")


def djxl_version():
    return _tool_version("djxl")


def _chunk(tag, data):
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _paeth(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _filter_row(raw, prev, width):
    best_row = None
    best_score = None
    for fno in range(5):
        row = bytearray(1 + width * 3)
        row[0] = fno
        for i in range(width * 3):
            x = raw[i]
            left = raw[i - 3] if i >= 3 else 0
            up = prev[i] if prev is not None else 0
            ul = prev[i - 3] if prev is not None and i >= 3 else 0
            if fno == 0:
                v = x
            elif fno == 1:
                v = (x - left) & 255
            elif fno == 2:
                v = (x - up) & 255
            elif fno == 3:
                v = (x - ((left + up) // 2)) & 255
            else:
                v = (x - _paeth(left, up, ul)) & 255
            row[1 + i] = v
        score = 0
        for b in row[1:]:
            score += b if b < 128 else 256 - b
        if best_score is None or score < best_score:
            best_score = score
            best_row = row
    return best_row


def encode_png(px, width, height):
    prev = None
    filtered = bytearray()
    stride = width * 3
    for y in range(height):
        raw = bytes(px[y * stride : (y + 1) * stride])
        row = _filter_row(raw, prev, width)
        filtered += row
        prev = raw
    compressed = zlib.compress(bytes(filtered), 9)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return PNG_SIG + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b"")


def _unfilter(filtered, width, height):
    stride = width * 3
    out = bytearray(width * height * 3)
    prev = bytearray(stride)
    off = 0
    for y in range(height):
        if off >= len(filtered):
            raise ValueError("truncated PNG scanlines")
        fno = filtered[off]
        off += 1
        row = filtered[off : off + stride]
        off += stride
        if len(row) != stride:
            raise ValueError("short PNG row")
        recon = bytearray(stride)
        for i in range(stride):
            left = recon[i - 3] if i >= 3 else 0
            up = prev[i]
            ul = prev[i - 3] if i >= 3 else 0
            v = row[i]
            if fno == 0:
                x = v
            elif fno == 1:
                x = (v + left) & 255
            elif fno == 2:
                x = (v + up) & 255
            elif fno == 3:
                x = (v + ((left + up) // 2)) & 255
            elif fno == 4:
                x = (v + _paeth(left, up, ul)) & 255
            else:
                raise ValueError("bad PNG filter")
            recon[i] = x
        out[y * stride : (y + 1) * stride] = recon
        prev = recon
    return bytes(out)


def decode_png(data):
    if data[:8] != PNG_SIG:
        raise ValueError("bad PNG signature")
    pos = 8
    width = height = None
    idat = bytearray()
    color_type = None
    bit_depth = None
    interlace = None
    while pos < len(data):
        if pos + 8 > len(data):
            raise ValueError("truncated PNG chunk")
        length = struct.unpack_from(">I", data, pos)[0]
        pos += 4
        tag = data[pos : pos + 4]
        pos += 4
        chunk = data[pos : pos + length]
        pos += length
        pos += 4
        if tag == b"IHDR":
            width, height, bit_depth, color_type, _c, _f, interlace = struct.unpack(">IIBBBBB", chunk)
        elif tag == b"IDAT":
            idat += chunk
        elif tag == b"IEND":
            break
    if width is None or color_type != 2 or bit_depth != 8 or interlace != 0:
        raise ValueError("unsupported PNG form")
    raw = zlib.decompress(bytes(idat))
    expected = height * (1 + width * 3)
    if len(raw) != expected:
        raise ValueError("PNG inflated size mismatch")
    return _unfilter(raw, width, height), width, height


def _write_ppm(path, px, width, height):
    header = ("P6\n%d %d\n255\n" % (width, height)).encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(px)


def _read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(b"P6"):
        raise ValueError("not P6")
    i = 2
    tokens = []
    n = len(data)
    while len(tokens) < 3:
        while i < n and data[i] in b" \t\r\n":
            i += 1
        if i < n and data[i] == ord("#"):
            while i < n and data[i] not in b"\n":
                i += 1
            continue
        start = i
        while i < n and data[i] not in b" \t\r\n":
            i += 1
        tokens.append(data[start:i])
    width = int(tokens[0])
    height = int(tokens[1])
    maxv = int(tokens[2])
    if maxv != 255:
        raise ValueError("unexpected PPM maxval")
    if i >= n or data[i] not in b" \t\r\n":
        raise ValueError("missing PPM raster separator")
    i += 1
    px = data[i:]
    if len(px) != width * height * 3:
        raise ValueError("PPM size mismatch")
    return px, width, height


def encode_jxl(px, width, height):
    exe = shutil.which("cjxl")
    if not exe:
        raise FileNotFoundError("cjxl")
    if JXL_EFFORT != 9:
        raise RuntimeError("JPEG XL effort must be 9")
    with tempfile.TemporaryDirectory(prefix="n2enc-jxl-") as tmp:
        src = os.path.join(tmp, "in.ppm")
        dst = os.path.join(tmp, "out.jxl")
        _write_ppm(src, px, width, height)
        cmd = [exe, src, dst, "-d", JXL_DISTANCE, "-e", str(JXL_EFFORT), "--quiet"]
        proc = subprocess.run(cmd, check=False, capture_output=True)
        if proc.returncode != 0 or not os.path.exists(dst):
            raise RuntimeError("cjxl failed rc=%s" % proc.returncode)
        with open(dst, "rb") as f:
            return f.read()


def decode_jxl(blob):
    exe = shutil.which("djxl")
    if not exe:
        raise FileNotFoundError("djxl")
    with tempfile.TemporaryDirectory(prefix="n2enc-jxl-") as tmp:
        src = os.path.join(tmp, "in.jxl")
        dst = os.path.join(tmp, "out.ppm")
        with open(src, "wb") as f:
            f.write(blob)
        cmd = [exe, src, dst, "--quiet"]
        proc = subprocess.run(cmd, check=False, capture_output=True)
        if proc.returncode != 0 or not os.path.exists(dst):
            raise RuntimeError("djxl failed rc=%s" % proc.returncode)
        px, width, height = _read_ppm(dst)
        return px, width, height
