"""Write the frozen 32x32 binary glyph fixtures.

Not imported by encode.py. The encoder must not import this module, read
the seed, or take generator arguments. After this script writes the
fixtures, those files are the only encoder input.

Seed: 20260908
"""

from __future__ import annotations

import math
import pathlib
import random

SEED = 20260908
WIDTH = 32
HEIGHT = 32
N_GLYPHS = 24


def blank() -> list[list[int]]:
    return [[0] * WIDTH for _ in range(HEIGHT)]


def stamp(img: list[list[int]], x: int, y: int, radius: int) -> None:
    for dy in range(-radius, radius + 1):
        yy = y + dy
        if yy < 0 or yy >= HEIGHT:
            continue
        row = img[yy]
        for dx in range(-radius, radius + 1):
            xx = x + dx
            if 0 <= xx < WIDTH:
                row[xx] = 1


def bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        pts.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return pts


def draw_line(img: list[list[int]], x0: int, y0: int, x1: int, y1: int, radius: int, gap: int = 0) -> None:
    pts = bresenham(x0, y0, x1, y1)
    for i, (x, y) in enumerate(pts):
        if gap and (i % (gap + 2)) >= 2 and (i % (gap + 2)) < 2 + gap:
            continue
        stamp(img, x, y, radius)


def draw_ring(img: list[list[int]], cx: int, cy: int, radius: int, stroke: int) -> None:
    steps = max(24, int(2 * math.pi * max(radius, 1)) + 8)
    for i in range(steps):
        a = 2 * math.pi * i / steps
        x = int(round(cx + radius * math.cos(a)))
        y = int(round(cy + radius * math.sin(a)))
        stamp(img, x, y, stroke)


def draw_arc(img: list[list[int]], cx: int, cy: int, radius: int, stroke: int, a0: float, a1: float) -> None:
    steps = max(12, int(abs(a1 - a0) * max(radius, 1)) + 6)
    for i in range(steps + 1):
        a = a0 + (a1 - a0) * i / steps
        x = int(round(cx + radius * math.cos(a)))
        y = int(round(cy + radius * math.sin(a)))
        stamp(img, x, y, stroke)


def pack_bits(img: list[list[int]]) -> bytes:
    out = bytearray(128)
    i = 0
    for y in range(HEIGHT):
        row = img[y]
        for x in range(WIDTH):
            if row[x]:
                out[i // 8] |= 1 << (7 - (i % 8))
            i += 1
    return bytes(out)


def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def make_glyph(i: int, rng: random.Random) -> list[list[int]]:
    """One synthetic glyph-like shape. Layout depends on index; jitter from rng."""
    img = blank()
    family = i % 8
    stroke = 0 if (i % 3) == 0 else 1
    jx = rng.randint(-2, 2)
    jy = rng.randint(-2, 2)
    cx = _clamp(16 + jx, 6, 26)
    cy = _clamp(16 + jy, 6, 26)

    if family == 0:
        # straight stroke
        ang = (i * 23) % 180
        rad = math.radians(ang)
        length = 10 + (i % 5)
        dx = int(round(math.cos(rad) * length))
        dy = int(round(math.sin(rad) * length))
        draw_line(img, cx - dx, cy - dy, cx + dx, cy + dy, stroke)
    elif family == 1:
        # ring
        r = 6 + (i % 4)
        draw_ring(img, cx, cy, r, 0 if i % 2 == 0 else stroke)
    elif family == 2:
        # corner
        arm = 8 + (i % 5)
        draw_line(img, cx, cy, cx + arm, cy, stroke)
        draw_line(img, cx, cy, cx, cy + arm, stroke)
        if i % 4 == 2:
            draw_line(img, cx + arm, cy, cx + arm, cy + 4, 0)
    elif family == 3:
        # cross
        arm = 7 + (i % 4)
        draw_line(img, cx - arm, cy, cx + arm, cy, stroke)
        draw_line(img, cx, cy - arm, cx, cy + arm, stroke)
    elif family == 4:
        # broken stroke
        ang = (i * 17) % 160
        rad = math.radians(ang)
        length = 12 + (i % 4)
        dx = int(round(math.cos(rad) * length))
        dy = int(round(math.sin(rad) * length))
        draw_line(img, cx - dx, cy - dy, cx + dx, cy + dy, stroke, gap=2)
    elif family == 5:
        # T junction plus a short bar
        arm = 8 + (i % 3)
        draw_line(img, cx - arm, cy - 4, cx + arm, cy - 4, stroke)
        draw_line(img, cx, cy - 4, cx, cy + arm, stroke)
    elif family == 6:
        # diamond / rectangle outline
        r = 6 + (i % 3)
        if i % 2 == 0:
            draw_line(img, cx, cy - r, cx + r, cy, stroke)
            draw_line(img, cx + r, cy, cx, cy + r, stroke)
            draw_line(img, cx, cy + r, cx - r, cy, stroke)
            draw_line(img, cx - r, cy, cx, cy - r, stroke)
        else:
            draw_line(img, cx - r, cy - r, cx + r, cy - r, stroke)
            draw_line(img, cx + r, cy - r, cx + r, cy + r, stroke)
            draw_line(img, cx + r, cy + r, cx - r, cy + r, stroke)
            draw_line(img, cx - r, cy + r, cx - r, cy - r, stroke)
    else:
        # polyline / zigzag with a short broken tail
        y0 = _clamp(cy - 6, 3, 20)
        x = _clamp(cx - 10, 2, 12)
        pts = [(x, y0), (x + 6, y0 + 7), (x + 12, y0 + 1), (x + 18, y0 + 8)]
        for a, b in zip(pts, pts[1:]):
            draw_line(img, a[0], a[1], b[0], b[1], stroke)
        draw_arc(img, cx + 2, cy + 6, 4, 0, math.pi, 2 * math.pi)

    ink = sum(sum(row) for row in img)
    if ink < 12:
        draw_line(img, 4, 16, 28, 16, 1)
    return img


def ascii_image(img: list[list[int]]) -> str:
    return "\n".join("".join("#" if v else "." for v in row) for row in img)


def generate_all() -> list[bytes]:
    rng = random.Random(SEED)
    packed: list[bytes] = []
    seen: set[bytes] = set()
    for i in range(N_GLYPHS):
        img = make_glyph(i, rng)
        blob = pack_bits(img)
        if blob in seen or blob == bytes(128):
            # Deterministic distinctness notch. Still seed-driven, not compression-driven.
            stamp(img, 1 + (i % 28), 30 - (i % 4), 0)
            blob = pack_bits(img)
        if blob in seen or blob == bytes(128):
            raise RuntimeError(f"glyph {i} is empty or duplicate")
        seen.add(blob)
        packed.append(blob)
    return packed


def write_fixtures(dest: pathlib.Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    blobs = generate_all()
    for i, blob in enumerate(blobs):
        path = dest / f"glyph_{i:02d}.bin"
        if len(blob) != 128:
            raise RuntimeError(f"glyph {i} is {len(blob)} bytes")
        path.write_bytes(blob)
        print(f"wrote {path.name} {len(blob)} ink={bin(int.from_bytes(blob, 'big')).count('1')}")


if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parent
    write_fixtures(root / "fixtures")
    # ASCII preview is for the operator freezing the set; encoder never reads this.
    rng = random.Random(SEED)
    for i in range(N_GLYPHS):
        img = make_glyph(i, rng)
        print(f"\n== glyph_{i:02d} family={i % 8} ==")
        print(ascii_image(img))
