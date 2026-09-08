"""Shared raster helpers for 32x32 binary glyphs.

Used by both encode and decode. No generator, no seed, no network.
"""

from __future__ import annotations

WIDTH = 32
HEIGHT = 32
NPIX = WIDTH * HEIGHT
PACKED_LEN = NPIX // 8  # 128

# 8-connected chain codes. Index is the stored 3-bit code.
# x increases right, y increases down.
# 0 E, 1 NE, 2 N, 3 NW, 4 W, 5 SW, 6 S, 7 SE
DIRS = (
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)
DIR_INDEX = {d: i for i, d in enumerate(DIRS)}


def new_image(width: int = WIDTH, height: int = HEIGHT) -> list[list[int]]:
    return [[0] * width for _ in range(height)]


def clone(img: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in img]


def unpack_bits(data: bytes, width: int = WIDTH, height: int = HEIGHT) -> list[list[int]]:
    if width <= 0 or height <= 0:
        raise ValueError("invalid dimensions")
    expected = (width * height + 7) // 8
    if len(data) != expected:
        raise ValueError(f"expected {expected} packed bytes, got {len(data)}")
    img = new_image(width, height)
    for i in range(width * height):
        byte = data[i // 8]
        bit = 7 - (i % 8)
        if (byte >> bit) & 1:
            img[i // width][i % width] = 1
    return img


def pack_bits(img: list[list[int]], width: int | None = None, height: int | None = None) -> bytes:
    height = height if height is not None else len(img)
    width = width if width is not None else (len(img[0]) if height else 0)
    n = width * height
    out = bytearray((n + 7) // 8)
    for y in range(height):
        row = img[y]
        for x in range(width):
            if row[x]:
                i = y * width + x
                out[i // 8] |= 1 << (7 - (i % 8))
    return bytes(out)


def xor_images(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    h = len(a)
    w = len(a[0]) if h else 0
    out = new_image(w, h)
    for y in range(h):
        ra = a[y]
        rb = b[y]
        ro = out[y]
        for x in range(w):
            ro[x] = 1 if ra[x] != rb[x] else 0
    return out


def images_equal(a: list[list[int]], b: list[list[int]]) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if ra != rb:
            return False
    return True


def dilate(img: list[list[int]], radius: int) -> list[list[int]]:
    """Chebyshev (square) dilation. radius 0 is a copy. Out-of-bounds ignored."""
    h = len(img)
    w = len(img[0]) if h else 0
    if radius <= 0:
        return clone(img)
    out = new_image(w, h)
    for y in range(h):
        row = img[y]
        for x in range(w):
            if not row[x]:
                continue
            y0 = y - radius
            y1 = y + radius
            x0 = x - radius
            x1 = x + radius
            if y0 < 0:
                y0 = 0
            if x0 < 0:
                x0 = 0
            if y1 >= h:
                y1 = h - 1
            if x1 >= w:
                x1 = w - 1
            for yy in range(y0, y1 + 1):
                orow = out[yy]
                for xx in range(x0, x1 + 1):
                    orow[xx] = 1
    return out


def zhang_suen(img: list[list[int]]) -> list[list[int]]:
    """Zhang-Suen thinning. Foreground is 1. Outside the image is background."""
    h = len(img)
    w = len(img[0]) if h else 0
    skel = clone(img)
    if h == 0 or w == 0:
        return skel

    def neighbors(x: int, y: int) -> list[int]:
        # P2, P3, P4, P5, P6, P7, P8, P9 (N, NE, E, SE, S, SW, W, NW)
        def at(xx: int, yy: int) -> int:
            if 0 <= xx < w and 0 <= yy < h:
                return skel[yy][xx]
            return 0

        return [
            at(x, y - 1),
            at(x + 1, y - 1),
            at(x + 1, y),
            at(x + 1, y + 1),
            at(x, y + 1),
            at(x - 1, y + 1),
            at(x - 1, y),
            at(x - 1, y - 1),
        ]

    def transitions(n: list[int]) -> int:
        count = 0
        for i in range(8):
            if n[i] == 0 and n[(i + 1) % 8] == 1:
                count += 1
        return count

    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            remove: list[tuple[int, int]] = []
            for y in range(h):
                row = skel[y]
                for x in range(w):
                    if row[x] != 1:
                        continue
                    n = neighbors(x, y)
                    b = n[0] + n[1] + n[2] + n[3] + n[4] + n[5] + n[6] + n[7]
                    if b < 2 or b > 6:
                        continue
                    if transitions(n) != 1:
                        continue
                    if step == 0:
                        if n[0] * n[2] * n[4] != 0:
                            continue
                        if n[2] * n[4] * n[6] != 0:
                            continue
                    else:
                        if n[0] * n[2] * n[6] != 0:
                            continue
                        if n[0] * n[4] * n[6] != 0:
                            continue
                    remove.append((x, y))
            if remove:
                changed = True
                for x, y in remove:
                    skel[y][x] = 0
    return skel


def ascii_image(img: list[list[int]]) -> str:
    lines = []
    for row in img:
        lines.append("".join("#" if v else "." for v in row))
    return "\n".join(lines)
