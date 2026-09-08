"""Image-only ordered paint-list codec. No correction bytes.

The encoder sees pixels only. It recovers a palette and an ordered list of
filled shapes (type, geometry, color index). Decode redraws that program with
the shared rasterizer. There is no XOR plane, residual, or extra correction
field. If no such program redraws the image exactly, encode returns None and
the image is not exact.

Recovery: palette colors in first-seen scan order. Try each color as the
background, which the rasterizer paints as palette[0]. Each other color's
pixel set must be exactly one filled rect, ellipse, or triangle. Shapes of
the accepted program are stored in a deterministic recovered order. Overlap
that clips a shape so the visible color mask is not one primitive fails
exactness. Paint order is not recovered from a raster that does not
determine it.
"""

from __future__ import annotations

import struct

from raster import (
    HEIGHT,
    TYPE_ELLIPSE,
    TYPE_RECT,
    TYPE_TRIANGLE,
    WIDTH,
    pixel_in_ellipse,
    pixel_in_triangle,
    render_scene,
)

MAGIC = b"PLN1"
VERSION = 1
SETTINGS_CODE = 1
SETTINGS_ID = "paint-list-nocorr-v1"
MAX_SHAPES = 6
ELLIPSE_MARGIN = 4
# Outward local offsets that covered generator triangles in seed sweeps.
# Chebyshev bound around the max-area hull corner, in the corner's outward frame.
TRI_LO = -2
TRI_HI = 6
LINE_RADIUS = 8


def _pack_shape(typ, color_index, geom):
    if typ in (TYPE_RECT, TYPE_ELLIPSE):
        coords = (geom[0], geom[1], geom[2], geom[3])
    elif typ == TYPE_TRIANGLE:
        coords = (
            geom[0][0],
            geom[0][1],
            geom[1][0],
            geom[1][1],
            geom[2][0],
            geom[2][1],
        )
    else:
        raise ValueError("bad shape type")
    if any(c < 0 or c > 255 for c in coords):
        raise ValueError("geometry out of range")
    if color_index < 0 or color_index > 255:
        raise ValueError("color index")
    return bytes([typ, color_index]) + bytes(coords)


def encode_program(scene):
    palette = scene["palette"]
    shapes = scene["shapes"]
    width = int(scene["width"])
    height = int(scene["height"])
    if not palette or len(palette) > 255 or len(shapes) > 255:
        raise ValueError("program size")
    out = bytearray()
    out += MAGIC
    out.append(VERSION)
    out.append(SETTINGS_CODE)
    out += struct.pack("<HHBB", width, height, len(palette), len(shapes))
    for r, g, b in palette:
        out += bytes((r & 255, g & 255, b & 255))
    for typ, color_index, geom in shapes:
        out += _pack_shape(typ, color_index, geom)
    return bytes(out)


def decode(payload):
    if payload[:4] != MAGIC or payload[4] != VERSION:
        raise ValueError("bad paint-list payload")
    if payload[5] != SETTINGS_CODE:
        raise ValueError("bad settings code")
    width, height, n_colors, n_shapes = struct.unpack_from("<HHBB", payload, 6)
    off = 6 + struct.calcsize("<HHBB")
    palette = []
    for _ in range(n_colors):
        palette.append((payload[off], payload[off + 1], payload[off + 2]))
        off += 3
    shapes = []
    for _ in range(n_shapes):
        typ = payload[off]
        color_index = payload[off + 1]
        off += 2
        if typ in (TYPE_RECT, TYPE_ELLIPSE):
            geom = (payload[off], payload[off + 1], payload[off + 2], payload[off + 3])
            off += 4
        elif typ == TYPE_TRIANGLE:
            geom = (
                (payload[off], payload[off + 1]),
                (payload[off + 2], payload[off + 3]),
                (payload[off + 4], payload[off + 5]),
            )
            off += 6
        else:
            raise ValueError("bad shape type")
        shapes.append((typ, color_index, geom))
    if off != len(payload):
        raise ValueError("trailing bytes in paint-list payload")
    scene = {
        "settings_id": SETTINGS_ID,
        "width": width,
        "height": height,
        "palette": palette,
        "shapes": shapes,
    }
    return render_scene(scene)


def _colors(px, width, height):
    order = []
    index_of = {}
    buckets = []
    for y in range(height):
        row = y * width * 3
        for x in range(width):
            o = row + x * 3
            color = (px[o], px[o + 1], px[o + 2])
            idx = index_of.get(color)
            if idx is None:
                idx = len(order)
                index_of[color] = idx
                order.append(color)
                buckets.append([])
            buckets[idx].append((x, y))
    return order, buckets


def _bbox(pts):
    x0 = y0 = 10**9
    x1 = y1 = -1
    for x, y in pts:
        if x < x0:
            x0 = x
        if x > x1:
            x1 = x
        if y < y0:
            y0 = y
        if y > y1:
            y1 = y
    return x0, y0, x1, y1


def _rect_equals(pts, x0, y0, x1, y1):
    if (x1 - x0 + 1) * (y1 - y0 + 1) != len(pts):
        return False
    return True


def _fit_rect(pts):
    x0, y0, x1, y1 = _bbox(pts)
    if _rect_equals(pts, x0, y0, x1, y1):
        return (TYPE_RECT, (x0, y0, x1, y1))
    return None


def _ellipse_count(x0, y0, x1, y1, bits, width):
    n = 0
    for y in range(y0, y1 + 1):
        row = y * width
        for x in range(x0, x1 + 1):
            if pixel_in_ellipse(x, y, x0, y0, x1, y1):
                if not bits[row + x]:
                    return -1
                n += 1
    return n


def _fit_ellipse(pts, bits, width, height):
    bx0, by0, bx1, by1 = _bbox(pts)
    want = len(pts)
    margin = ELLIPSE_MARGIN
    for left in range(0, margin + 1):
        for right in range(0, margin + 1):
            for top in range(0, margin + 1):
                for bot in range(0, margin + 1):
                    x0 = bx0 - left
                    x1 = bx1 + right
                    y0 = by0 - top
                    y1 = by1 + bot
                    if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
                        continue
                    n = _ellipse_count(x0, y0, x1, y1, bits, width)
                    if n == want:
                        return (TYPE_ELLIPSE, (x0, y0, x1, y1))
    return None


def _convex_hull(points):
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _area2(a, b, c):
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _max_area_corners(hull):
    n = len(hull)
    best = None
    for i in range(n):
        ai = hull[i]
        for j in range(i + 1, n):
            aj = hull[j]
            for k in range(j + 1, n):
                ak = hull[k]
                area = _area2(ai, aj, ak)
                if best is None or area > best[0]:
                    best = (area, ai, aj, ak)
    return (best[1], best[2], best[3])


def _triangle_covers(pts, bits, width, v0, v1, v2, want):
    xs = (v0[0], v1[0], v2[0])
    ys = (v0[1], v1[1], v2[1])
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    if x0 < 0 or y0 < 0 or x1 >= WIDTH or y1 >= HEIGHT:
        return False
    for x, y in pts:
        if x < x0 or x > x1 or y < y0 or y > y1:
            return False
        if not pixel_in_triangle(x, y, v0, v1, v2):
            return False
    n = 0
    for y in range(y0, y1 + 1):
        row = y * width
        for x in range(x0, x1 + 1):
            if pixel_in_triangle(x, y, v0, v1, v2):
                if not bits[row + x]:
                    return False
                n += 1
                if n > want:
                    return False
    return n == want


def _outward_window(corner, others, width, height):
    cx = (corner[0] + others[0][0] + others[1][0]) / 3.0
    cy = (corner[1] + others[0][1] + others[1][1]) / 3.0
    sx = 1 if corner[0] >= cx else -1
    sy = 1 if corner[1] >= cy else -1
    cands = []
    for ox in range(TRI_LO, TRI_HI + 1):
        for oy in range(TRI_LO, TRI_HI + 1):
            x = corner[0] + ox * sx
            y = corner[1] + oy * sy
            if 0 <= x < width and 0 <= y < height:
                cands.append((abs(ox) + abs(oy), x, y))
    cands.sort()
    seen = set()
    uniq = []
    for _d, x, y in cands:
        if (x, y) not in seen:
            seen.add((x, y))
            uniq.append((x, y))
    return uniq


def _window_around(p, radius, width, height):
    out = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x = p[0] + dx
            y = p[1] + dy
            if 0 <= x < width and 0 <= y < height:
                out.append((abs(dx) + abs(dy), x, y))
    out.sort()
    seen = set()
    uniq = []
    for _d, x, y in out:
        if (x, y) not in seen:
            seen.add((x, y))
            uniq.append((x, y))
    return uniq


def _fit_degenerate_line(pts, bits, width, height):
    """Collinear generator triangles rasterize as a line. Hull has fewer than 3 points.

    Store a degenerate triangle (endpoint, endpoint, repeated endpoint). The
    repeated vertex is geometry, not a residual.
    """
    if not pts:
        return None
    want = len(pts)
    poles = [
        min(pts),
        max(pts),
        min(pts, key=lambda q: (q[0] + q[1], q[0], q[1])),
        max(pts, key=lambda q: (q[0] + q[1], q[0], q[1])),
        min(pts, key=lambda q: (q[0] - q[1], q[1], q[0])),
        max(pts, key=lambda q: (q[0] - q[1], q[1], q[0])),
    ]
    # Furthest pole pair is the visible extent of the line.
    best = None
    for i in range(len(poles)):
        for j in range(i + 1, len(poles)):
            a, b = poles[i], poles[j]
            d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if best is None or d > best[0]:
                best = (d, a, b)
    if best is None:
        return None
    _d, a0, b0 = best
    wa = _window_around(a0, LINE_RADIUS, width, height)
    wb = _window_around(b0, LINE_RADIUS, width, height)
    for a in wa:
        for b in wb:
            if a == b:
                continue
            if _triangle_covers(pts, bits, width, a, b, a, want):
                verts = tuple(sorted((a, b, a)))
                return (TYPE_TRIANGLE, (verts[0], verts[1], verts[2]))
    return None


def _fit_triangle(pts, bits, width, height):
    if not pts:
        return None
    hull = _convex_hull(pts)
    if len(hull) < 3:
        return _fit_degenerate_line(pts, bits, width, height)
    corners = _max_area_corners(hull)
    wins = [
        _outward_window(corners[0], (corners[1], corners[2]), width, height),
        _outward_window(corners[1], (corners[0], corners[2]), width, height),
        _outward_window(corners[2], (corners[0], corners[1]), width, height),
    ]
    want = len(pts)
    for a in wins[0]:
        for b in wins[1]:
            if a == b:
                continue
            for c in wins[2]:
                if c == a or c == b or _area2(a, b, c) == 0:
                    continue
                if _triangle_covers(pts, bits, width, a, b, c, want):
                    verts = tuple(sorted((a, b, c)))
                    return (TYPE_TRIANGLE, (verts[0], verts[1], verts[2]))
    return _fit_degenerate_line(pts, bits, width, height)


def _bits_of(pts, width, height):
    bits = bytearray(width * height)
    for x, y in pts:
        bits[y * width + x] = 1
    return bits


def fit_color_mask(pts, width, height):
    if not pts:
        return None
    bits = _bits_of(pts, width, height)
    got = _fit_rect(pts)
    if got:
        return got
    got = _fit_ellipse(pts, bits, width, height)
    if got:
        return got
    return _fit_triangle(pts, bits, width, height)


def _anchor(typ, geom):
    if typ in (TYPE_RECT, TYPE_ELLIPSE):
        return (geom[1], geom[0], typ, geom)
    ys = (geom[0][1], geom[1][1], geom[2][1])
    xs = (geom[0][0], geom[1][0], geom[2][0])
    return (min(ys), min(xs), typ, geom)


def recover_program(px, width, height):
    """Return a scene that redraws px, or None if no exact paint list is recovered."""
    if width != WIDTH or height != HEIGHT:
        return None
    colors, buckets = _colors(px, width, height)
    if not colors or len(colors) > 255:
        return None
    # A color mask is a primitive or not, independent of which color is background.
    fitted_by_color = [fit_color_mask(pts, width, height) for pts in buckets]
    accepted = None
    for bg_index, bg in enumerate(colors):
        shapes = []
        ok = True
        for cidx, color in enumerate(colors):
            if cidx == bg_index:
                continue
            fitted = fitted_by_color[cidx]
            if fitted is None:
                ok = False
                break
            shapes.append((fitted[0], color, fitted[1]))
            if len(shapes) > MAX_SHAPES:
                ok = False
                break
        if not ok:
            continue
        palette = [bg] + [c for i, c in enumerate(colors) if i != bg_index]
        remap = {color: i for i, color in enumerate(palette)}
        ordered = []
        for typ, color, geom in shapes:
            ordered.append((typ, remap[color], geom))
        ordered.sort(key=lambda s: _anchor(s[0], s[2]) + (s[1],))
        scene = {
            "settings_id": SETTINGS_ID,
            "width": width,
            "height": height,
            "palette": palette,
            "shapes": ordered,
        }
        drawn = render_scene(scene)
        if drawn == px:
            # Fewest shapes, then earliest background in scan order.
            key = (len(ordered), bg_index)
            if accepted is None or key < accepted[0]:
                accepted = (key, scene)
    if accepted is None:
        return None
    return accepted[1]


def encode(px, width, height):
    """Recover and serialize a paint list. None means this image is not exact."""
    scene = recover_program(px, width, height)
    if scene is None:
        return None
    payload = encode_program(scene)
    # Refuse to store a program that does not redraw exactly.
    if decode(payload) != px:
        return None
    return payload
