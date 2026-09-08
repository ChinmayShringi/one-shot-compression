"""Hard-edge integer rasterizer for filled shapes.

Copied from experiments/paint_list_n2_encode/raster.py.
No antialiasing. Coordinates are integers. Inclusive bounds.
"""

from __future__ import annotations

WIDTH = 64
HEIGHT = 64

TYPE_RECT = 0
TYPE_ELLIPSE = 1
TYPE_TRIANGLE = 2


def new_image(width=WIDTH, height=HEIGHT, color=(0, 0, 0)):
    r, g, b = color
    px = bytearray(width * height * 3)
    for i in range(0, len(px), 3):
        px[i] = r
        px[i + 1] = g
        px[i + 2] = b
    return px


def get_rgb(px, width, x, y):
    i = (y * width + x) * 3
    return (px[i], px[i + 1], px[i + 2])


def set_rgb(px, width, x, y, color):
    i = (y * width + x) * 3
    px[i] = color[0]
    px[i + 1] = color[1]
    px[i + 2] = color[2]


def fill_rect(px, width, height, x0, y0, x1, y1, color):
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    x0 = 0 if x0 < 0 else x0
    y0 = 0 if y0 < 0 else y0
    x1 = width - 1 if x1 >= width else x1
    y1 = height - 1 if y1 >= height else y1
    if x0 > x1 or y0 > y1:
        return
    for y in range(y0, y1 + 1):
        row = (y * width + x0) * 3
        for _x in range(x1 - x0 + 1):
            px[row] = color[0]
            px[row + 1] = color[1]
            px[row + 2] = color[2]
            row += 3


def pixel_in_ellipse(x, y, x0, y0, x1, y1):
    if x1 < x0 or y1 < y0:
        return False
    if x < x0 or x > x1 or y < y0 or y > y1:
        return False
    if x0 == x1 and y0 == y1:
        return True
    rx = x1 - x0 + 1
    ry = y1 - y0 + 1
    dx = 2 * x - x0 - x1
    dy = 2 * y - y0 - y1
    return dx * dx * ry * ry + dy * dy * rx * rx <= rx * rx * ry * ry


def fill_ellipse(px, width, height, x0, y0, x1, y1, color):
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
                set_rgb(px, width, x, y, color)


def _orient(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def pixel_in_triangle(x, y, v0, v1, v2):
    px, py = 2 * x + 1, 2 * y + 1
    ax, ay = 2 * v0[0], 2 * v0[1]
    bx, by = 2 * v1[0], 2 * v1[1]
    cx, cy = 2 * v2[0], 2 * v2[1]
    o0 = _orient(ax, ay, bx, by, px, py)
    o1 = _orient(bx, by, cx, cy, px, py)
    o2 = _orient(cx, cy, ax, ay, px, py)
    has_neg = (o0 < 0) or (o1 < 0) or (o2 < 0)
    has_pos = (o0 > 0) or (o1 > 0) or (o2 > 0)
    return not (has_neg and has_pos)


def fill_triangle(px, width, height, v0, v1, v2, color):
    xs = (v0[0], v1[0], v2[0])
    ys = (v0[1], v1[1], v2[1])
    x0 = max(0, min(xs))
    y0 = max(0, min(ys))
    x1 = min(width - 1, max(xs))
    y1 = min(height - 1, max(ys))
    if x0 > x1 or y0 > y1:
        return
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if pixel_in_triangle(x, y, v0, v1, v2):
                set_rgb(px, width, x, y, color)


def render_scene(scene):
    """Redraw a stored scene. scene has width, height, palette, shapes.

    shapes: list of (type, color_index, geometry).
    geometry is (x0,y0,x1,y1) for rect/ellipse or ((x,y),(x,y),(x,y)) for triangle.
    Background is palette[0], painted first. Later shapes overwrite.
    """
    width = scene["width"]
    height = scene["height"]
    palette = scene["palette"]
    px = new_image(width, height, palette[0])
    for typ, color_index, geom in scene["shapes"]:
        color = palette[color_index]
        if typ == TYPE_RECT:
            fill_rect(px, width, height, geom[0], geom[1], geom[2], geom[3], color)
        elif typ == TYPE_ELLIPSE:
            fill_ellipse(px, width, height, geom[0], geom[1], geom[2], geom[3], color)
        elif typ == TYPE_TRIANGLE:
            fill_triangle(px, width, height, geom[0], geom[1], geom[2], color)
        else:
            raise ValueError("unknown shape type %s" % typ)
    return px


def mismatch_count(a, b):
    if len(a) != len(b):
        return max(len(a), len(b)) // 3
    n = 0
    for i in range(0, len(a), 3):
        if a[i] != b[i] or a[i + 1] != b[i + 1] or a[i + 2] != b[i + 2]:
            n += 1
    return n


def pixels_equal(a, b):
    return a == b
