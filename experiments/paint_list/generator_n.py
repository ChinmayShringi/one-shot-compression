"""Family N: non-overlapping filled shapes.

Same constraints as Family O: 64x64 RGB, at most 6 filled shapes, palette
size at most 8, no antialiasing, hard edges, integer coordinates. Deterministic.

Seed 20260908. Images 0-15 fit, 16-23 held-out. Shapes are placed by rejection
so their rasterized pixel sets are disjoint. That is geometric non-overlap for
this rasterizer: inclusive hard-edge fills do not share a pixel. Paint order
does not change the raster.

Each placed shape gets a color distinct from the background and from every
other placed shape, so a color mask is exactly one shape. Unused palette
candidates are dropped; the image is never dropped. Placement failure stores
fewer shapes, not a missing image.
"""

from __future__ import annotations

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

MASTER_SEED = 20260908
N_IMAGES = 24
FIT_COUNT = 16
MAX_SHAPES = 6
MAX_PALETTE = 8
SETTINGS_ID = "paint-list-family-n-v1"
FAMILY = "N"
PLACE_ATTEMPTS = 96
SEED_NOTE = (
    "Family N uses seed 20260908. Rejection sampling consumes the image RNG "
    "until each shape is disjoint or the attempt budget ends. The seed was not changed."
)


class Rng:
    """Numerical Recipes LCG. Same recurrence as Family O."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
        if self.s == 0:
            self.s = 1

    def next_u32(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s

    def randint(self, lo, hi):
        span = hi - lo + 1
        return lo + (self.next_u32() % span)


def image_rng(image_index, seed=MASTER_SEED):
    state = (seed + (image_index + 1) * 0x9E3779B9) & 0xFFFFFFFF
    rng = Rng(state)
    rng.next_u32()
    rng.next_u32()
    return rng


def _unique_palette(rng, n_colors):
    palette = []
    seen = set()
    guard = 0
    while len(palette) < n_colors and guard < 100000:
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        guard += 1
        if color in seen:
            continue
        seen.add(color)
        palette.append(color)
    bump = 0
    while len(palette) < n_colors:
        color = (bump & 255, (bump >> 8) & 255, (bump >> 16) & 255)
        bump += 1
        if color not in seen:
            seen.add(color)
            palette.append(color)
    return palette


def shape_pixels(typ, geom):
    pts = []
    if typ in (TYPE_RECT, TYPE_ELLIPSE):
        x0, y0, x1, y1 = geom
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        x0c = 0 if x0 < 0 else x0
        y0c = 0 if y0 < 0 else y0
        x1c = WIDTH - 1 if x1 >= WIDTH else x1
        y1c = HEIGHT - 1 if y1 >= HEIGHT else y1
        if typ == TYPE_RECT:
            for y in range(y0c, y1c + 1):
                for x in range(x0c, x1c + 1):
                    pts.append((x, y))
        else:
            for y in range(y0c, y1c + 1):
                for x in range(x0c, x1c + 1):
                    if pixel_in_ellipse(x, y, x0, y0, x1, y1):
                        pts.append((x, y))
    else:
        v0, v1, v2 = geom
        xs = (v0[0], v1[0], v2[0])
        ys = (v0[1], v1[1], v2[1])
        x0 = 0 if min(xs) < 0 else min(xs)
        y0 = 0 if min(ys) < 0 else min(ys)
        x1 = WIDTH - 1 if max(xs) >= WIDTH else max(xs)
        y1 = HEIGHT - 1 if max(ys) >= HEIGHT else max(ys)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if pixel_in_triangle(x, y, v0, v1, v2):
                    pts.append((x, y))
    return pts


def _sample_geom(rng, typ):
    if typ == TYPE_TRIANGLE:
        return (
            (rng.randint(0, WIDTH - 1), rng.randint(0, HEIGHT - 1)),
            (rng.randint(0, WIDTH - 1), rng.randint(0, HEIGHT - 1)),
            (rng.randint(0, WIDTH - 1), rng.randint(0, HEIGHT - 1)),
        )
    x0 = rng.randint(0, WIDTH - 1)
    y0 = rng.randint(0, HEIGHT - 1)
    x1 = rng.randint(0, WIDTH - 1)
    y1 = rng.randint(0, HEIGHT - 1)
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)


def generate_scene(image_index, seed=MASTER_SEED):
    if image_index < 0 or image_index >= N_IMAGES:
        raise ValueError("image_index out of range")
    rng = image_rng(image_index, seed)
    n_shapes_wanted = rng.randint(1, MAX_SHAPES)
    # Background plus one unique color per wanted shape. Cap at the palette limit.
    n_colors = n_shapes_wanted + 1
    if n_colors > MAX_PALETTE:
        n_colors = MAX_PALETTE
        n_shapes_wanted = n_colors - 1
    candidates = _unique_palette(rng, n_colors)
    occupied = set()
    shapes = []
    used_colors = [candidates[0]]
    for color_index in range(1, n_colors):
        placed = None
        for _attempt in range(PLACE_ATTEMPTS):
            typ = rng.randint(0, 2)
            geom = _sample_geom(rng, typ)
            pts = shape_pixels(typ, geom)
            if not pts:
                continue
            hit = False
            for p in pts:
                if p in occupied:
                    hit = True
                    break
            if hit:
                continue
            occupied.update(pts)
            placed = (typ, geom)
            break
        if placed is None:
            continue
        typ, geom = placed
        used_colors.append(candidates[color_index])
        shapes.append((typ, len(used_colors) - 1, geom))
    return {
        "family": FAMILY,
        "settings_id": SETTINGS_ID,
        "seed": seed,
        "seed_note": SEED_NOTE,
        "image_index": image_index,
        "width": WIDTH,
        "height": HEIGHT,
        "max_shapes": MAX_SHAPES,
        "max_palette": MAX_PALETTE,
        "place_attempts": PLACE_ATTEMPTS,
        "palette": used_colors,
        "shapes": shapes,
    }


def render(scene):
    return render_scene(scene)


def split_name(image_index):
    return "fit" if image_index < FIT_COUNT else "heldout"


def masks_disjoint(scene):
    occupied = set()
    for typ, _color, geom in scene["shapes"]:
        pts = shape_pixels(typ, geom)
        for p in pts:
            if p in occupied:
                return False
            occupied.add(p)
    return True
