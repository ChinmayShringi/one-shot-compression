"""Family N2: non-overlapping filled shapes, colors may repeat.

64x64 RGB, at most 6 filled shapes, palette size at most 8, no antialiasing,
hard edges, integer coordinates. Deterministic. Shapes do not share pixels.

Seed 20260909. Images 0-15 fit, 16-23 held-out. The seed and this split are
frozen before any payload size is measured. The seed was not changed.

A color may be used by more than one shape. Same-color shapes are placed with
a Chebyshev gap of at least 2 so their pixel sets are distinct 8-connected
components. Every held-out image is required, by this frozen rule, to place
at least two shapes that share one palette color. Images are not dropped.
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

MASTER_SEED = 20260909
N_IMAGES = 24
FIT_COUNT = 16
MAX_SHAPES = 6
MAX_PALETTE = 8
SETTINGS_ID = "paint-list-family-n2-v1"
FAMILY = "N2"
PLACE_ATTEMPTS = 192
SAME_COLOR_GAP = 2
SEED_NOTE = (
    "Family N2 uses seed 20260909. The seed was not changed. "
    "Images 0-15 fit, 16-23 held-out. Held-out images are required by this "
    "frozen rule to place at least two shapes that share one palette color. "
    "Same-color shapes keep a Chebyshev gap of at least 2. Images are not dropped."
)


class Rng:
    """Numerical Recipes LCG. Same recurrence as Family N."""

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


def require_reuse(image_index):
    """Frozen before size measurement. All held-out images must reuse a color."""
    return image_index >= FIT_COUNT or (image_index % 3 == 0)


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


def _one_component(pts):
    if not pts:
        return False
    remaining = set(pts)
    start = pts[0]
    remaining.remove(start)
    stack = [start]
    seen = 1
    while stack:
        x, y = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                q = (x + dx, y + dy)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    seen += 1
    return seen == len(pts) and not remaining


def _hits_occupied(pts, occupied):
    for p in pts:
        if p in occupied:
            return True
    return False


def _near_same_color(pts, same_pixels):
    if not same_pixels:
        return False
    for x, y in pts:
        for dy in range(-(SAME_COLOR_GAP - 1), SAME_COLOR_GAP):
            for dx in range(-(SAME_COLOR_GAP - 1), SAME_COLOR_GAP):
                if (x + dx, y + dy) in same_pixels:
                    return True
    return False


def _forced_reuse_rects(rng):
    """Two filled rects that share a color, disjoint, Chebyshev gap at least 3.

    Geometry is drawn from the image RNG and clamped into opposite halves so
    the reuse constraint does not depend on later size measurements.
    """
    w1 = rng.randint(4, 18)
    h1 = rng.randint(4, 22)
    x0 = rng.randint(0, 26 - w1)
    y0 = rng.randint(0, HEIGHT - h1)
    w2 = rng.randint(4, 18)
    h2 = rng.randint(4, 22)
    x2 = rng.randint(34, WIDTH - w2)
    y2 = rng.randint(0, HEIGHT - h2)
    return (
        (TYPE_RECT, (x0, y0, x0 + w1 - 1, y0 + h1 - 1)),
        (TYPE_RECT, (x2, y2, x2 + w2 - 1, y2 + h2 - 1)),
    )


def reuses_color(scene):
    counts = {}
    for _typ, color_index, _geom in scene["shapes"]:
        counts[color_index] = counts.get(color_index, 0) + 1
    return any(n >= 2 for n in counts.values())


def generate_scene(image_index, seed=MASTER_SEED):
    if image_index < 0 or image_index >= N_IMAGES:
        raise ValueError("image_index out of range")
    rng = image_rng(image_index, seed)
    reuse_required = require_reuse(image_index)
    if reuse_required:
        n_shapes_wanted = rng.randint(2, MAX_SHAPES)
        n_distinct = rng.randint(1, n_shapes_wanted - 1)
    else:
        n_shapes_wanted = rng.randint(1, MAX_SHAPES)
        n_distinct = n_shapes_wanted
    if n_distinct > MAX_PALETTE - 1:
        n_distinct = MAX_PALETTE - 1
    candidates = _unique_palette(rng, 1 + n_distinct)
    assignment = []
    if reuse_required:
        assignment.extend([1, 1])
        while len(assignment) < n_shapes_wanted:
            assignment.append(rng.randint(1, n_distinct))
    else:
        assignment = list(range(1, n_shapes_wanted + 1))

    occupied = set()
    same = {slot: set() for slot in range(1, n_distinct + 1)}
    shapes = []
    used_slots = {}
    used_colors = [candidates[0]]

    def place(typ, geom, slot):
        pts = shape_pixels(typ, geom)
        if not pts or not _one_component(pts):
            return False
        if _hits_occupied(pts, occupied):
            return False
        if _near_same_color(pts, same[slot]):
            return False
        occupied.update(pts)
        same[slot].update(pts)
        if slot not in used_slots:
            used_colors.append(candidates[slot])
            used_slots[slot] = len(used_colors) - 1
        shapes.append((typ, used_slots[slot], geom))
        return True

    next_assign = 0
    if reuse_required:
        for typ, geom in _forced_reuse_rects(rng):
            if not place(typ, geom, 1):
                raise RuntimeError("forced reuse rects collided")
        next_assign = 2

    for slot in assignment[next_assign:]:
        placed = False
        for _attempt in range(PLACE_ATTEMPTS):
            typ = rng.randint(0, 2)
            geom = _sample_geom(rng, typ)
            if place(typ, geom, slot):
                placed = True
                break
        if not placed:
            continue

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
        "same_color_gap": SAME_COLOR_GAP,
        "reuse_required": reuse_required,
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
