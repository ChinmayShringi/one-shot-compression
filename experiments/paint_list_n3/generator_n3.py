"""Family N3 fixture generator. Used only to CREATE rasters.

64x64 RGB, at most 6 filled shapes, palette size at most 8, no antialiasing,
hard edges, integer coordinates. Deterministic. Shapes do not share pixels.
No residual. No correction. No photos. No overlap rescue.

Seed 20260910. Images 0-15 fit, 16-23 held-out. The seed and this split are
frozen before any payload size is measured. The seed was not changed.

Held-out images whose index is even (16, 18, 20, 22) place two distinct
same-color rectangles that share a pixel edge. Corner-only contact is not
used. After painting they remain two shapes in the true program. A naive
4-connected color mask merges them. Other same-color shapes, if any, keep a
Chebyshev gap of at least 2.
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

MASTER_SEED = 20260910
N_IMAGES = 24
FIT_COUNT = 16
MAX_SHAPES = 6
MAX_PALETTE = 8
SETTINGS_ID = "paint-list-family-n3-v1"
FAMILY = "N3"
PLACE_ATTEMPTS = 192
SAME_COLOR_GAP = 2
SEED_NOTE = (
    "Family N3 uses seed 20260910. The seed was not changed. "
    "Images 0-15 fit, 16-23 held-out. Frozen before payload size measurement. "
    "Held-out image 16 places a same-color rectangle and ellipse that share a "
    "pixel edge. Held-out images 18, 20, 22 place two same-color rectangles that "
    "share a pixel edge. Those pairs remain two shapes in the true program. "
    "Corner-only contact does not count. No overlap."
)

# Frozen geometry if the RNG pair fails the shared-edge invariant.
# Not selected by payload size. Seed is unchanged.
FALLBACK_PAIRS = {
    16: ((1, 2, 14, 20), (15, 8, 28, 30)),
    18: ((4, 3, 20, 18), (21, 10, 34, 32)),
    20: ((2, 2, 22, 12), (8, 13, 30, 28)),
    22: ((3, 6, 24, 16), (10, 17, 36, 34)),
}


class Rng:
    """Numerical Recipes LCG. Same recurrence as Family N2."""

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


def require_shared_edge(image_index):
    """Frozen before size measurement. Four held-out images, at least three."""
    return image_index >= FIT_COUNT and (image_index % 2 == 0)


def split_name(image_index):
    return "fit" if image_index < FIT_COUNT else "heldout"


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


def pixels_share_edge(a, b):
    """4-connected adjacency. Corner-only contact returns False."""
    if len(a) > len(b):
        a, b = b, a
    bset = b if isinstance(b, set) else set(b)
    for x, y in a:
        if (x + 1, y) in bset or (x - 1, y) in bset or (x, y + 1) in bset or (x, y - 1) in bset:
            return True
    return False


def _on_canvas_rect(geom):
    x0, y0, x1, y1 = geom
    if x0 > x1 or y0 > y1:
        return False
    if x0 < 0 or y0 < 0 or x1 >= WIDTH or y1 >= HEIGHT:
        return False
    if (x1 - x0 + 1) < 4 or (y1 - y0 + 1) < 4:
        return False
    return True


def _union_is_rect(a, b):
    pts = set(a) | set(b)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return len(pts) == (x1 - x0 + 1) * (y1 - y0 + 1)


def pair_shares_edge(geom1, geom2):
    if not _on_canvas_rect(geom1) or not _on_canvas_rect(geom2):
        return False
    a = shape_pixels(TYPE_RECT, geom1)
    b = shape_pixels(TYPE_RECT, geom2)
    if not a or not b:
        return False
    sa, sb = set(a), set(b)
    if sa & sb:
        return False
    if not pixels_share_edge(sa, sb):
        return False
    if _union_is_rect(sa, sb):
        return False
    return True


def _sample_shared_pair(rng):
    kind = rng.randint(0, 1)
    w1 = rng.randint(6, 14)
    h1 = rng.randint(10, 22)
    w2 = rng.randint(6, 14)
    h2 = rng.randint(7, 18)
    ox = rng.randint(0, 6)
    oy = rng.randint(0, 6)
    shift = rng.randint(3, 9)
    if kind == 0:
        if abs(h1 - h2) < 3:
            h2 = h1 - 4 if h1 >= 12 else h1 + 4
        x0 = 1 + ox
        y0 = 1 + oy
        if y0 + h1 > HEIGHT:
            y0 = HEIGHT - h1
        x1 = x0 + w1 - 1
        y1 = y0 + h1 - 1
        x2 = x1 + 1
        y2 = y0 + shift
        if y2 + h2 > HEIGHT:
            y2 = HEIGHT - h2
        if y2 < 0:
            y2 = 0
        if x2 + w2 > WIDTH:
            return None
        geom1 = (x0, y0, x1, y1)
        geom2 = (x2, y2, x2 + w2 - 1, y2 + h2 - 1)
    else:
        if abs(w1 - w2) < 3:
            w2 = w1 - 4 if w1 >= 12 else w1 + 4
        x0 = 1 + ox
        y0 = 1 + oy
        if x0 + w1 > WIDTH:
            x0 = WIDTH - w1
        x1 = x0 + w1 - 1
        y1 = y0 + h1 - 1
        y2 = y1 + 1
        x2 = x0 + shift
        if x2 + w2 > WIDTH:
            x2 = WIDTH - w2
        if x2 < 0:
            x2 = 0
        if y2 + h2 > HEIGHT:
            return None
        geom1 = (x0, y0, x1, y1)
        geom2 = (x2, y2, x2 + w2 - 1, y2 + h2 - 1)
    if pair_shares_edge(geom1, geom2):
        return geom1, geom2
    return None


def shared_edge_pairs(scene):
    """Pairs of distinct same-color shapes whose pixels share a 4-edge."""
    groups = {}
    for idx, (typ, color_index, geom) in enumerate(scene["shapes"]):
        groups.setdefault(color_index, []).append(idx)
    pairs = []
    for color_index, idxs in groups.items():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a = scene["shapes"][idxs[i]]
                b = scene["shapes"][idxs[j]]
                pa = set(shape_pixels(a[0], a[2]))
                pb = set(shape_pixels(b[0], b[2]))
                if pa & pb:
                    continue
                if pixels_share_edge(pa, pb):
                    pairs.append((idxs[i], idxs[j], color_index))
    return pairs


# Recorded with seed 20260910. Not a fitter-constant sweep.
# Image 16 is the held-out non-rectangle shared-edge pair.
NONRECT_HELD_OUT = 16
NONRECT_RECT = (2, 10, 18, 40)
NONRECT_ELLIPSE = (19, 8, 42, 44)


def generate_nonrect_heldout(image_index, seed=MASTER_SEED):
    """Held-out rect + ellipse that share a 4-edge. Seed recorded. No fitter retune."""
    if image_index != NONRECT_HELD_OUT:
        raise ValueError("nonrect held-out index")
    rng = image_rng(image_index, seed)
    palette = _unique_palette(rng, 2)
    geom_rect = NONRECT_RECT
    geom_ellipse = NONRECT_ELLIPSE
    ra = set(shape_pixels(TYPE_RECT, geom_rect))
    ea = set(shape_pixels(TYPE_ELLIPSE, geom_ellipse))
    if not ra or not ea or (ra & ea) or not pixels_share_edge(ra, ea):
        raise RuntimeError("recorded nonrect pair does not share a 4-edge")
    scene = {
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
        "shared_edge_required": True,
        "used_fallback_pair": False,
        "nonrect_shared_edge": True,
        "palette": palette,
        "shapes": [
            (TYPE_RECT, 1, geom_rect),
            (TYPE_ELLIPSE, 1, geom_ellipse),
        ],
    }
    pairs = shared_edge_pairs(scene)
    if len(pairs) < 1:
        raise RuntimeError("held-out image missing shared edge")
    types = [scene["shapes"][i][0] for i, _j, _c in pairs] + [scene["shapes"][j][0] for i, j, _c in pairs]
    if all(t == TYPE_RECT for t in types):
        raise RuntimeError("held-out shared-edge pair is two rectangles")
    return scene


def generate_scene(image_index, seed=MASTER_SEED):
    if image_index < 0 or image_index >= N_IMAGES:
        raise ValueError("image_index out of range")
    if image_index == NONRECT_HELD_OUT:
        return generate_nonrect_heldout(image_index, seed)
    rng = image_rng(image_index, seed)
    shared_required = require_shared_edge(image_index)
    if shared_required:
        n_shapes_wanted = rng.randint(2, MAX_SHAPES)
        n_distinct = rng.randint(1, n_shapes_wanted - 1)
    else:
        n_shapes_wanted = rng.randint(1, MAX_SHAPES)
        n_distinct = n_shapes_wanted
    if n_distinct > MAX_PALETTE - 1:
        n_distinct = MAX_PALETTE - 1
    candidates = _unique_palette(rng, 1 + n_distinct)
    assignment = []
    if shared_required:
        assignment.extend([1, 1])
        while len(assignment) < n_shapes_wanted:
            assignment.append(rng.randint(1, n_distinct))
        # The forced pair owns slot 1. Later copies of slot 1 are refused
        # so exactly two shapes share that color and the shared edge.
    else:
        assignment = list(range(1, n_shapes_wanted + 1))

    occupied = set()
    same = {slot: set() for slot in range(1, n_distinct + 1)}
    shapes = []
    used_slots = {}
    used_colors = [candidates[0]]
    used_fallback = False

    def place(typ, geom, slot, allow_same_touch):
        pts = shape_pixels(typ, geom)
        if not pts or not _one_component(pts):
            return False
        sp = set(pts)
        if sp & occupied:
            return False
        if not allow_same_touch and _near_same_color(pts, same[slot]):
            return False
        occupied.update(sp)
        same[slot].update(sp)
        if slot not in used_slots:
            used_colors.append(candidates[slot])
            used_slots[slot] = len(used_colors) - 1
        shapes.append((typ, used_slots[slot], geom))
        return True

    if shared_required:
        sampled = _sample_shared_pair(rng)
        if sampled is None:
            geom1, geom2 = FALLBACK_PAIRS[image_index]
            used_fallback = True
        else:
            geom1, geom2 = sampled
        if not pair_shares_edge(geom1, geom2):
            geom1, geom2 = FALLBACK_PAIRS[image_index]
            used_fallback = True
        if not place(TYPE_RECT, geom1, 1, allow_same_touch=False):
            raise RuntimeError("forced shared-edge rect failed")
        if not place(TYPE_RECT, geom2, 1, allow_same_touch=True):
            raise RuntimeError("forced shared-edge mate failed")
        next_assign = 2
    else:
        next_assign = 0

    for slot in assignment[next_assign:]:
        if shared_required and slot == 1:
            continue
        placed = False
        for _attempt in range(PLACE_ATTEMPTS):
            typ = rng.randint(0, 2)
            geom = _sample_geom(rng, typ)
            if place(typ, geom, slot, allow_same_touch=False):
                placed = True
                break
        if not placed:
            continue

    scene = {
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
        "shared_edge_required": shared_required,
        "used_fallback_pair": used_fallback,
        "palette": used_colors,
        "shapes": shapes,
    }
    pairs = shared_edge_pairs(scene)
    if shared_required and len(pairs) < 1:
        raise RuntimeError("held-out image missing shared edge")
    if shared_required and len(shapes) < 2:
        raise RuntimeError("shared-edge pair collapsed")
    return scene


def _near_same_color(pts, same_pixels):
    if not same_pixels:
        return False
    for x, y in pts:
        for dy in range(-(SAME_COLOR_GAP - 1), SAME_COLOR_GAP):
            for dx in range(-(SAME_COLOR_GAP - 1), SAME_COLOR_GAP):
                if (x + dx, y + dy) in same_pixels:
                    return True
    return False


def render(scene):
    return render_scene(scene)


def masks_disjoint(scene):
    occupied = set()
    for typ, _color, geom in scene["shapes"]:
        pts = shape_pixels(typ, geom)
        for p in pts:
            if p in occupied:
                return False
            occupied.add(p)
    return True
