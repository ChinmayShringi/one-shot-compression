"""Deterministic filled-shape generator.

Seed 20260908. 24 images, 64x64 RGB. At most 6 shapes, palette size at most 8.
No antialiasing, hard edges, integer coordinates. Does not reject images by size.
"""

from __future__ import annotations

from raster import HEIGHT, WIDTH, render_scene

MASTER_SEED = 20260908
N_IMAGES = 24
FIT_COUNT = 16
MAX_SHAPES = 6
MAX_PALETTE = 8
SETTINGS_ID = "shape-family-abc-v1"


class Rng:
    """Numerical Recipes LCG. Deterministic, independent of Python version."""

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
    # Mix a few steps so nearby indices do not share the first draw.
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


def generate_scene(image_index, seed=MASTER_SEED):
    if image_index < 0 or image_index >= N_IMAGES:
        raise ValueError("image_index out of range")
    rng = image_rng(image_index, seed)
    n_colors = rng.randint(2, MAX_PALETTE)
    n_shapes = rng.randint(1, MAX_SHAPES)
    palette = _unique_palette(rng, n_colors)
    shapes = []
    for _ in range(n_shapes):
        typ = rng.randint(0, 2)
        color_index = rng.randint(0, n_colors - 1)
        if typ == 2:
            verts = (
                (rng.randint(0, WIDTH - 1), rng.randint(0, HEIGHT - 1)),
                (rng.randint(0, WIDTH - 1), rng.randint(0, HEIGHT - 1)),
                (rng.randint(0, WIDTH - 1), rng.randint(0, HEIGHT - 1)),
            )
            shapes.append((typ, color_index, verts))
        else:
            x0 = rng.randint(0, WIDTH - 1)
            y0 = rng.randint(0, HEIGHT - 1)
            x1 = rng.randint(0, WIDTH - 1)
            y1 = rng.randint(0, HEIGHT - 1)
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            shapes.append((typ, color_index, (x0, y0, x1, y1)))
    return {
        "settings_id": SETTINGS_ID,
        "seed": seed,
        "image_index": image_index,
        "width": WIDTH,
        "height": HEIGHT,
        "max_shapes": MAX_SHAPES,
        "max_palette": MAX_PALETTE,
        "palette": palette,
        "shapes": shapes,
    }


def render(scene):
    return render_scene(scene)


def split_name(image_index):
    return "fit" if image_index < FIT_COUNT else "heldout"
