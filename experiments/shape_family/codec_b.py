"""Codec B: pixels only. Recover palette and region bounds, redraw, XOR-correct.

The encoder never receives generator args or the seed. It extracts the palette
from unique colors, labels 4-connected components, stores each component as a
filled primitive inferred from that component (rect, ellipse, or triangle) or
the component bounding box if no primitive matches exactly, redraws, then
stores an XOR correction of remaining mismatches. Correction bytes are part of
the payload.
"""

from __future__ import annotations

import struct
import zlib

from raster import (
    TYPE_ELLIPSE,
    TYPE_RECT,
    TYPE_TRIANGLE,
    new_image,
    pixel_in_ellipse,
    pixel_in_triangle,
    render_scene,
)

MAGIC = b"SFB1"
VERSION = 1
CORR_NONE = 0
CORR_SPARSE = 1
CORR_PLANE = 2


def _unique_palette(px, width, height):
    palette = []
    index_of = {}
    indices = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            i = y * width + x
            o = i * 3
            color = (px[o], px[o + 1], px[o + 2])
            idx = index_of.get(color)
            if idx is None:
                idx = len(palette)
                index_of[color] = idx
                palette.append(color)
            indices[i] = idx
    return palette, indices, index_of


def _modal_bg(indices):
    counts = {}
    for v in indices:
        counts[v] = counts.get(v, 0) + 1
    # Lowest palette index wins ties.
    return min(counts, key=lambda k: (-counts[k], k))


def _components(indices, width, height, color):
    seen = bytearray(width * height)
    comps = []
    for y in range(height):
        for x in range(width):
            start = y * width + x
            if seen[start] or indices[start] != color:
                continue
            stack = [(x, y)]
            seen[start] = 1
            pixels = []
            minx = maxx = x
            miny = maxy = y
            while stack:
                cx, cy = stack.pop()
                pixels.append((cx, cy))
                if cx < minx:
                    minx = cx
                elif cx > maxx:
                    maxx = cx
                if cy < miny:
                    miny = cy
                elif cy > maxy:
                    maxy = cy
                if cx > 0:
                    n = cy * width + (cx - 1)
                    if not seen[n] and indices[n] == color:
                        seen[n] = 1
                        stack.append((cx - 1, cy))
                if cx + 1 < width:
                    n = cy * width + (cx + 1)
                    if not seen[n] and indices[n] == color:
                        seen[n] = 1
                        stack.append((cx + 1, cy))
                if cy > 0:
                    n = (cy - 1) * width + cx
                    if not seen[n] and indices[n] == color:
                        seen[n] = 1
                        stack.append((cx, cy - 1))
                if cy + 1 < height:
                    n = (cy + 1) * width + cx
                    if not seen[n] and indices[n] == color:
                        seen[n] = 1
                        stack.append((cx, cy + 1))
            comps.append(
                {
                    "pixels": pixels,
                    "bbox": (minx, miny, maxx, maxy),
                    "color": color,
                }
            )
    return comps


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


def _mask_set(pixels):
    return set(pixels)


def _rect_mask(x0, y0, x1, y1):
    out = set()
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            out.add((x, y))
    return out


def _ellipse_mask(x0, y0, x1, y1):
    out = set()
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if pixel_in_ellipse(x, y, x0, y0, x1, y1):
                out.add((x, y))
    return out


def _triangle_mask(v0, v1, v2):
    xs = (v0[0], v1[0], v2[0])
    ys = (v0[1], v1[1], v2[1])
    out = set()
    for y in range(min(ys), max(ys) + 1):
        for x in range(min(xs), max(xs) + 1):
            if pixel_in_triangle(x, y, v0, v1, v2):
                out.add((x, y))
    return out


def _area2(a, b, c):
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _triangle_candidates(hull):
    if len(hull) < 3:
        return []
    # Axis extremes are always candidates; sharp turns cover the corners.
    extremes = []
    extremes.append(min(hull, key=lambda p: (p[0], p[1])))
    extremes.append(max(hull, key=lambda p: (p[0], -p[1])))
    extremes.append(min(hull, key=lambda p: (p[1], p[0])))
    extremes.append(max(hull, key=lambda p: (p[1], -p[0])))
    n = len(hull)
    turns = []
    for i in range(n):
        a = hull[(i - 1) % n]
        b = hull[i]
        c = hull[(i + 1) % n]
        turns.append((abs(_area2(a, b, c)), b))
    turns.sort(key=lambda t: (-t[0], t[1]))
    chosen = []
    seen = set()
    for p in extremes:
        if p not in seen:
            seen.add(p)
            chosen.append(p)
    for _score, p in turns:
        if p in seen:
            continue
        seen.add(p)
        chosen.append(p)
        if len(chosen) >= 12:
            break
    if len(hull) <= 16:
        for p in hull:
            if p not in seen:
                chosen.append(p)
    cands = []
    m = len(chosen)
    for i in range(m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                a, b, c = chosen[i], chosen[j], chosen[k]
                if _area2(a, b, c) == 0:
                    continue
                cands.append(tuple(sorted((a, b, c))))
    # Unique, highest area first, then lex for determinism.
    uniq = sorted(set(cands), key=lambda t: (-_area2(t[0], t[1], t[2]), t))
    return uniq[:64]


def infer_region(comp):
    pixels = comp["pixels"]
    x0, y0, x1, y1 = comp["bbox"]
    color = comp["color"]
    mask = _mask_set(pixels)
    if mask == _rect_mask(x0, y0, x1, y1):
        return (TYPE_RECT, color, (x0, y0, x1, y1))
    if mask == _ellipse_mask(x0, y0, x1, y1):
        return (TYPE_ELLIPSE, color, (x0, y0, x1, y1))
    hull = _convex_hull(pixels)
    for verts in _triangle_candidates(hull):
        if _triangle_mask(verts[0], verts[1], verts[2]) == mask:
            return (TYPE_TRIANGLE, color, verts)
    # Bounds fallback: store the component bounding box as a filled rect.
    return (TYPE_RECT, color, (x0, y0, x1, y1))


def recover_regions(px, width, height):
    palette, indices, _index_of = _unique_palette(px, width, height)
    bg = _modal_bg(indices)
    regions = []
    for color in range(len(palette)):
        if color == bg:
            continue
        for comp in _components(indices, width, height, color):
            regions.append(infer_region(comp))
    regions.sort(
        key=lambda r: (
            -_bbox_area(r),
            _anchor(r)[1],
            _anchor(r)[0],
            r[1],
            r[0],
            _geom_key(r),
        )
    )
    return palette, bg, regions


def _bbox_area(region):
    typ, _color, geom = region
    if typ in (TYPE_RECT, TYPE_ELLIPSE):
        return (geom[2] - geom[0] + 1) * (geom[3] - geom[1] + 1)
    xs = (geom[0][0], geom[1][0], geom[2][0])
    ys = (geom[0][1], geom[1][1], geom[2][1])
    return (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)


def _anchor(region):
    typ, _color, geom = region
    if typ in (TYPE_RECT, TYPE_ELLIPSE):
        return (geom[0], geom[1])
    xs = (geom[0][0], geom[1][0], geom[2][0])
    ys = (geom[0][1], geom[1][1], geom[2][1])
    return (min(xs), min(ys))


def _geom_key(region):
    typ, _color, geom = region
    if typ in (TYPE_RECT, TYPE_ELLIPSE):
        return geom
    return tuple(v for vert in geom for v in vert)


def redraw(width, height, palette, bg, regions):
    scene = {
        "width": width,
        "height": height,
        "palette": palette,
        "shapes": [(TYPE_RECT, bg, (0, 0, width - 1, height - 1))] + list(regions),
    }
    # Background index is painted first as a full-canvas rect, then recovered regions.
    # render_scene always paints palette[0] first, so rebuild palette with bg at 0
    # only for drawing, then map colors explicitly here instead.
    px = new_image(width, height, palette[bg])
    scene = {
        "width": width,
        "height": height,
        "palette": palette,
        "shapes": regions,
    }
    # render_scene fills palette[0] before shapes. Paint bg ourselves, then shapes
    # with a dummy palette[0] match by calling fills via a scene whose palette[0]
    # is already the bg only if we skip that fill. Call render only when bg==0
    # and no extra rect. Always paint manually by swapping?
    _ = scene
    for typ, color_index, geom in regions:
        color = palette[color_index]
        if typ == TYPE_RECT:
            from raster import fill_rect

            fill_rect(px, width, height, geom[0], geom[1], geom[2], geom[3], color)
        elif typ == TYPE_ELLIPSE:
            from raster import fill_ellipse

            fill_ellipse(px, width, height, geom[0], geom[1], geom[2], geom[3], color)
        elif typ == TYPE_TRIANGLE:
            from raster import fill_triangle

            fill_triangle(px, width, height, geom[0], geom[1], geom[2], color)
        else:
            raise ValueError("bad region type")
    return px


def _pack_region(region):
    typ, color, geom = region
    if typ in (TYPE_RECT, TYPE_ELLIPSE):
        coords = (geom[0], geom[1], geom[2], geom[3])
    else:
        coords = (
            geom[0][0],
            geom[0][1],
            geom[1][0],
            geom[1][1],
            geom[2][0],
            geom[2][1],
        )
    return bytes([typ, color, len(coords)]) + bytes(coords)


def _correction(pred, orig, palette_index, width, height):
    n_pix = width * height
    sparse = bytearray()
    n = 0
    plane = bytearray(n_pix)
    for i in range(n_pix):
        o = i * 3
        color = (orig[o], orig[o + 1], orig[o + 2])
        plane[i] = palette_index[color]
        if pred[o] != orig[o] or pred[o + 1] != orig[o + 1] or pred[o + 2] != orig[o + 2]:
            n += 1
            sparse += struct.pack("<HB", i, palette_index[color])
    if n == 0:
        return CORR_NONE, b"", n
    sparse_blob = struct.pack("<I", n) + bytes(sparse)
    z_sparse = zlib.compress(sparse_blob, 9)
    z_plane = zlib.compress(bytes(plane), 9)
    if len(z_sparse) <= len(z_plane):
        return CORR_SPARSE, z_sparse, n
    return CORR_PLANE, z_plane, n


def _apply_correction(px, width, height, method, blob, palette):
    if method == CORR_NONE:
        return
    raw = zlib.decompress(blob)
    if method == CORR_SPARSE:
        n = struct.unpack_from("<I", raw, 0)[0]
        off = 4
        for _ in range(n):
            i, pal = struct.unpack_from("<HB", raw, off)
            off += 3
            color = palette[pal]
            o = i * 3
            px[o] = color[0]
            px[o + 1] = color[1]
            px[o + 2] = color[2]
        if off != len(raw):
            raise ValueError("bad sparse correction")
        return
    if method == CORR_PLANE:
        if len(raw) != width * height:
            raise ValueError("bad plane correction")
        for i, pal in enumerate(raw):
            color = palette[pal]
            o = i * 3
            px[o] = color[0]
            px[o + 1] = color[1]
            px[o + 2] = color[2]
        return
    raise ValueError("bad correction method")


def encode(px, width, height):
    palette, bg, regions = recover_regions(px, width, height)
    pred = redraw(width, height, palette, bg, regions)
    index_of = {c: i for i, c in enumerate(palette)}
    method, blob, n_mis = _correction(pred, px, index_of, width, height)
    out = bytearray()
    out += MAGIC
    out.append(VERSION)
    out += struct.pack("<HHBBH", width, height, len(palette), bg, len(regions))
    for r, g, b in palette:
        out += bytes((r, g, b))
    for region in regions:
        out += _pack_region(region)
    # Correction section is always present so its cost is counted.
    out.append(method)
    out += struct.pack("<I", len(blob))
    out += blob
    return bytes(out), {
        "n_regions": len(regions),
        "n_palette": len(palette),
        "bg_index": bg,
        "mismatch_pixels_before_correction": n_mis,
        "correction_method": method,
        "description_bytes": None,  # filled by caller from split
    }


def encode_with_breakdown(px, width, height):
    payload, info = encode(px, width, height)
    # Recompute description length from a parse rather than guessing.
    desc_len, corr_len = split_lengths(payload)
    info["description_bytes"] = desc_len
    info["correction_bytes"] = corr_len
    info["payload_bytes"] = len(payload)
    return payload, info


def split_lengths(payload):
    _magic, _ver, width, height, n_colors, bg, n_regions, off, palette = _parse_header(
        payload
    )
    del width, height, bg, palette
    for _ in range(n_regions):
        ncoord = payload[off + 2]
        off += 3 + ncoord
    desc_len = off
    corr_len = len(payload) - off
    return desc_len, corr_len


def _parse_header(payload):
    if payload[:4] != MAGIC or payload[4] != VERSION:
        raise ValueError("bad B payload")
    width, height, n_colors, bg, n_regions = struct.unpack_from("<HHBBH", payload, 5)
    off = 5 + struct.calcsize("<HHBBH")
    palette = []
    for _ in range(n_colors):
        palette.append((payload[off], payload[off + 1], payload[off + 2]))
        off += 3
    return MAGIC, VERSION, width, height, n_colors, bg, n_regions, off, palette


def decode(payload):
    _magic, _ver, width, height, n_colors, bg, n_regions, off, palette = _parse_header(
        payload
    )
    del n_colors
    regions = []
    for _ in range(n_regions):
        typ = payload[off]
        color = payload[off + 1]
        ncoord = payload[off + 2]
        off += 3
        coords = list(payload[off : off + ncoord])
        off += ncoord
        if typ in (TYPE_RECT, TYPE_ELLIPSE):
            geom = (coords[0], coords[1], coords[2], coords[3])
        elif typ == TYPE_TRIANGLE:
            geom = (
                (coords[0], coords[1]),
                (coords[2], coords[3]),
                (coords[4], coords[5]),
            )
        else:
            raise ValueError("bad region type")
        regions.append((typ, color, geom))
    method = payload[off]
    off += 1
    blen = struct.unpack_from("<I", payload, off)[0]
    off += 4
    blob = payload[off : off + blen]
    off += blen
    if off != len(payload):
        raise ValueError("trailing bytes in B payload")
    px = redraw(width, height, palette, bg, regions)
    _apply_correction(px, width, height, method, blob, palette)
    return px
