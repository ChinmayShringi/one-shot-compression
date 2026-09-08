"""Codec A: store generator args and redraw. No XOR correction.

Payload includes seed, image index, settings, palette, and shape outlines
(type, geometry, palette index). Decode calls the same rasterizer as the
generator. A mismatch means A failed for that image.
"""

from __future__ import annotations

import struct

from raster import TYPE_ELLIPSE, TYPE_RECT, TYPE_TRIANGLE, render_scene

MAGIC = b"SFA1"
VERSION = 1


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
    return bytes([typ, color_index, len(coords)]) + bytes(coords)


def encode(scene):
    """Serialize generator args. Encoder is given the scene, not an LLM prompt."""
    palette = scene["palette"]
    shapes = scene["shapes"]
    if not palette or len(palette) > 255:
        raise ValueError("palette size")
    settings_id = scene["settings_id"].encode("ascii")
    if len(settings_id) > 255:
        raise ValueError("settings id too long")
    out = bytearray()
    out += MAGIC
    out.append(VERSION)
    out += struct.pack(
        "<IHHHBB",
        int(scene["seed"]) & 0xFFFFFFFF,
        int(scene["image_index"]),
        int(scene["width"]),
        int(scene["height"]),
        len(palette),
        len(shapes),
    )
    out.append(len(settings_id))
    out += settings_id
    out.append(int(scene["max_shapes"]) & 0xFF)
    out.append(int(scene["max_palette"]) & 0xFF)
    for r, g, b in palette:
        out += bytes((r & 255, g & 255, b & 255))
    for typ, color_index, geom in shapes:
        out += _pack_shape(typ, color_index, geom)
    return bytes(out)


def decode(payload):
    if payload[:4] != MAGIC:
        raise ValueError("bad A magic")
    if payload[4] != VERSION:
        raise ValueError("bad A version")
    off = 5
    seed, image_index, width, height, n_colors, n_shapes = struct.unpack_from(
        "<IHHHBB", payload, off
    )
    off += struct.calcsize("<IHHHBB")
    sid_len = payload[off]
    off += 1
    settings_id = payload[off : off + sid_len].decode("ascii")
    off += sid_len
    max_shapes = payload[off]
    max_palette = payload[off + 1]
    off += 2
    palette = []
    for _ in range(n_colors):
        palette.append((payload[off], payload[off + 1], payload[off + 2]))
        off += 3
    shapes = []
    for _ in range(n_shapes):
        typ = payload[off]
        color_index = payload[off + 1]
        ncoord = payload[off + 2]
        off += 3
        coords = list(payload[off : off + ncoord])
        off += ncoord
        if typ in (TYPE_RECT, TYPE_ELLIPSE):
            if ncoord != 4:
                raise ValueError("rect/ellipse coords")
            geom = (coords[0], coords[1], coords[2], coords[3])
        elif typ == TYPE_TRIANGLE:
            if ncoord != 6:
                raise ValueError("triangle coords")
            geom = (
                (coords[0], coords[1]),
                (coords[2], coords[3]),
                (coords[4], coords[5]),
            )
        else:
            raise ValueError("bad shape type in A payload")
        shapes.append((typ, color_index, geom))
    if off != len(payload):
        raise ValueError("trailing bytes in A payload")
    scene = {
        "settings_id": settings_id,
        "seed": seed,
        "image_index": image_index,
        "width": width,
        "height": height,
        "max_shapes": max_shapes,
        "max_palette": max_palette,
        "palette": palette,
        "shapes": shapes,
    }
    return render_scene(scene)
