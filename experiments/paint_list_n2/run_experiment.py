#!/usr/bin/env python3
"""Regenerate Family N2 fixtures and programs from the frozen seed.

Does not change fitter constants. Does not drop images. Does not write a residual.
The checker does not use this script and does not read results.json.

    python3 experiments/paint_list_n2/run_experiment.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import codec
import generator_n2 as gen
from raster import HEIGHT, WIDTH

CONSTANTS = {
    "ELLIPSE_MARGIN": 4,
    "TRI_LO": -2,
    "TRI_HI": 6,
    "LINE_RADIUS": 8,
}


def main():
    if codec.ELLIPSE_MARGIN != CONSTANTS["ELLIPSE_MARGIN"]:
        raise SystemExit("ELLIPSE_MARGIN changed")
    if codec.TRI_LO != CONSTANTS["TRI_LO"]:
        raise SystemExit("TRI_LO changed")
    if codec.TRI_HI != CONSTANTS["TRI_HI"]:
        raise SystemExit("TRI_HI changed")
    if codec.LINE_RADIUS != CONSTANTS["LINE_RADIUS"]:
        raise SystemExit("LINE_RADIUS changed")
    if gen.MASTER_SEED != 20260909:
        raise SystemExit("seed changed")
    os.makedirs(os.path.join(HERE, "fixtures"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "programs"), exist_ok=True)
    held_reuse = 0
    for index in range(gen.N_IMAGES):
        scene = gen.generate_scene(index, gen.MASTER_SEED)
        px = bytes(gen.render(scene))
        fpath = os.path.join(HERE, "fixtures", "img_%02d.rgb" % index)
        with open(fpath, "wb") as f:
            f.write(px)
        if gen.split_name(index) == "heldout" and gen.reuses_color(scene):
            held_reuse += 1
        if len(scene["shapes"]) > gen.MAX_SHAPES or len(scene["palette"]) > gen.MAX_PALETTE:
            raise SystemExit("family limit")
        if not gen.masks_disjoint(scene):
            raise SystemExit("overlap")
    if held_reuse < 3:
        raise SystemExit("held-out reuse below 3 before size measurement")
    print("generated", gen.N_IMAGES, "seed", gen.MASTER_SEED, "heldout_reuse", held_reuse)
    print("frozen", CONSTANTS)
    for index in range(gen.N_IMAGES):
        fpath = os.path.join(HERE, "fixtures", "img_%02d.rgb" % index)
        px = open(fpath, "rb").read()
        payload = codec.encode(px, WIDTH, HEIGHT)
        ppath = os.path.join(HERE, "programs", "img_%02d.pln" % index)
        if payload is None or codec.decode(payload) != px:
            if os.path.exists(ppath):
                os.remove(ppath)
            print("image", index, "exact", 0, "program", "missing")
            continue
        with open(ppath, "wb") as f:
            f.write(payload)
        print("image", index, "exact", 1, "program", len(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
