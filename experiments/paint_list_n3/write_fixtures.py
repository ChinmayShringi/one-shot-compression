#!/usr/bin/env python3
"""Create Family N3 fixtures. Imports the generator. Does not measure payload sizes.

Frozen constants are already recorded in FROZEN_CONSTANTS.txt before this runs.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import generator_n3

FIXTURE_DIR = os.path.join(HERE, "fixtures")
SIDECAR = os.path.join(HERE, "sidecar", "shared_edges.json")


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SIDECAR), exist_ok=True)
    rows = []
    for index in range(generator_n3.N_IMAGES):
        scene = generator_n3.generate_scene(index)
        if not generator_n3.masks_disjoint(scene):
            raise RuntimeError("overlap in image %d" % index)
        px = generator_n3.render(scene)
        path = os.path.join(FIXTURE_DIR, "img_%02d.rgb" % index)
        with open(path, "wb") as fh:
            fh.write(px)
        pairs = generator_n3.shared_edge_pairs(scene)
        row = {
            "index": index,
            "split": generator_n3.split_name(index),
            "n_shapes": len(scene["shapes"]),
            "n_colors": len(scene["palette"]),
            "shared_edge_required": bool(scene["shared_edge_required"]),
            "has_shared_edge": len(pairs) >= 1,
            "shared_edge_pairs": len(pairs),
            "used_fallback_pair": bool(scene["used_fallback_pair"]),
            "true_program_keeps_pair_as_two_shapes": len(pairs) >= 1,
        }
        rows.append(row)
        print(
            "fixture",
            index,
            row["split"],
            "shapes",
            row["n_shapes"],
            "shared_edge",
            int(row["has_shared_edge"]),
            "fallback",
            int(row["used_fallback_pair"]),
        )
    held = [r for r in rows if r["split"] == "heldout" and r["has_shared_edge"]]
    if len(held) < 3:
        raise RuntimeError("held-out shared-edge count %d < 3" % len(held))
    sidecar = {
        "family": "N3",
        "seed": generator_n3.MASTER_SEED,
        "seed_changed": False,
        "split": "0-15 fit, 16-23 held-out",
        "frozen_before_size_measurement": True,
        "frozen_constants_copied_before_heldout_generation": True,
        "frozen_constants": {
            "ELLIPSE_MARGIN": 4,
            "TRI_LO": -2,
            "TRI_HI": 6,
            "LINE_RADIUS": 8,
        },
        "encoder_must_not_read_this_file": True,
        "note": (
            "Written by the generator at fixture-write time. "
            "encode.py must not open this file. "
            "Corner-only contact is not counted as a shared edge."
        ),
        "heldout_shared_edge_images": len(held),
        "images": rows,
    }
    with open(SIDECAR, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)
        fh.write("\n")
    print("heldout_shared_edge_images", len(held))
    print("sidecar", os.path.relpath(SIDECAR, os.path.join(HERE, "..", "..")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
