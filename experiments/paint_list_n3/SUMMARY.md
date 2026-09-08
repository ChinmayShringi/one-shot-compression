# Family N3 paint-list, shared edges

Same canvas as N2: 64x64 RGB, at most 6 filled shapes, palette size at most 8, no overlap, no antialiasing, hard edges, integer coordinates. No residual. No correction. No photos. No overlap rescue.

Seed 20260910. Not changed. Images 0-15 fit, 16-23 held-out. Seed and split frozen before payload size measurement.

Frozen fitter constants copied from experiments/paint_list_n2/FROZEN_CONSTANTS.txt before held-out generation. Not retuned.

ELLIPSE_MARGIN=4
TRI_LO=-2
TRI_HI=6
LINE_RADIUS=8

Held-out images 16, 18, 20, and 22 each contain two distinct same-color rectangles that share a pixel edge and remain two shapes in the true program. Corner-only contact does not count. A naive 4-connected color mask merges each pair. The encoder splits those merged masks from raster bytes only.

Sidecar split: experiments/paint_list_n3/sidecar/shared_edges.json is written by the generator at fixture-write time. encode.py does not import the generator and does not open that file. The checker reads it only to report the held-out shared-edge count.

Import graph: clean. encode.py does not import generator_n3 or any generator. Recovery imports raster.py only.

Held-out exact: 8/8. Redraw sha256 matches committed fixture sha256.

Held-out program payload 410 bytes. PNG 2732. JPEG XL effort 9: 929. Program is smaller than JPEG XL effort 9.

Held-out shared-edge images: 4.

Advance: pass. Representation stops: no.

Checker: python3 experiments/paint_list_n3/check_n3.py
