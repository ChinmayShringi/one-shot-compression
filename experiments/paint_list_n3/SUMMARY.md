# Family N3 paint-list, shared edges

Same canvas as N2: 64x64 RGB, at most 6 filled shapes, palette size at most 8, no overlap, no antialiasing, hard edges, integer coordinates. No residual. No correction. No photos. No overlap rescue.

Seed 20260910. Not changed. Images 0-15 fit, 16-23 held-out. Seed and split frozen before payload size measurement.

Frozen fitter constants copied from experiments/paint_list_n2/FROZEN_CONSTANTS.txt before held-out generation. Not retuned. Not widened.

ELLIPSE_MARGIN=4
TRI_LO=-2
TRI_HI=6
LINE_RADIUS=8

Shared-edge count is computed from the committed pixel rasters. A shared edge is two distinct same-color shapes whose pixels touch along a 4-connected edge. Corner-only contact does not count. The checker does not open experiments/paint_list_n3/sidecar.

Held-out images 18, 20, and 22 still recover as two filled rectangles that share a pixel edge. Those pairs are counted from pixels. Image 16 was regenerated under seed 20260910 as a same-color rectangle and ellipse that share a 4-edge. encode.py accepts only that raster and returns no program. No residual was added. Constants were not retuned. The fitter was not widened.

The paint list stops at disconnected or two-rectangle masks.

Held-out exact: not 8/8. The non-rectangle shared-edge pair is not counted, because it is not recovered from the raster.

Import graph: clean. encode.py does not import generator_n3 or any generator. Recovery imports raster.py only.

Advance: fail. Representation stops: yes.

Checker: python3 experiments/paint_list_n3/check_n3.py
