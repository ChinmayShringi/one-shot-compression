# Family N2 paint-list, repeated colors

Family N2 seed 20260909. The seed was not changed. Images 0-15 fit, 16-23 held-out. 24 images, 64x64 RGB, at most 6 filled shapes, palette size at most 8, no overlap, no antialiasing, hard edges, integer coordinates. Colors may repeat across shapes in the same image. Encoder sees pixels only. Stored program is palette plus ordered shapes. No XOR, residual, or correction bytes. Not tuned on held-out. No images dropped. No unique-color rerun.

Frozen fitter constants copied from experiments/paint_list/codec.py at e1af093fe560908b2a9a8c9939de4436462ea57e before held-out images were generated. Not swept on held-out seeds.

- ELLIPSE_MARGIN = 4
- TRI_LO = -2
- TRI_HI = 6
- LINE_RADIUS = 8

Checker: `python3 experiments/paint_list_n2/check_n2.py`

The checker hashes committed fixture bytes, decodes committed programs only, and does not re-encode to prove exactness. It does not read results.json. It does not hardcode a payload byte table.

## Held-out totals (8 images)

| codec | payload bytes | exact |
| --- | ---: | ---: |
| program | 374 | 8/8 |
| PNG | 2278 | 8/8 |
| JPEG XL effort 9 | 743 | 8/8 |

Held-out images that reuse a color (at least two shapes share one palette color): 8.

## Fit totals (16 images)

| codec | payload bytes | exact |
| --- | ---: | ---: |
| program | 818 | 16/16 |
| PNG | 5443 | 16/16 |
| JPEG XL effort 9 | 1966 | 16/16 |

Fit images that reuse a color: 6.

## Advance bar

- Held-out exact: 8/8
- Held-out program bytes 374; PNG 2278; JPEG XL effort 9 743
- Held-out program payload smaller than JPEG XL effort 9: yes
- Checker redraw hashes match committed image hashes: yes
- Advance: pass

## Payload contents

- Program: magic/version, settings code 1 (paint-list-n2-v1), width, height, palette RGB, shape type, geometry, color index, recovered order. No seed. No XOR. No residual. No correction bytes.
- Failed images store no program. Their payload is missing, not a residual. This run stored a program for every image.
- PNG: full lossless PNG file. JPEG XL: full lossless `.jxl` file from `cjxl -d 0 -e 9`.
- Shared decoder size: 21989 bytes from experiments/paint_list_n2/raster.py and experiments/paint_list_n2/codec.py. Not added to per-image payloads.

## Versions and commands

- Python: 3.13.5
- zlib: 1.3.1
- cjxl: cjxl v0.11.2 [AVX2,SSE4,SSE2]
- djxl: djxl v0.11.2 [AVX2,SSE4,SSE2]
- JPEG XL effort: 9
- JPEG XL install failure class: None
- PNG command: stdlib PNG writer in baselines.py (8-bit RGB, filter 0-4, zlib level 9)
- JPEG XL encode: `cjxl INPUT.ppm OUTPUT.jxl -d 0 -e 9 --quiet`
- JPEG XL decode: `djxl INPUT.jxl OUTPUT.ppm --quiet`
- Checker: `python3 experiments/paint_list_n2/check_n2.py`
- Finished: 2026-09-08 16:26:34 EDT
