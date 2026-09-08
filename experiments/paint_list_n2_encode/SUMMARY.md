# N2 pure encode from committed rasters

Reads the committed Family N2 raw RGB fixtures only. Does not regenerate them.
Does not treat them as a new family. Does not edit experiments/paint_list_n2.
Does not retune fitter constants. No residual, no correction bytes, no photos, no LLM.

Fixture commit: b59041517f5756bb0950f9d1441760b72249fc52
Paths: experiments/paint_list_n2/fixtures/img_00.rgb through img_23.rgb
Each file is 64*64*3 raw RGB bytes. Held-out is images 16-23.

`encode(px, width, height)` accepts only the raster bytes and dimensions. It does not import generator_n2, does not read a seed, and does not receive generator arguments. Decode redraws the stored program.

## Import graph

Source-text scan of encode.py and its local imports (raster.py). The scan does not import a generator.

- import_graph: clean
- modules: experiments/paint_list_n2_encode/encode.py experiments/paint_list_n2_encode/raster.py
- result: encode.py does not import generator_n2 or any generator

This is not recipe replay. The encoder recovered a program from the committed raster bytes alone.

## Held-out totals (8 images)

| codec | payload bytes | exact |
| --- | ---: | ---: |
| program | 374 | 8/8 |
| PNG | 2278 | 8/8 |
| JPEG XL effort 9 | 743 | 8/8 |

Held-out program payload is smaller than JPEG XL effort 9: yes.

## Fit totals (16 images)

| codec | payload bytes | exact |
| --- | ---: | ---: |
| program | 818 | 16/16 |
| PNG | 5443 | 16/16 |
| JPEG XL effort 9 | 1966 | 16/16 |

## Advance bar

1. Import graph clean: yes
2. Held-out 8 exact (redraw hash equals committed fixture hash): yes, 8/8
3. Held-out program payload smaller than JPEG XL effort 9: yes, 374 < 743

Advance: pass
Representation stops: no

## Payload contents

- Program: magic/version, settings code 1 (paint-list-n2-v1), width, height, palette RGB, shape type, geometry, color index. No seed. No XOR. No residual. No correction bytes.
- PNG: full lossless PNG file. JPEG XL: full lossless `.jxl` file from `cjxl -d 0 -e 9`.
- Frozen constants copied from experiments/paint_list_n2/FROZEN_CONSTANTS.txt and not changed: ELLIPSE_MARGIN=4, TRI_LO=-2, TRI_HI=6, LINE_RADIUS=8.

## Versions and commands

- Python: 3.13.5
- zlib: 1.3.1
- cjxl: cjxl v0.11.2 [AVX2,SSE4,SSE2]
- djxl: djxl v0.11.2 [AVX2,SSE4,SSE2]
- JPEG XL effort: 9
- Checker: `python3 experiments/paint_list_n2_encode/check_encode.py`
- Finished: 2026-09-08 16:38:36 EDT
