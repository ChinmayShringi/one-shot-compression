# Paint-list recovery, no correction

Family N seed 20260908. Family O seed 20260908, same generator behavior as experiments/shape_family/generator.py at d287e17. Images 0-15 fit, 16-23 held-out. 64x64 RGB, at most 6 filled shapes, palette at most 8, hard edges, integer coordinates. No antialiasing. Encoder sees pixels only. Stored program is palette plus ordered shapes. No XOR, residual, or correction bytes. Not tuned on held-out. No images dropped.

Regenerate: `python3 experiments/paint_list/run_experiment.py`

Family N uses seed 20260908. Rejection sampling consumes the image RNG until each shape is disjoint or the attempt budget ends. The seed was not changed.

## Family N held-out totals (8 images)

| codec | payload bytes | exact | missing payloads |
| --- | ---: | ---: | ---: |
| program | 364 | 8/8 | 0 |
| PNG | 2614 | 8/8 | 0 |
| JPEG_XL | 938 | 8/8 | 0 |

## Family N fit totals (16 images)

| codec | payload bytes | exact | missing payloads |
| --- | ---: | ---: | ---: |
| program | 804 | 16/16 | 0 |
| PNG | 5777 | 16/16 | 0 |
| JPEG_XL | 2134 | 16/16 | 0 |

## Family O held-out totals (8 images)

| codec | payload bytes | exact | missing payloads |
| --- | ---: | ---: | ---: |
| program | 39 | 2/8 | 6 |
| PNG | 2294 | 8/8 | 0 |
| JPEG_XL | 791 | 8/8 | 0 |

## Family O fit totals (16 images)

| codec | payload bytes | exact | missing payloads |
| --- | ---: | ---: | ---: |
| program | 168 | 6/16 | 10 |
| PNG | 5727 | 16/16 | 0 |
| JPEG_XL | 1933 | 16/16 | 0 |

## Advance bar

- Advance only if Family N is exact on every held-out image AND Family N held-out program payload is smaller than JPEG XL effort 9 on those same 8: pass
- Family N held-out exact: 8/8
- Family N held-out program bytes 364; PNG 2614; JPEG XL effort 9 938
- Family O held-out exact: 2/8
- Family O held-out program bytes 39; PNG 2294; JPEG XL effort 9 791
- Finding: A raster does not determine paint order.

## Payload contents

- Program: magic/version, settings code 1 (paint-list-nocorr-v1), width, height, palette RGB, shape type, geometry, color index, recovered order. No seed (encoder sees pixels only). No XOR. No residual. No correction bytes.
- Failed images store no program. Their payload is missing, not a residual.
- PNG: full lossless PNG file. JPEG XL: full lossless `.jxl` file from `cjxl -d 0 -e 9`.
- Shared decoder size: 18833 bytes from experiments/paint_list/raster.py, experiments/paint_list/codec.py. Not added to per-image payloads.

## Versions and commands

- Python: 3.13.5
- zlib: 1.3.1
- cjxl: cjxl v0.11.2 [AVX2,SSE4,SSE2]
Copyright (c) the JPEG XL Project
- djxl: djxl v0.11.2 [AVX2,SSE4,SSE2]
Copyright (c) the JPEG XL Project
- JPEG XL effort: 9
- JPEG XL install failure class: None
- PNG command: stdlib PNG writer in baselines.py (8-bit RGB, filter 0-4, zlib level 9)
- JPEG XL encode: `cjxl INPUT.ppm OUTPUT.jxl -d 0 -e 9 --quiet`
- JPEG XL decode: `djxl INPUT.jxl OUTPUT.ppm --quiet`
- Checker: `python3 experiments/paint_list/check_published.py`
- Wall clock: 46.004661 s
- Started: 2026-09-08 16:05:59 EDT
- Finished: 2026-09-08 16:06:45 EDT

No runner exceptions. Exactness failures are recorded as not exact.
