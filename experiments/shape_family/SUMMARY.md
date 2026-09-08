# Frozen shape-family A/B/C

Seed 20260908. Images 0-15 fit, 16-23 held-out. 64x64 RGB, filled shapes, at most 6 shapes, palette at most 8, hard edges, integer coordinates. No antialiasing. Not tuned on held-out. No images dropped.

Regenerate: `python3 experiments/shape_family/run_experiment.py`

## Held-out totals (8 images)

| codec | payload bytes | exact | mismatch pixels | encode s | decode s |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 624 | 8/8 | 0 | 0.000186 | 0.008828 |
| B | 1008 | 8/8 | 0 | 0.242396 | 0.005594 |
| C | 1008 | 8/8 | 0 | 0.234318 | 0.005126 |
| PNG | 2294 | 8/8 | 0 | 0.102273 | 0.014813 |
| JPEG_XL | 786 | 8/8 | 0 | 0.106616 | 0.077975 |

## Fit totals (16 images)

| codec | payload bytes | exact | mismatch pixels | encode s | decode s |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 1292 | 16/16 | 0 | 0.000372 | 0.017428 |
| B | 2653 | 16/16 | 0 | 0.463812 | 0.012798 |
| C | 2653 | 16/16 | 0 | 0.468600 | 0.012291 |
| PNG | 5727 | 16/16 | 0 | 0.208305 | 0.030103 |
| JPEG_XL | 1996 | 16/16 | 0 | 0.224278 | 0.159222 |

## Advance bar

- Bar 1 (A exact on every held-out image AND A held-out payload smaller than JPEG XL lossless): pass
- Bar 2 (B plus correction beats JPEG XL lossless on the held-out 8): fail
- A held-out exact: True
- A held-out bytes 624; JPEG XL held-out bytes 786
- B held-out bytes 1008 (description 277, correction 731); JPEG XL held-out bytes 786
- Held-out XOR correction bytes are smaller than JPEG XL lossless.
- C held-out description byte-identical: 8/8

## Payload contents

- A: seed, image index, settings id, max shapes, max palette, width, height, palette RGB, shape type, geometry, palette index. No XOR correction. No LLM prompt.
- B: width, height, palette RGB, background index, region type, geometry bounds, palette index, XOR correction (method byte, length, zlib blob). No generator seed. Encoder sees pixels only.
- C: same stored description as a second B encode after one decode. Byte-identical means the second payload equals the first. One cycle only.
- PNG: full lossless PNG file. JPEG XL: full lossless `.jxl` file from `cjxl -d 0 -e 7`.
- Shared decoder size: 24174 bytes from experiments/shape_family/raster.py, experiments/shape_family/codec_a.py, experiments/shape_family/codec_b.py, experiments/shape_family/codec_c.py. Not added to per-image payloads. generator.py is 3247 bytes and is not required to decode stored payloads.

## Versions and commands

- Python: 3.13.5
- zlib: 1.3.1
- cjxl: cjxl v0.11.2 [AVX2,SSE4,SSE2]
Copyright (c) the JPEG XL Project
- djxl: djxl v0.11.2 [AVX2,SSE4,SSE2]
Copyright (c) the JPEG XL Project
- JPEG XL install failure class: None
- PNG command: stdlib PNG writer in baselines.py (8-bit RGB, filter 0-4, zlib level 9)
- JPEG XL encode: `cjxl INPUT.ppm OUTPUT.jxl -d 0 -e 7 --quiet`
- JPEG XL decode: `djxl INPUT.jxl OUTPUT.ppm --quiet`
- Wall clock: 3.191661 s
- Started: 2026-09-08 15:23:04 EDT
- Finished: 2026-09-08 15:23:08 EDT
- Output: experiments/shape_family/results.json and experiments/shape_family/SUMMARY.md
- Machine-readable results: experiments/shape_family/results.json

No runner failures.
