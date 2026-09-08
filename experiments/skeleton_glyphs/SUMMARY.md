# Skeleton glyph compression

Advance: PASS
Representation stops: no
Correction paid for the skeleton: no
Seed: 20260908
Measured: 2026-09-08 5:28 PM ET
Runtime: 0.559 s (wall time of one full measurement pass: re-encode 24, decode twice each, cjxl+djxl lossless effort 9)

## Question
On a frozen set of 24 synthetic 32x32 binary glyphs, does a Zhang-Suen skeleton or stroke list plus an exact XOR correction stream beat both packed bits and JPEG XL lossless effort 9? Advance uses held-out glyphs 16-23 only.

## Frozen constants
See FROZEN_CONSTANTS.txt. Radius 1 (Chebyshev dilation) and 3-bit chain codes were picked once before measurement and not searched after held-out. Dimensions are not stored in the payload; the shared decoder knows 32x32, same rule as the packed-bits baseline. No shared stroke library, codebook, LLM, or network at decode.

## Versions
- Python: 3.13.5 (main, Jul 15 2026, 20:25:40) [GCC 14.2.0]
- zlib: 1.3.1
- cjxl:
```
cjxl v0.11.2 [AVX2,SSE4,SSE2]
Copyright (c) the JPEG XL Project
```
- djxl:
```
djxl v0.11.2 [AVX2,SSE4,SSE2]
Copyright (c) the JPEG XL Project
```

## Commands
- Packed bits: each fixture file is the payload, 128 bytes, no header.
- Program: `encode(raster_bytes_128, width=32, height=32)` writes `payloads/glyph_XX.sg`.
- JPEG XL: `cjxl INPUT.pgm -d 0 -e 9 --quiet OUTPUT.jxl` then `djxl OUTPUT.jxl OUTPUT.pgm --quiet`. Count the full `.jxl` file. Pixel bits must match the fixture.
- Check: `python3 check.py` recomputes sizes, exact counts, and JPEG XL from fixtures. It does not read a results JSON.

## Payload contents
Every byte is counted. Per image:
- 8-byte header: magic `SG`, version 1, flags, desc_len uint16 LE, corr_len uint16 LE
- description: smaller of packed skeleton bitmap and stroke list, zlib-9 only if strictly smaller
- correction stream: XOR of dilated reconstruction versus the fixture; smaller of packed mask and coordinate list, zlib-9 only if strictly smaller; omitted if empty
Held-out header bytes are 8*8 = 64, included in the 420-byte total (64 + desc 154 + corr 202 = 420).

Shared decoder Python is not in the per-image payload. Shared decoder byte size = decode.py + raster.py = 5899 + 5471 = 11370.

## Stop rule
correction bytes >= bytes saved versus packed bits, where bytes saved = packed - (payload - correction). On held-out that is 202 >= 806, which is false, so the correction stream does not pay for the skeleton. representation_stops: no. No retuning was done after seeing held-out.

## Tables

| glyph | split | payload_bytes | packed_bits | jxl_bytes | exact | desc_bytes | corr_bytes | corr_pays | repr | fixture_sha256 | redraw_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00 | fit | 46 | 128 | 38 | 1 | 14 | 24 | 0 | stroke | 666ad8d68afec965979f558d49eeff103459868d4c64510701044aac42bf611e | 666ad8d68afec965979f558d49eeff103459868d4c64510701044aac42bf611e |
| 01 | fit | 34 | 128 | 71 | 1 | 26 | 0 | 0 | stroke | 3ab55ff96fd48cefd9ea6eb81ae2dbb6320f42ddd300283e5fee90d1469fe13a | 3ab55ff96fd48cefd9ea6eb81ae2dbb6320f42ddd300283e5fee90d1469fe13a |
| 02 | fit | 45 | 128 | 43 | 1 | 15 | 22 | 0 | stroke | d93cdc93e48a9a64ed2b63a4039822fac45473fe99d00df78edf8b81ab242a0e | d93cdc93e48a9a64ed2b63a4039822fac45473fe99d00df78edf8b81ab242a0e |
| 03 | fit | 65 | 128 | 47 | 1 | 23 | 34 | 0 | skel | d5ee38cdd9c6bc342b1f2355f5f2bd8bd2fee855caef01a5b8eb1d6c98696cfc | d5ee38cdd9c6bc342b1f2355f5f2bd8bd2fee855caef01a5b8eb1d6c98696cfc |
| 04 | fit | 54 | 128 | 63 | 1 | 14 | 32 | 0 | stroke | f1cd919913bea6a205009ebc81eb27061bc3e29b256be4140363be50c378b70b | f1cd919913bea6a205009ebc81eb27061bc3e29b256be4140363be50c378b70b |
| 05 | fit | 45 | 128 | 41 | 1 | 23 | 14 | 0 | stroke | a64000ba8774eacc57a0ff58e5360d4d8ef7f272d33c6568b61f4130e1640346 | a64000ba8774eacc57a0ff58e5360d4d8ef7f272d33c6568b61f4130e1640346 |
| 06 | fit | 78 | 128 | 54 | 1 | 15 | 55 | 0 | stroke | 46a744bc7772f4f887f40ed085970207a967544b2a106254f39c49523f803476 | 46a744bc7772f4f887f40ed085970207a967544b2a106254f39c49523f803476 |
| 07 | fit | 78 | 128 | 80 | 1 | 23 | 47 | 0 | stroke | 3dff6b3bcdc1e1f1785616318f954fb0e208f38c4bb4ca7dad3b6cb2301aee04 | 3dff6b3bcdc1e1f1785616318f954fb0e208f38c4bb4ca7dad3b6cb2301aee04 |
| 08 | fit | 31 | 128 | 49 | 1 | 15 | 8 | 0 | stroke | 4ebe448b608039243f92a9ab1ac02875ff070c301239ecefb7216bba226a46fa | 4ebe448b608039243f92a9ab1ac02875ff070c301239ecefb7216bba226a46fa |
| 09 | fit | 93 | 128 | 62 | 1 | 26 | 59 | 0 | stroke | d1da157719cd76bbc31a640f0143d4077ee3a341ebd5c8b11af88ceb61a03163 | d1da157719cd76bbc31a640f0143d4077ee3a341ebd5c8b11af88ceb61a03163 |
| 10 | fit | 43 | 128 | 43 | 1 | 14 | 21 | 0 | stroke | ba6544c8b011f3cdb1c5e2cc7df8f081ee79d0bdbe9c212957f18732d01c7e91 | ba6544c8b011f3cdb1c5e2cc7df8f081ee79d0bdbe9c212957f18732d01c7e91 |
| 11 | fit | 45 | 128 | 45 | 1 | 23 | 14 | 0 | skel | 88145b8e96f4b48859301bb71b62509739c5e61363b380d0653905d69406908d | 88145b8e96f4b48859301bb71b62509739c5e61363b380d0653905d69406908d |
| 12 | fit | 87 | 128 | 53 | 1 | 33 | 46 | 0 | stroke | cbe965ad6480a0dd65d07604d7a1d0a9fa056339eced8e949ba0fd0cd812ced3 | cbe965ad6480a0dd65d07604d7a1d0a9fa056339eced8e949ba0fd0cd812ced3 |
| 13 | fit | 44 | 128 | 41 | 1 | 22 | 14 | 0 | skel | 03ade59079b11f79827ceeebd9d96232caf517766a373bc25d95ae66b10913dc | 03ade59079b11f79827ceeebd9d96232caf517766a373bc25d95ae66b10913dc |
| 14 | fit | 63 | 128 | 76 | 1 | 18 | 37 | 0 | stroke | 560713b849110c017d17c616463fe78d8c15660eb2d7d635c58b897934ab8aea | 560713b849110c017d17c616463fe78d8c15660eb2d7d635c58b897934ab8aea |
| 15 | fit | 104 | 128 | 64 | 1 | 23 | 73 | 0 | stroke | a7c32d501bb7e9ecdd404fdc648e30f78bcf5d1761fbb59638248e33c2de99c4 | a7c32d501bb7e9ecdd404fdc648e30f78bcf5d1761fbb59638248e33c2de99c4 |
| 16 | held-out | 30 | 128 | 56 | 1 | 14 | 8 | 0 | stroke | 50698380aa46a01eea276c42d30f9be06c722c8caf238e30e2d0f5505dafa222 | 50698380aa46a01eea276c42d30f9be06c722c8caf238e30e2d0f5505dafa222 |
| 17 | held-out | 34 | 128 | 71 | 1 | 26 | 0 | 0 | stroke | 90afa773106de4839c29e8728dc71d0fb4373964ca0698f09b1399679da247e6 | 90afa773106de4839c29e8728dc71d0fb4373964ca0698f09b1399679da247e6 |
| 18 | held-out | 61 | 128 | 43 | 1 | 16 | 37 | 0 | stroke | b7f6b8f14ddd1c8200b2cbe52e98d1b6971972cf4b529ee56e80c3125fb519a2 | b7f6b8f14ddd1c8200b2cbe52e98d1b6971972cf4b529ee56e80c3125fb519a2 |
| 19 | held-out | 45 | 128 | 45 | 1 | 23 | 14 | 0 | skel | 4d26ac2ddc6245f0b93f9d7d4a6cba26427b13d73f190404c70200a1434458fe | 4d26ac2ddc6245f0b93f9d7d4a6cba26427b13d73f190404c70200a1434458fe |
| 20 | held-out | 46 | 128 | 63 | 1 | 14 | 24 | 0 | stroke | a6492d4341fb4f780a8ee24018989c9e1e022c34ff86d7856052fb5a52af461b | a6492d4341fb4f780a8ee24018989c9e1e022c34ff86d7856052fb5a52af461b |
| 21 | held-out | 61 | 128 | 43 | 1 | 21 | 32 | 0 | stroke | 05b06a91cd852422575dae5fc6645fda524f4cbf5de4d41b32dd3fb7c39fe212 | 05b06a91cd852422575dae5fc6645fda524f4cbf5de4d41b32dd3fb7c39fe212 |
| 22 | held-out | 61 | 128 | 72 | 1 | 17 | 36 | 0 | stroke | 18e7829a9e0c4587980251cadc93d848c991ad86b0ef22efe233b766aba326d3 | 18e7829a9e0c4587980251cadc93d848c991ad86b0ef22efe233b766aba326d3 |
| 23 | held-out | 82 | 128 | 80 | 1 | 23 | 51 | 0 | stroke | 59eb5ed5aa7282e39c2a7a659ffb688ad84af86bec82e2537b09e4feed65f45a | 59eb5ed5aa7282e39c2a7a659ffb688ad84af86bec82e2537b09e4feed65f45a |

| split | n | payload_bytes | packed_bits | jxl_bytes | exact |
| --- | --- | --- | --- | --- | --- |
| fit | 16 | 955 | 2048 | 870 | 16/16 |
| held-out | 8 | 420 | 1024 | 473 | 8/8 |
| all | 24 | 1375 | 3072 | 1343 | 24/24 |

| key | value |
| --- | --- |
| advance | PASS |
| representation_stops | no |
| correction_paid_for_skeleton | no |
| import_graph | clean |
| held_out_payload | 420 |
| held_out_packed | 1024 |
| held_out_jxl | 473 |
| held_out_exact | 8/8 |
| held_out_desc_bytes | 154 |
| held_out_corr_bytes | 202 |
| held_out_bytes_saved_vs_packed | 806 |
| fit_payload | 955 |
| fit_packed | 2048 |
| fit_jxl | 870 |
| fit_exact | 16/16 |
| all_payload | 1375 |
| all_packed | 3072 |
| all_jxl | 1343 |
| all_exact | 24/24 |
| shared_decoder_bytes | 11370 |

## Held-out versus JPEG XL, glyph by glyph
The advance bar is the held-out sum, not each glyph. Individual losses are still recorded: 18 (61 > 43), 21 (61 > 43), 23 (82 > 80); 19 ties at 45. The sum is 420 < 473.

## Exactness
Exact means redraw packed bits equal fixture packed bits. The measurement pass saw fixture_sha256 equal redraw_sha256 for 24/24, and two decoder runs matched. Exact counts above are from that pass; check.py recomputes them.

## Failures
None. JPEG XL installed and ran. No invented sizes.

## Import graph
status: clean. encode.py local imports: raster.py, decode.py, plus stdlib struct, zlib, pathlib. Does not import generator and does not read seed 20260908. See import_graph.txt.
