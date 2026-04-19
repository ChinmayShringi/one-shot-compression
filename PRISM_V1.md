# PRISM v1 - Results & Architecture

## What It Is
From-scratch lossless image compression codec. No compression libraries used.
1,400 lines of Python. Custom arithmetic coder, color transform, prediction engine.

## Results on img.png (1630x1626 RGB)

| Codec | Size | Ratio | Notes |
|-------|------|-------|-------|
| Raw RGB | 7,765 KB | 1.00x | Uncompressed |
| PNG | 2,699 KB | 2.88x | Standard |
| **PRISM v1** | **1,485 KB** | **5.23x** | **From scratch** |
| WebP lossless | 1,330 KB | 5.84x | Google |
| JPEG-XL lossless | 917 KB | 8.47x | ISO standard |

## Architecture
```
RGB -> YCoCg-R color decorrelation ->
  per-channel block prediction (10 modes, 8x8 blocks) ->
    adaptive arithmetic coding (bucket-split, 0.4% of entropy) ->
      .prism file
```

## Components Built From Scratch
- `transforms.py`: Reversible YCoCg-R, lifting-based PCA, Hilbert curves, zigzag
- `predict_fast.py`: 10 vectorized predictors + polynomial field extrapolation + cross-channel + block mode selection
- `arith.py`: Witten/Neal/Cleary adaptive arithmetic coder
- `codec.py`: Full pipeline, container format, lossless verification
- Verified bit-exact lossless on full 1630x1626 image

## Why It's Limited
PRISM v1 still operates in the **pixel prediction paradigm**:
1. Treats image as grid of pixel values
2. Predicts each pixel from neighbors
3. Entropy-codes residuals

This paradigm has a floor: the conditional entropy of pixel values given their context.
For this image, that floor is ~1,278 KB. No amount of better prediction or entropy
coding within this paradigm can go below ~1,000 KB.

To reach extreme compression, we need to BREAK these assumptions entirely.
