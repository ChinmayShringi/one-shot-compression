"""GENESIS: Generative Encoding System for Image Synthesis.

The RADICAL paradigm shift: Don't store the image as data.
Store it as a PROGRAM that generates the data.

Traditional: pixel values → compressed pixel values
GENESIS: image → generative model → model parameters (compact) + corrections (sparse)

The model is a tiny "image synthesizer" - a function f(x,y) → (r,g,b)
parameterized compactly. At decode time, the decoder RUNS the model
to regenerate all pixels, then applies exact corrections.

If the model is good, most corrections are zero → extreme compression.

Approach:
1. Fit a multi-layer function to the image:
   Layer 0: Global mean color (3 bytes)
   Layer 1: Low-rank SVD (captures gradients, ~100 bytes)
   Layer 2: Piecewise linear blocks (captures regions, ~1KB)
   Layer 3: Template matching (captures repeated textures, ~1KB)
   Layer 4: Exact sparse corrections (whatever remains)

Each layer captures structure that the previous layers missed.
The corrections from layer 4 should be extremely sparse.
"""

import numpy as np
from PIL import Image
import math
from collections import Counter
from prism.transforms import rgb_to_ycocg_r, ycocg_r_to_rgb, zigzag_encode


def entropy_bps(data):
    flat = data.flatten()
    if len(flat) == 0:
        return 0.0
    counts = Counter(flat.tolist())
    total = len(flat)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def genesis_analyze(image_path):
    """Build a multi-layer generative model and measure compression."""
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = img.shape
    tp = h * w

    print(f"=== GENESIS: Multi-Layer Generative Model ===")
    print(f"Image: {w}x{h} = {tp:,} pixels")
    print()

    # Work in YCoCg-R space
    ycocg = rgb_to_ycocg_r(img)
    channels = [ycocg[:, :, i].astype(np.int32) for i in range(3)]
    ch_names = ['Y', 'Co', 'Cg']

    total_model_bytes = 0
    total_correction_bits = 0

    for ch_idx, (ch, name) in enumerate(zip(channels, ch_names)):
        print(f"  --- Channel {name} ---")
        current_residual = ch.copy()
        ch_model_bytes = 0

        # LAYER 0: Global mean
        mean_val = int(np.round(ch.mean()))
        current_residual = current_residual - mean_val
        ch_model_bytes += 2  # 2 bytes for mean
        n_zeros = (current_residual == 0).sum()
        print(f"    L0 (mean={mean_val}): zeros={n_zeros/tp*100:.1f}%, "
              f"residual range=[{current_residual.min()},{current_residual.max()}]")

        # LAYER 1: Rank-1 SVD (captures the dominant gradient)
        U, S, Vt = np.linalg.svd(current_residual.astype(np.float64), full_matrices=False)
        rank1_approx = np.round(S[0] * np.outer(U[:, 0], Vt[0, :])).astype(np.int32)
        current_residual = current_residual - rank1_approx
        # Store: U column (h values), S (1 value), V row (w values) as int16
        ch_model_bytes += (h + 1 + w) * 2
        n_zeros = (current_residual == 0).sum()
        print(f"    L1 (SVD rank-1): +{(h+1+w)*2/1024:.1f} KB, zeros={n_zeros/tp*100:.1f}%, "
              f"range=[{current_residual.min()},{current_residual.max()}]")

        # LAYER 2: Rank-2 through rank-5 SVD (more gradients/structure)
        for r in range(1, 5):
            rk_approx = np.round(S[r] * np.outer(U[:, r], Vt[r, :])).astype(np.int32)
            current_residual = current_residual - rk_approx
            ch_model_bytes += (h + 1 + w) * 2

        n_zeros = (current_residual == 0).sum()
        print(f"    L2 (SVD rank 2-5): +{4*(h+1+w)*2/1024:.1f} KB, zeros={n_zeros/tp*100:.1f}%, "
              f"range=[{current_residual.min()},{current_residual.max()}]")

        # LAYER 3: Block-mean correction (16x16 blocks)
        block_size = 16
        n_blocks_y = (h + block_size - 1) // block_size
        n_blocks_x = (w + block_size - 1) // block_size
        block_means = np.zeros((n_blocks_y, n_blocks_x), dtype=np.int32)

        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                y0, y1 = by * block_size, min((by + 1) * block_size, h)
                x0, x1 = bx * block_size, min((bx + 1) * block_size, w)
                block = current_residual[y0:y1, x0:x1]
                bm = int(np.round(block.mean()))
                block_means[by, bx] = bm
                current_residual[y0:y1, x0:x1] -= bm

        bm_bytes = n_blocks_y * n_blocks_x * 2
        ch_model_bytes += bm_bytes
        n_zeros = (current_residual == 0).sum()
        print(f"    L3 (block means): +{bm_bytes/1024:.1f} KB, zeros={n_zeros/tp*100:.1f}%, "
              f"range=[{current_residual.min()},{current_residual.max()}]")

        # LAYER 4: Self-reference (find repeating 4x4 patches in the residual)
        # For now, just measure the residual entropy
        zz = zigzag_encode(current_residual)
        res_bps = entropy_bps(zz)
        res_kb = res_bps * tp / 8 / 1024

        print(f"    L4 (corrections): H={res_bps:.3f} bps = {res_kb:.0f} KB")
        print(f"    CHANNEL TOTAL: model={ch_model_bytes/1024:.1f} KB + corrections={res_kb:.0f} KB "
              f"= {ch_model_bytes/1024 + res_kb:.0f} KB")

        total_model_bytes += ch_model_bytes
        total_correction_bits += res_bps * tp

    total_model_kb = total_model_bytes / 1024
    total_correction_kb = total_correction_bits / 8 / 1024
    total_kb = total_model_kb + total_correction_kb

    print()
    print(f"  === GENESIS TOTAL ===")
    print(f"  Model:       {total_model_kb:.0f} KB")
    print(f"  Corrections: {total_correction_kb:.0f} KB")
    print(f"  TOTAL:       {total_kb:.0f} KB")
    print(f"  vs PRISM v1: 1485 KB ({'BETTER' if total_kb < 1485 else 'WORSE'})")
    print(f"  vs JPEG-XL:  917 KB ({'BETTER' if total_kb < 917 else 'WORSE'})")

    # Now try with spatial prediction on the final residual
    print()
    print(f"  === GENESIS + PRISM Prediction on Corrections ===")
    from prism.predict_fast import compute_all_predictors, select_best_mode_per_block

    total_pred_kb = 0
    for ch_idx, name in enumerate(ch_names):
        ch = channels[ch_idx].copy()

        # Subtract model layers
        mean_val = int(np.round(ch.mean()))
        ch = ch - mean_val

        U, S, Vt = np.linalg.svd(ch.astype(np.float64), full_matrices=False)
        for r in range(5):
            rk = np.round(S[r] * np.outer(U[:, r], Vt[r, :])).astype(np.int32)
            ch = ch - rk

        # Block means
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                y0, y1 = by * block_size, min((by + 1) * block_size, h)
                x0, x1 = bx * block_size, min((bx + 1) * block_size, w)
                bm = int(np.round(ch[y0:y1, x0:x1].mean()))
                ch[y0:y1, x0:x1] -= bm

        # Predict the remaining residual
        preds = compute_all_predictors(ch)
        _, pred_res, _ = select_best_mode_per_block(ch, preds, block_size=8)
        zz = zigzag_encode(pred_res)
        res_bps = entropy_bps(zz)
        pred_kb = res_bps * tp / 8 / 1024 + 10  # +10KB mode overhead
        total_pred_kb += pred_kb
        print(f"    {name}: {pred_kb:.0f} KB (H={res_bps:.3f} bps)")

    total_with_pred = total_model_kb + total_pred_kb
    print(f"  Model + Predicted Corrections: {total_model_kb:.0f} + {total_pred_kb:.0f} = {total_with_pred:.0f} KB")
    print(f"  vs JPEG-XL: 917 KB ({'BETTER' if total_with_pred < 917 else 'WORSE'})")
    print()

    return total_with_pred


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "img.png"
    genesis_analyze(path)
