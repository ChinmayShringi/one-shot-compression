"""RADICAL compression experiments - breaking the pixel paradigm.

Approach 1: SVD decomposition + exact residual coding
  Image channel = rank-k approximation + integer residual
  Rank-1 captures 90%+ energy in 6KB. Residual is sparse/low-entropy.

Approach 2: Hierarchical multi-scale coding
  Encode at multiple resolutions. Smooth regions need almost no refinement.

Approach 3: Patch dictionary (self-referential)
  One 4x4 patch repeats 42K times in our image. Exploit this.

Goal: push WELL beyond JPEG-XL (917 KB) toward extreme compression.
"""

import numpy as np
from PIL import Image
import math
from collections import Counter
import time
import struct

from prism.transforms import zigzag_encode, zigzag_decode
from prism.arith import encode_symbols_adaptive, decode_symbols_adaptive


def entropy_bps(data):
    flat = data.flatten()
    if len(flat) == 0:
        return 0.0
    counts = Counter(flat.tolist())
    total = len(flat)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


# ============================================================
# APPROACH 1: SVD + Exact Residual
# ============================================================

def svd_analyze(image_path):
    """Analyze SVD compression potential with exact residual coding."""
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = img.shape
    total_pixels = h * w

    print(f"=== SVD + Exact Residual Analysis ===")
    print(f"Image: {w}x{h}")
    print()

    for rank_target in [1, 2, 5, 10, 13, 20, 30, 50]:
        total_svd_bytes = 0
        total_residual_bits = 0
        total_exact_zeros = 0

        for ch_idx in range(3):
            ch = img[:, :, ch_idx].astype(np.float64)
            U, S, Vt = np.linalg.svd(ch, full_matrices=False)

            # Rank-k approximation
            k = min(rank_target, len(S))
            approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
            approx_int = np.round(approx).astype(np.int32)

            # Exact integer residual
            residual = img[:, :, ch_idx].astype(np.int32) - approx_int
            residual = residual.clip(-255, 255)

            # SVD storage: U (h*k), S (k), Vt (k*w) as float16
            svd_bytes = (h * k + k + k * w) * 2  # float16

            # Residual entropy
            zz = zigzag_encode(residual)
            res_bps = entropy_bps(zz)
            res_bits = res_bps * total_pixels

            # Count exact zeros in residual
            exact_zeros = (residual == 0).sum()
            total_exact_zeros += exact_zeros

            total_svd_bytes += svd_bytes
            total_residual_bits += res_bits

        total_svd_kb = total_svd_bytes / 1024
        total_residual_kb = total_residual_bits / 8 / 1024
        total_kb = total_svd_kb + total_residual_kb
        zero_pct = total_exact_zeros / (total_pixels * 3) * 100

        print(f"  Rank {rank_target:>3d}: SVD={total_svd_kb:>6.0f} KB + "
              f"Residual={total_residual_kb:>6.0f} KB = "
              f"TOTAL={total_kb:>7.0f} KB | "
              f"Zeros={zero_pct:.1f}%")

    print()


# ============================================================
# APPROACH 2: Multi-Scale Progressive
# ============================================================

def multiscale_analyze(image_path):
    """Analyze multi-scale progressive compression potential."""
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = img.shape
    print(f"=== Multi-Scale Progressive Analysis ===")
    print(f"Image: {w}x{h}")
    print()

    # Build image pyramid
    current = img.astype(np.float64)
    levels = [current]
    while current.shape[0] > 2 and current.shape[1] > 2:
        # Downsample by 2 using averaging
        ch, cw = current.shape[0], current.shape[1]
        nh, nw = ch // 2, cw // 2
        downsampled = (current[:nh * 2, :nw * 2]
                       .reshape(nh, 2, nw, 2, 3)
                       .mean(axis=(1, 3)))
        levels.append(downsampled)
        current = downsampled

    print(f"  Pyramid levels: {len(levels)}")
    for i, level in enumerate(levels):
        print(f"    Level {i}: {level.shape[1]}x{level.shape[0]}")

    # Compute refinement cost at each level
    total_cost_bits = 0
    for i in range(len(levels) - 1, 0, -1):
        coarse = levels[i]
        fine = levels[i - 1]
        ch, cw = coarse.shape[0], coarse.shape[1]
        fh, fw = fine.shape[0], fine.shape[1]

        # Upsample coarse by 2 (nearest neighbor for simplicity)
        upsampled = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)
        # Trim to match fine dimensions
        upsampled = upsampled[:fh, :fw]

        # Refinement = fine - upsampled
        refinement = fine - upsampled
        refinement_int = np.round(refinement).astype(np.int32)

        # Entropy of refinement
        for ch_idx in range(3):
            ref_ch = refinement_int[:, :, ch_idx]
            zz = zigzag_encode(ref_ch)
            bps = entropy_bps(zz)
            bits = bps * fh * fw
            total_cost_bits += bits

    # Base level cost
    base = levels[-1]
    base_bytes = base.size * 2  # float16
    total_cost_bits += base_bytes * 8

    total_kb = total_cost_bits / 8 / 1024
    print(f"  Total progressive cost: {total_kb:.0f} KB")
    print(f"  vs PRISM v1: 1485 KB, JPEG-XL: 917 KB")
    print()


# ============================================================
# APPROACH 3: Patch Dictionary
# ============================================================

def patch_dict_analyze(image_path, patch_size=4):
    """Analyze patch dictionary compression potential."""
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = img.shape
    print(f"=== Patch Dictionary Analysis (patch={patch_size}x{patch_size}) ===")

    # Extract all patches
    patches = []
    patch_bytes_list = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = img[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            patch_bytes_list.append(patch.tobytes())

    n_patches = len(patches)
    unique_patches = set(patch_bytes_list)
    n_unique = len(unique_patches)

    # Build frequency table
    freq = Counter(patch_bytes_list)
    sorted_freq = freq.most_common()

    print(f"  Total patches: {n_patches:,}")
    print(f"  Unique patches: {n_unique:,} ({n_unique / n_patches * 100:.1f}%)")
    print(f"  Top 10 most common:")
    for i, (patch_bytes, count) in enumerate(sorted_freq[:10]):
        pct = count / n_patches * 100
        print(f"    #{i}: {count:,} times ({pct:.1f}%)")

    # Estimate dictionary coding cost
    # Option A: Full dictionary coding
    # Store dictionary (n_unique entries * patch_size^2 * 3 bytes)
    # + index per patch (log2(n_unique) bits)
    dict_bytes = n_unique * patch_size ** 2 * 3
    index_bits = n_patches * math.ceil(math.log2(max(n_unique, 2)))

    # Option B: Entropy-coded indices
    index_probs = [count / n_patches for _, count in sorted_freq]
    index_entropy = -sum(p * math.log2(p) for p in index_probs if p > 0)
    entropy_index_bits = index_entropy * n_patches

    total_a_kb = (dict_bytes + index_bits / 8) / 1024
    total_b_kb = (dict_bytes + entropy_index_bits / 8) / 1024

    print(f"\n  Dict size: {dict_bytes / 1024:.0f} KB ({n_unique} entries)")
    print(f"  Index (fixed): {index_bits / 8 / 1024:.0f} KB")
    print(f"  Index (entropy): {entropy_index_bits / 8 / 1024:.0f} KB (H={index_entropy:.2f} bps)")
    print(f"  Total (fixed): {total_a_kb:.0f} KB")
    print(f"  Total (entropy): {total_b_kb:.0f} KB")

    # Option C: Approximate dictionary - quantize patches
    for q_bits in [7, 6, 5, 4]:
        shift = 8 - q_bits
        quantized_patches = []
        for patch in patches:
            qp = (patch >> shift).tobytes()
            quantized_patches.append(qp)

        q_freq = Counter(quantized_patches)
        n_q_unique = len(q_freq)
        q_sorted = q_freq.most_common()
        q_probs = [c / n_patches for _, c in q_sorted]
        q_entropy = -sum(p * math.log2(p) for p in q_probs if p > 0)

        # For lossless: store quantized index + per-pixel residual
        q_dict_bytes = n_q_unique * patch_size ** 2 * 3  # quantized dict
        q_index_bits = q_entropy * n_patches  # entropy-coded indices

        # Residual: difference between original patch and quantized representative
        total_res_bits = 0
        # Each pixel residual is in range [0, 2^shift - 1] = shift bits
        total_res_bits = n_patches * patch_size ** 2 * 3 * shift

        q_total_kb = (q_dict_bytes + q_index_bits / 8 + total_res_bits / 8) / 1024
        print(f"  Quantized {q_bits}-bit: {n_q_unique:,} unique, "
              f"index H={q_entropy:.2f} bps, "
              f"total ~{q_total_kb:.0f} KB")

    print()


# ============================================================
# APPROACH 4: Combined SVD + Residual with proper coding
# ============================================================

def svd_compress_measure(image_path, rank=13):
    """Actually measure compressed size with SVD + residual coding."""
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = img.shape
    total_pixels = h * w

    print(f"=== SVD Rank-{rank} + Arithmetic Coded Residual ===")

    total_size = 0

    for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
        ch = img[:, :, ch_idx].astype(np.float64)
        U, S, Vt = np.linalg.svd(ch, full_matrices=False)

        k = rank
        approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
        approx_int = np.clip(np.round(approx), 0, 255).astype(np.int32)

        # Exact residual
        residual = img[:, :, ch_idx].astype(np.int32) - approx_int

        # Measure residual
        n_zeros = (residual == 0).sum()
        max_abs = np.abs(residual).max()
        zz = zigzag_encode(residual)
        res_bps = entropy_bps(zz)
        res_kb = res_bps * total_pixels / 8 / 1024

        # SVD component size (quantized to float16)
        svd_components = np.concatenate([
            (U[:, :k] * S[:k]).flatten(),  # h*k values
            Vt[:k, :].flatten()  # k*w values
        ])
        # Quantize to int16 range
        svd_scale = max(abs(svd_components.max()), abs(svd_components.min())) + 1e-10
        svd_quantized = np.round(svd_components / svd_scale * 32767).astype(np.int16)
        svd_bytes = svd_quantized.nbytes + 4  # +4 for scale factor

        ch_total = svd_bytes / 1024 + res_kb
        total_size += ch_total

        print(f"  {ch_name}: SVD={svd_bytes / 1024:.0f} KB + "
              f"Residual={res_kb:.0f} KB (H={res_bps:.3f} bps, "
              f"zeros={n_zeros / total_pixels * 100:.1f}%, "
              f"max={max_abs}) = {ch_total:.0f} KB")

    print(f"  TOTAL: {total_size:.0f} KB")
    print(f"  vs JPEG-XL: 917 KB | {'BETTER' if total_size < 917 else 'WORSE'}")
    print()

    # Try different ranks
    print("  --- Rank sweep ---")
    for r in [1, 2, 3, 5, 8, 10, 13, 15, 20, 30]:
        total_kb = 0
        for ch_idx in range(3):
            ch = img[:, :, ch_idx].astype(np.float64)
            U, S, Vt = np.linalg.svd(ch, full_matrices=False)
            k = min(r, len(S))
            approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
            approx_int = np.clip(np.round(approx), 0, 255).astype(np.int32)
            residual = img[:, :, ch_idx].astype(np.int32) - approx_int

            svd_bytes = (h * k + k * w) * 2 + 4
            zz = zigzag_encode(residual)
            res_bits = entropy_bps(zz) * total_pixels
            total_kb += svd_bytes / 1024 + res_bits / 8 / 1024

        print(f"    Rank {r:>3d}: {total_kb:>7.0f} KB | "
              f"{'<JXL' if total_kb < 917 else ''}")


# ============================================================
# APPROACH 5: SVD in YCoCg-R space with cross-channel prediction
# ============================================================

def svd_ycocg_measure(image_path, rank=10):
    """SVD in decorrelated color space."""
    from prism.transforms import rgb_to_ycocg_r

    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = img.shape
    total_pixels = h * w
    ycocg = rgb_to_ycocg_r(img)

    print(f"=== SVD in YCoCg-R + Ranked Residual ===")

    total_kb = 0
    for ch_idx, ch_name in enumerate(['Y', 'Co', 'Cg']):
        ch = ycocg[:, :, ch_idx].astype(np.float64)
        U, S, Vt = np.linalg.svd(ch, full_matrices=False)

        k = rank
        approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
        approx_int = np.round(approx).astype(np.int32)
        residual = ycocg[:, :, ch_idx].astype(np.int32) - approx_int

        svd_bytes = (h * k + k * w) * 2 + 4
        zz = zigzag_encode(residual)
        res_bps = entropy_bps(zz)
        res_kb = res_bps * total_pixels / 8 / 1024
        svd_kb = svd_bytes / 1024
        ch_kb = svd_kb + res_kb
        total_kb += ch_kb

        print(f"  {ch_name}: SVD={svd_kb:.0f} KB + Res={res_kb:.0f} KB = {ch_kb:.0f} KB "
              f"(H={res_bps:.3f} bps, zeros={(residual == 0).sum() / total_pixels * 100:.1f}%)")

    print(f"  TOTAL: {total_kb:.0f} KB")

    # Rank sweep
    print("  --- Rank sweep ---")
    for r in [1, 2, 3, 5, 8, 10, 13, 15, 20, 25, 30]:
        total_kb = 0
        for ch_idx in range(3):
            ch = ycocg[:, :, ch_idx].astype(np.float64)
            U, S, Vt = np.linalg.svd(ch, full_matrices=False)
            k = min(r, len(S))
            approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
            approx_int = np.round(approx).astype(np.int32)
            residual = ycocg[:, :, ch_idx].astype(np.int32) - approx_int

            svd_bytes = (h * k + k * w) * 2 + 4
            zz = zigzag_encode(residual)
            res_bits = entropy_bps(zz) * total_pixels
            total_kb += svd_bytes / 1024 + res_bits / 8 / 1024

        marker = ' <JXL' if total_kb < 917 else ''
        marker += ' <PRISM' if total_kb < 1485 else ''
        print(f"    Rank {r:>3d}: {total_kb:>7.0f} KB{marker}")


# ============================================================
# APPROACH 6: SVD + PRISM-style prediction on residual
# ============================================================

def svd_plus_prediction(image_path, rank=5):
    """SVD for global structure + prediction for local detail."""
    from prism.transforms import rgb_to_ycocg_r
    from prism.predict_fast import compute_all_predictors, select_best_mode_per_block

    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = img.shape
    total_pixels = h * w
    ycocg = rgb_to_ycocg_r(img)

    print(f"=== SVD Rank-{rank} + Prediction on Residual ===")

    total_kb = 0
    for ch_idx, ch_name in enumerate(['Y', 'Co', 'Cg']):
        ch = ycocg[:, :, ch_idx].astype(np.float64)
        U, S, Vt = np.linalg.svd(ch, full_matrices=False)

        k = rank
        approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
        approx_int = np.round(approx).astype(np.int32)
        residual = ycocg[:, :, ch_idx].astype(np.int32) - approx_int

        # SVD cost
        svd_bytes = (h * k + k * w) * 2 + 4
        svd_kb = svd_bytes / 1024

        # Now apply PRISM prediction to the RESIDUAL
        preds = compute_all_predictors(residual)
        modes, pred_residual, mode_names = select_best_mode_per_block(
            residual, preds, block_size=8
        )
        zz = zigzag_encode(pred_residual)
        res_bps = entropy_bps(zz)
        res_kb = res_bps * total_pixels / 8 / 1024

        mode_kb = modes.size / 1024
        ch_kb = svd_kb + res_kb + mode_kb
        total_kb += ch_kb

        print(f"  {ch_name}: SVD={svd_kb:.0f} KB + PredRes={res_kb:.0f} KB + Modes={mode_kb:.0f} KB = {ch_kb:.0f} KB")

    print(f"  TOTAL: {total_kb:.0f} KB")
    marker = ' BEATS JXL!' if total_kb < 917 else f' (JXL=917)'
    print(f"  {marker}")

    # Sweep
    print("  --- Rank sweep ---")
    for r in [1, 2, 3, 5, 8, 10, 13]:
        total_kb = 0
        for ch_idx in range(3):
            ch = ycocg[:, :, ch_idx].astype(np.float64)
            U, S, Vt = np.linalg.svd(ch, full_matrices=False)
            k = min(r, len(S))
            approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
            approx_int = np.round(approx).astype(np.int32)
            residual = ycocg[:, :, ch_idx].astype(np.int32) - approx_int

            svd_bytes = (h * k + k * w) * 2 + 4
            preds = compute_all_predictors(residual)
            _, pred_res, _ = select_best_mode_per_block(residual, preds, block_size=8)
            zz = zigzag_encode(pred_res)
            res_bits = entropy_bps(zz) * total_pixels
            total_kb += svd_bytes / 1024 + res_bits / 8 / 1024 + 10  # ~10KB modes

        marker = ' <JXL' if total_kb < 917 else ''
        print(f"    Rank {r:>3d}: {total_kb:>7.0f} KB{marker}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "img.png"

    svd_analyze(path)
    svd_compress_measure(path, rank=13)
    svd_ycocg_measure(path, rank=10)
    svd_plus_prediction(path, rank=5)
    multiscale_analyze(path)
    patch_dict_analyze(path, patch_size=4)
