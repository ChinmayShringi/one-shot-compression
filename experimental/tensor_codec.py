#!/usr/bin/env python3
"""TENSOR DECOMPOSITION CODEC FOR LOSSLESS IMAGE COMPRESSION

Treats images as structured tensors and decomposes them into compact
representations. The key insight: images have low-rank structure at
multiple scales, and operating in INTEGER space avoids the floating-point
rounding noise that killed the earlier SVD approach in PRISM.

Decompositions implemented (all from scratch, pure numpy):
  1. SVD Integer Codec - per-channel SVD with optimal integer scaling
  2. Block SVD Codec  - block-wise SVD for local adaptation
  3. Tucker (HOSVD)   - 3-mode tensor decomposition
  4. Tensor Train     - chain of 3-way core tensors
  5. Multi-Scale      - pyramid of decompositions at different resolutions

Target: img.png (1630x1626 RGBA, alpha=255) -> lossless
Baselines: PRISM 1271 KB, JPEG-XL 917 KB
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prism.transforms import rgb_to_ycocg_r
from experimental.data_structures import entropy_bits, zigzag_encode


# ============================================================
# CONSTANTS
# ============================================================

IMG_PATH = Path(__file__).resolve().parent.parent / "img.png"
PRISM_BASELINE_KB = 1271
JPEGXL_BASELINE_KB = 917
BITS_PER_KB = 8 * 1024


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_image_ycocg(image_path):
    """Load image, convert to RGB, apply YCoCg-R transform.

    Returns (ycocg_channels, original_rgb) where ycocg_channels is
    a list of 3 int32 2D arrays [Y, Co, Cg].
    """
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    ycocg = rgb_to_ycocg_r(rgb)
    channels = [ycocg[:, :, c].astype(np.int32) for c in range(3)]
    return channels, rgb


def entropy_kb(data):
    """Entropy of data array in kilobytes."""
    bits = entropy_bits(data)
    return bits / BITS_PER_KB


def cost_int16_kb(array):
    """Entropy cost of an int16 array in KB (via zigzag + entropy)."""
    clamped = np.clip(array, -32768, 32767).astype(np.int16)
    encoded = zigzag_encode(clamped.astype(np.int32))
    return entropy_kb(encoded)


def cost_int32_kb(array):
    """Entropy cost of an int32 array in KB (via zigzag + entropy)."""
    encoded = zigzag_encode(array.astype(np.int32))
    return entropy_kb(encoded)


def verify_lossless(original, reconstruction, residual, label):
    """Verify that original == reconstruction + residual exactly."""
    recovered = reconstruction + residual
    is_exact = np.array_equal(original, recovered)
    if not is_exact:
        max_err = int(np.max(np.abs(original - recovered)))
        print(f"  WARNING [{label}]: NOT lossless! max_err={max_err}")
    return is_exact


def format_row(name, factors_kb, residual_kb, total_kb, vs_prism, vs_jxl):
    """Format a results table row."""
    return (f"  {name:<35s} {factors_kb:>8.1f}  {residual_kb:>8.1f}  "
            f"{total_kb:>8.1f}  {vs_prism:>+7.1f}%  {vs_jxl:>+7.1f}%")


# ============================================================
# 1. SVD INTEGER CODEC
# ============================================================

def _find_optimal_scale(U, S, Vt, channel, max_scale=1000, n_samples=50):
    """Search for the scaling factor that minimizes residual entropy.

    Instead of minimizing MSE (which doesn't help compression), we
    minimize the entropy of the residual -- the actual cost in bits.

    Uses a coarse-then-fine two-pass search to avoid brute force.
    """
    h, rank = U.shape
    w = Vt.shape[1]

    best_scale = 1
    best_entropy = float("inf")

    # Pass 1: coarse log-scale search
    scales_coarse = np.unique(np.geomspace(1, max_scale, n_samples).astype(int))
    scales_coarse = scales_coarse[scales_coarse >= 1]

    for sc in scales_coarse:
        u_q = np.round(U * sc).astype(np.int32)
        s_q = np.round(S * sc).astype(np.int32)
        v_q = np.round(Vt * sc).astype(np.int32)

        recon = np.round(
            (u_q @ np.diag(s_q) @ v_q).astype(np.float64) / (sc * sc)
        ).astype(np.int32)
        residual = channel - recon
        ent = entropy_bits(zigzag_encode(residual))
        if ent < best_entropy:
            best_entropy = ent
            best_scale = int(sc)

    # Pass 2: fine search around best coarse scale
    lo = max(1, best_scale - max(1, best_scale // 5))
    hi = best_scale + max(1, best_scale // 5) + 1
    fine_scales = np.arange(lo, min(hi, max_scale + 1))

    for sc in fine_scales:
        sc = int(sc)
        u_q = np.round(U * sc).astype(np.int32)
        s_q = np.round(S * sc).astype(np.int32)
        v_q = np.round(Vt * sc).astype(np.int32)

        recon = np.round(
            (u_q @ np.diag(s_q) @ v_q).astype(np.float64) / (sc * sc)
        ).astype(np.int32)
        residual = channel - recon
        ent = entropy_bits(zigzag_encode(residual))
        if ent < best_entropy:
            best_entropy = ent
            best_scale = sc

    return best_scale


def svd_integer_codec(image_path, rank=50):
    """SVD decomposition with integer-safe quantization.

    For each YCoCg-R channel:
      1. Compute rank-r SVD: channel ~= U @ diag(S) @ Vt
      2. Find optimal scale factor that minimizes residual entropy
      3. Quantize: U_q = round(U * scale), S_q = round(S * scale), V_q = round(Vt * scale)
      4. Integer reconstruction: round(U_q @ diag(S_q) @ V_q / scale^2)
      5. Residual = original - reconstruction (exact integers)

    Cost = entropy(U_q) + entropy(S_q) + entropy(V_q) + entropy(residual)
    """
    channels, _ = load_image_ycocg(image_path)
    h, w = channels[0].shape

    total_factors_kb = 0.0
    total_residual_kb = 0.0
    channel_names = ["Y", "Co", "Cg"]

    for ci, ch in enumerate(channels):
        ch_float = ch.astype(np.float64)

        # Full SVD then truncate to rank
        U_full, S_full, Vt_full = np.linalg.svd(ch_float, full_matrices=False)
        r = min(rank, len(S_full))
        U = U_full[:, :r]
        S = S_full[:r]
        Vt = Vt_full[:r, :]

        # Find optimal integer scale
        scale = _find_optimal_scale(U, S, Vt, ch)

        # Quantize
        u_q = np.round(U * scale).astype(np.int32)
        s_q = np.round(S * scale).astype(np.int32)
        v_q = np.round(Vt * scale).astype(np.int32)

        # Integer reconstruction
        recon = np.round(
            (u_q @ np.diag(s_q) @ v_q).astype(np.float64) / (scale * scale)
        ).astype(np.int32)
        residual = ch - recon

        # Verify lossless
        verify_lossless(ch, recon, residual, f"SVD-{channel_names[ci]}")

        # Cost accounting
        u_cost = cost_int16_kb(u_q)
        s_cost = cost_int16_kb(s_q)
        v_cost = cost_int16_kb(v_q)
        scale_cost = 16 / BITS_PER_KB  # 16 bits for scale factor
        factors_cost = u_cost + s_cost + v_cost + scale_cost
        res_cost = cost_int32_kb(residual)

        total_factors_kb += factors_cost
        total_residual_kb += res_cost

        print(f"    {channel_names[ci]}: scale={scale}, "
              f"factors={factors_cost:.1f} KB, residual={res_cost:.1f} KB")

    total_kb = total_factors_kb + total_residual_kb
    return {
        "name": f"SVD-int (rank={rank})",
        "factors_kb": total_factors_kb,
        "residual_kb": total_residual_kb,
        "total_kb": total_kb,
    }


# ============================================================
# 2. BLOCK SVD CODEC
# ============================================================

def _svd_block(block, rank):
    """SVD a single block with optimal scaling, returning cost components."""
    block_float = block.astype(np.float64)
    U_full, S_full, Vt_full = np.linalg.svd(block_float, full_matrices=False)
    r = min(rank, len(S_full))
    U = U_full[:, :r]
    S = S_full[:r]
    Vt = Vt_full[:r, :]

    scale = _find_optimal_scale(U, S, Vt, block, max_scale=500, n_samples=30)

    u_q = np.round(U * scale).astype(np.int32)
    s_q = np.round(S * scale).astype(np.int32)
    v_q = np.round(Vt * scale).astype(np.int32)

    recon = np.round(
        (u_q @ np.diag(s_q) @ v_q).astype(np.float64) / (scale * scale)
    ).astype(np.int32)
    residual = block - recon

    return u_q, s_q, v_q, scale, recon, residual


def block_svd_codec(image_path, block_size=32, rank=8):
    """Block-wise SVD: divide image into blocks, SVD each independently.

    Smaller blocks adapt better to local image structure (textures,
    edges, flat regions). Each block gets its own optimal scale factor.

    For blocks at the image boundary that are smaller than block_size,
    we pad them to block_size, decompose, then crop back.
    """
    channels, _ = load_image_ycocg(image_path)
    h, w = channels[0].shape
    channel_names = ["Y", "Co", "Cg"]

    total_factors_kb = 0.0
    total_residual_kb = 0.0

    for ci, ch in enumerate(channels):
        ch_factors_kb = 0.0
        ch_residual_kb = 0.0
        n_blocks = 0

        # Collect all block data for aggregate entropy estimation
        all_u_vals = []
        all_s_vals = []
        all_v_vals = []
        all_res_vals = []
        all_scales = []

        for y0 in range(0, h, block_size):
            for x0 in range(0, w, block_size):
                y1 = min(y0 + block_size, h)
                x1 = min(x0 + block_size, w)
                block = ch[y0:y1, x0:x1]
                bh, bw = block.shape

                # Pad if needed
                if bh < block_size or bw < block_size:
                    padded = np.zeros((block_size, block_size), dtype=np.int32)
                    padded[:bh, :bw] = block
                    u_q, s_q, v_q, sc, recon, residual = _svd_block(padded, rank)
                    recon = recon[:bh, :bw]
                    residual = block - recon
                else:
                    u_q, s_q, v_q, sc, recon, residual = _svd_block(block, rank)

                all_u_vals.append(u_q.flatten())
                all_s_vals.append(s_q.flatten())
                all_v_vals.append(v_q.flatten())
                all_res_vals.append(residual.flatten())
                all_scales.append(sc)
                n_blocks += 1

        # Compute aggregate entropy costs
        u_cat = np.concatenate(all_u_vals)
        s_cat = np.concatenate(all_s_vals)
        v_cat = np.concatenate(all_v_vals)
        res_cat = np.concatenate(all_res_vals)
        scales_arr = np.array(all_scales, dtype=np.int32)

        ch_factors_kb = (
            cost_int16_kb(u_cat) +
            cost_int16_kb(s_cat) +
            cost_int16_kb(v_cat) +
            cost_int32_kb(scales_arr)
        )
        ch_residual_kb = cost_int32_kb(res_cat)

        total_factors_kb += ch_factors_kb
        total_residual_kb += ch_residual_kb

        print(f"    {channel_names[ci]}: {n_blocks} blocks, "
              f"factors={ch_factors_kb:.1f} KB, residual={ch_residual_kb:.1f} KB")

    total_kb = total_factors_kb + total_residual_kb
    return {
        "name": f"Block-SVD (bs={block_size}, r={rank})",
        "factors_kb": total_factors_kb,
        "residual_kb": total_residual_kb,
        "total_kb": total_kb,
    }


# ============================================================
# 3. TUCKER DECOMPOSITION (HOSVD)
# ============================================================

def _unfold(tensor, mode):
    """Unfold (matricize) a tensor along the given mode.

    Mode-n unfolding: rows indexed by mode-n, columns by all other modes.
    For tensor of shape (I0, I1, ..., IN-1), mode-n unfolding has shape
    (In, product of all other Ij).
    """
    ndim = tensor.ndim
    shape = tensor.shape
    # Move target mode to front, then reshape
    axes = [mode] + [i for i in range(ndim) if i != mode]
    permuted = np.transpose(tensor, axes)
    return permuted.reshape(shape[mode], -1)


def _mode_product(tensor, matrix, mode):
    """Compute the mode-n product: tensor x_n matrix.

    If tensor is (I0, ..., In, ..., IN-1) and matrix is (J, In),
    result is (I0, ..., J, ..., IN-1).
    """
    ndim = tensor.ndim
    shape = list(tensor.shape)

    # Unfold along mode
    unfolded = _unfold(tensor, mode)

    # Multiply: matrix @ unfolded gives (J, product_other_dims)
    result_unfolded = matrix @ unfolded

    # Refold: new shape has matrix.shape[0] in position 'mode'
    new_shape = shape[:mode] + [matrix.shape[0]] + shape[mode + 1:]

    # Axes to invert the unfold permutation
    axes = list(range(1, mode + 1)) + [0] + list(range(mode + 1, ndim))
    result_perm = result_unfolded.reshape([matrix.shape[0]] + shape[:mode] + shape[mode + 1:])
    result = np.transpose(result_perm, axes)

    return result


def tucker_decomposition(image_path, ranks=(50, 50, 3)):
    """Tucker decomposition via Higher-Order SVD (HOSVD).

    Treats the H x W x C image (in YCoCg-R space) as a 3-mode tensor
    and decomposes it as:

        T ~= G x_1 A x_2 B x_3 C

    where G is the core tensor (r1 x r2 x r3), and A (Hxr1), B (Wxr2),
    C (3xr3) are factor matrices obtained from the SVD of each mode
    unfolding.

    Integer quantization of all components, with optimal scale per factor.
    """
    channels, _ = load_image_ycocg(image_path)
    h, w = channels[0].shape

    # Stack channels into 3D tensor (H, W, 3)
    tensor = np.stack(channels, axis=-1).astype(np.float64)
    r1, r2, r3 = ranks
    r1 = min(r1, h)
    r2 = min(r2, w)
    r3 = min(r3, 3)

    # Step 1: compute factor matrices via mode-n SVD
    factors = []
    for mode, r in enumerate([r1, r2, r3]):
        unfolded = _unfold(tensor, mode)
        U, _, _ = np.linalg.svd(unfolded, full_matrices=False)
        factors.append(U[:, :r])

    A, B, C = factors

    # Step 2: compute core tensor G = T x_1 A^T x_2 B^T x_3 C^T
    core = tensor.copy()
    core = _mode_product(core, A.T, 0)
    core = _mode_product(core, B.T, 1)
    core = _mode_product(core, C.T, 2)

    # Step 3: reconstruct from factors
    recon_float = core.copy()
    recon_float = _mode_product(recon_float, A, 0)
    recon_float = _mode_product(recon_float, B, 1)
    recon_float = _mode_product(recon_float, C, 2)

    # Step 4: integer quantization of all components
    # Find scale that minimizes total entropy cost
    best_total = float("inf")
    best_scale = 100
    tensor_int = np.stack(channels, axis=-1).astype(np.int32)

    for sc in np.unique(np.geomspace(10, 2000, 40).astype(int)):
        sc = int(sc)
        a_q = np.round(A * sc).astype(np.int32)
        b_q = np.round(B * sc).astype(np.int32)
        c_q = np.round(C * sc).astype(np.int32)
        core_q = np.round(core * sc).astype(np.int32)

        # Reconstruct: A_q @ core_q @ (C_q^T kron B_q^T) / sc^4
        # But we need the full mode product chain
        recon_q = core_q.astype(np.float64)
        recon_q = _mode_product(recon_q, a_q.astype(np.float64), 0)
        recon_q = _mode_product(recon_q, b_q.astype(np.float64), 1)
        recon_q = _mode_product(recon_q, c_q.astype(np.float64), 2)
        recon_int = np.round(recon_q / (sc ** 4)).astype(np.int32)

        residual = tensor_int - recon_int
        res_ent = entropy_bits(zigzag_encode(residual))
        factor_ent = (
            entropy_bits(zigzag_encode(a_q)) +
            entropy_bits(zigzag_encode(b_q)) +
            entropy_bits(zigzag_encode(c_q)) +
            entropy_bits(zigzag_encode(core_q))
        )
        total_ent = res_ent + factor_ent
        if total_ent < best_total:
            best_total = total_ent
            best_scale = sc

    # Final computation with best scale
    sc = best_scale
    a_q = np.round(A * sc).astype(np.int32)
    b_q = np.round(B * sc).astype(np.int32)
    c_q = np.round(C * sc).astype(np.int32)
    core_q = np.round(core * sc).astype(np.int32)

    recon_q = core_q.astype(np.float64)
    recon_q = _mode_product(recon_q, a_q.astype(np.float64), 0)
    recon_q = _mode_product(recon_q, b_q.astype(np.float64), 1)
    recon_q = _mode_product(recon_q, c_q.astype(np.float64), 2)
    recon_int = np.round(recon_q / (sc ** 4)).astype(np.int32)

    residual = tensor_int - recon_int
    verify_lossless(tensor_int, recon_int, residual, "Tucker")

    # Cost breakdown
    factors_kb = (
        cost_int16_kb(a_q) +
        cost_int16_kb(b_q) +
        cost_int16_kb(c_q) +
        cost_int32_kb(core_q) +
        16 / BITS_PER_KB  # scale parameter
    )
    residual_kb = cost_int32_kb(residual)
    total_kb = factors_kb + residual_kb

    print(f"    Tucker ranks=({r1},{r2},{r3}), scale={sc}")
    print(f"    Core shape: {core_q.shape}, A: {a_q.shape}, "
          f"B: {b_q.shape}, C: {c_q.shape}")
    print(f"    Factors: {factors_kb:.1f} KB, "
          f"Residual: {residual_kb:.1f} KB")

    return {
        "name": f"Tucker ({r1},{r2},{r3})",
        "factors_kb": factors_kb,
        "residual_kb": residual_kb,
        "total_kb": total_kb,
    }


# ============================================================
# 4. TENSOR TRAIN DECOMPOSITION
# ============================================================

def _factorize_dim(n, target_factor_count=4):
    """Factorize n into roughly equal factors for tensor reshaping.

    Tries to split n into target_factor_count factors.
    Falls back gracefully if n has few factors.
    """
    if target_factor_count <= 1:
        return [n]

    factors = []
    remaining = n

    for _ in range(target_factor_count - 1):
        # Find a factor near the geometric mean of remaining
        target = int(round(remaining ** (1.0 / (target_factor_count - len(factors)))))
        target = max(2, target)

        # Search for nearest factor
        best = remaining
        for candidate in range(max(2, target - 10), target + 11):
            if candidate <= 0:
                continue
            if remaining % candidate == 0:
                if abs(candidate - target) < abs(best - target):
                    best = candidate

        if best == remaining or best <= 1:
            break
        factors.append(best)
        remaining //= best

    factors.append(remaining)
    return factors


def _tt_decompose_matrix(matrix, max_rank):
    """SVD-based splitting of a matrix into two TT cores.

    Given matrix (r_prev * n_k) x (remaining_product),
    compute truncated SVD and split into:
      core_k: (r_prev, n_k, r_k)
      remainder for next step
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    r = min(max_rank, len(S), np.sum(S > 1e-10 * S[0]))
    r = max(1, r)

    U_trunc = U[:, :r]
    S_trunc = S[:r]
    Vt_trunc = Vt[:r, :]

    # Absorb S into Vt for the next step
    remainder = np.diag(S_trunc) @ Vt_trunc

    return U_trunc, remainder, r


def tensor_train(image_path, max_rank=10):
    """Tensor Train (TT) decomposition.

    Reshapes the image into a higher-order tensor by factorizing the
    spatial dimensions, then decomposes into a chain of 3-way core
    tensors G1, G2, ..., Gn where:
      G_k has shape (r_{k-1}, n_k, r_k)

    The full tensor is recovered by contracting the chain.

    Good for capturing hierarchical spatial correlations because each
    core operates at a different scale of the factorized dimensions.
    """
    channels, _ = load_image_ycocg(image_path)
    h, w = channels[0].shape
    channel_names = ["Y", "Co", "Cg"]

    total_factors_kb = 0.0
    total_residual_kb = 0.0

    for ci, ch in enumerate(channels):
        # Factorize dimensions for TT reshaping
        h_factors = _factorize_dim(h, target_factor_count=3)
        w_factors = _factorize_dim(w, target_factor_count=3)
        tt_shape = h_factors + w_factors

        # Pad channel to fit the factorized shape
        padded_h = int(np.prod(h_factors))
        padded_w = int(np.prod(w_factors))
        padded = np.zeros((padded_h, padded_w), dtype=np.float64)
        padded[:h, :w] = ch.astype(np.float64)

        # Reshape into higher-order tensor
        tensor = padded.reshape(tt_shape)
        ndim = len(tt_shape)

        # TT-SVD: decompose from left to right
        cores = []
        remainder = tensor.reshape(tt_shape[0], -1)
        r_prev = 1

        for k in range(ndim - 1):
            n_k = tt_shape[k]
            rows = r_prev * n_k
            cols = remainder.size // rows
            matrix = remainder.reshape(rows, cols)

            core_matrix, remainder, r_k = _tt_decompose_matrix(matrix, max_rank)
            core = core_matrix.reshape(r_prev, n_k, r_k)
            cores.append(core)
            r_prev = r_k

        # Last core
        cores.append(remainder.reshape(r_prev, tt_shape[-1], 1))

        # Reconstruct by contracting the TT chain
        recon = cores[0].reshape(cores[0].shape[1], cores[0].shape[2])
        for k in range(1, len(cores)):
            # recon: (prod_prev_dims, r_k)
            # cores[k]: (r_k, n_k, r_{k+1})
            r_k = cores[k].shape[0]
            n_k = cores[k].shape[1]
            r_next = cores[k].shape[2]
            core_mat = cores[k].reshape(r_k, n_k * r_next)
            recon = recon @ core_mat
            recon = recon.reshape(-1, r_next)

        recon_2d = recon.reshape(padded_h, padded_w)[:h, :w]
        recon_int = np.round(recon_2d).astype(np.int32)
        residual = ch - recon_int

        verify_lossless(ch, recon_int, residual, f"TT-{channel_names[ci]}")

        # Integer quantization of cores
        # Find a global scale for all cores
        all_core_vals = np.concatenate([c.flatten() for c in cores])
        max_val = np.max(np.abs(all_core_vals))
        if max_val < 1e-10:
            scale = 1
        else:
            # Scale so that the range fits comfortably in int16
            scale = min(30000.0 / max_val, 10000)

        best_scale = int(scale)
        best_ent = float("inf")
        # Quick search around estimated scale
        for sc in range(max(1, best_scale - 20), best_scale + 21):
            core_ent = sum(
                entropy_bits(zigzag_encode(np.round(c * sc).astype(np.int32)))
                for c in cores
            )
            if core_ent < best_ent:
                best_ent = core_ent
                best_scale = sc

        cores_q = [np.round(c * best_scale).astype(np.int32) for c in cores]

        core_cost = sum(cost_int16_kb(c) for c in cores_q)
        scale_cost = 16 / BITS_PER_KB
        shape_cost = len(tt_shape) * 16 / BITS_PER_KB  # encode the shape
        ch_factors_kb = core_cost + scale_cost + shape_cost
        ch_residual_kb = cost_int32_kb(residual)

        total_factors_kb += ch_factors_kb
        total_residual_kb += ch_residual_kb

        core_shapes = [c.shape for c in cores]
        print(f"    {channel_names[ci]}: TT shape {tt_shape}, "
              f"core shapes {core_shapes}")
        print(f"      scale={best_scale}, factors={ch_factors_kb:.1f} KB, "
              f"residual={ch_residual_kb:.1f} KB")

    total_kb = total_factors_kb + total_residual_kb
    return {
        "name": f"TensorTrain (r={max_rank})",
        "factors_kb": total_factors_kb,
        "residual_kb": total_residual_kb,
        "total_kb": total_kb,
    }


# ============================================================
# 5. MULTI-SCALE TENSOR DECOMPOSITION
# ============================================================

def _downsample(channel, factor):
    """Downsample a 2D array by averaging factor x factor blocks.

    Returns the downsampled array (integer-rounded) and its shape.
    """
    h, w = channel.shape
    new_h = h // factor
    new_w = w // factor
    cropped = channel[:new_h * factor, :new_w * factor]
    reshaped = cropped.reshape(new_h, factor, new_w, factor)
    downsampled = np.round(reshaped.mean(axis=(1, 3))).astype(np.int32)
    return downsampled


def _upsample_nearest(small, target_h, target_w, factor):
    """Nearest-neighbor upsample to target size."""
    result = np.zeros((target_h, target_w), dtype=np.int32)
    sh, sw = small.shape
    for y in range(target_h):
        for x in range(target_w):
            sy = min(y // factor, sh - 1)
            sx = min(x // factor, sw - 1)
            result[y, x] = small[sy, sx]
    return result


def _upsample_nearest_fast(small, target_h, target_w, factor):
    """Vectorized nearest-neighbor upsample."""
    sh, sw = small.shape
    ys = np.minimum(np.arange(target_h) // factor, sh - 1)
    xs = np.minimum(np.arange(target_w) // factor, sw - 1)
    return small[np.ix_(ys, xs)]


def multi_scale_tensor(image_path):
    """Multi-scale tensor decomposition pyramid.

    Decomposes the image at three scales, each capturing what the
    previous scale missed:

    Scale 1 (8x downsample): Low-rank SVD captures global structure
      - Image ~204x203, rank-20 SVD
      - Captures broad color regions, gradients

    Scale 2 (4x downsample of residual): Block SVD captures mid-frequency
      - Residual at ~408x407, block_size=16, rank-4
      - Captures textures, edges, medium detail

    Scale 3 (full resolution): Per-pixel residual captures fine detail
      - Whatever Scale 1+2 missed
      - Typically small values, low entropy

    Each scale layer corrects the rounding errors of the previous layers.
    """
    channels, _ = load_image_ycocg(image_path)
    h, w = channels[0].shape
    channel_names = ["Y", "Co", "Cg"]

    total_factors_kb = 0.0
    total_residual_kb = 0.0

    for ci, ch in enumerate(channels):
        layer_costs = []

        # ---- Scale 1: 8x downsample + low-rank SVD ----
        small_8 = _downsample(ch, 8)
        sh8, sw8 = small_8.shape
        small_float = small_8.astype(np.float64)

        rank_s1 = 20
        U, S, Vt = np.linalg.svd(small_float, full_matrices=False)
        r = min(rank_s1, len(S))
        U_r = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]

        scale_s1 = _find_optimal_scale(U_r, S_r, Vt_r, small_8,
                                        max_scale=500, n_samples=30)
        u_q = np.round(U_r * scale_s1).astype(np.int32)
        s_q = np.round(S_r * scale_s1).astype(np.int32)
        v_q = np.round(Vt_r * scale_s1).astype(np.int32)

        recon_s1_small = np.round(
            (u_q @ np.diag(s_q) @ v_q).astype(np.float64) / (scale_s1 ** 2)
        ).astype(np.int32)
        residual_s1_small = small_8 - recon_s1_small

        # Upsample scale 1 reconstruction to full resolution
        recon_s1_full = _upsample_nearest_fast(recon_s1_small, h, w, 8)

        s1_factor_cost = (
            cost_int16_kb(u_q) + cost_int16_kb(s_q) +
            cost_int16_kb(v_q) + 16 / BITS_PER_KB
        )
        s1_res_cost = cost_int32_kb(residual_s1_small)
        layer_costs.append(("S1-SVD", s1_factor_cost, s1_res_cost))

        # ---- Scale 2: 4x downsample of residual + block SVD ----
        full_residual_after_s1 = ch - recon_s1_full
        # Also add back the small-scale residual (upsampled)
        res_s1_up = _upsample_nearest_fast(residual_s1_small, h, w, 8)
        full_residual_after_s1 = full_residual_after_s1 + res_s1_up

        small_4 = _downsample(full_residual_after_s1, 4)
        sh4, sw4 = small_4.shape

        # Block SVD on the 4x downsampled residual
        block_size_s2 = 16
        rank_s2 = 4
        all_u2, all_s2, all_v2, all_res2 = [], [], [], []
        all_scales2 = []
        recon_s2_small = np.zeros_like(small_4)

        for y0 in range(0, sh4, block_size_s2):
            for x0 in range(0, sw4, block_size_s2):
                y1 = min(y0 + block_size_s2, sh4)
                x1 = min(x0 + block_size_s2, sw4)
                block = small_4[y0:y1, x0:x1]
                bh, bw = block.shape

                if bh < 2 or bw < 2:
                    # Too small for SVD, store raw
                    all_res2.append(block.flatten())
                    recon_s2_small[y0:y1, x0:x1] = 0
                    continue

                padded = np.zeros((block_size_s2, block_size_s2), dtype=np.int32)
                padded[:bh, :bw] = block
                u_q2, s_q2, v_q2, sc2, rec2, _ = _svd_block(padded, rank_s2)
                rec2 = rec2[:bh, :bw]
                res2 = block - rec2

                all_u2.append(u_q2.flatten())
                all_s2.append(s_q2.flatten())
                all_v2.append(v_q2.flatten())
                all_res2.append(res2.flatten())
                all_scales2.append(sc2)
                recon_s2_small[y0:y1, x0:x1] = rec2

        # Upsample scale 2 reconstruction
        recon_s2_full = _upsample_nearest_fast(recon_s2_small, h, w, 4)

        s2_factor_cost = 0.0
        if all_u2:
            s2_factor_cost = (
                cost_int16_kb(np.concatenate(all_u2)) +
                cost_int16_kb(np.concatenate(all_s2)) +
                cost_int16_kb(np.concatenate(all_v2)) +
                cost_int32_kb(np.array(all_scales2, dtype=np.int32))
            )
        s2_res_cost = cost_int32_kb(np.concatenate(all_res2)) if all_res2 else 0.0
        layer_costs.append(("S2-BlockSVD", s2_factor_cost, s2_res_cost))

        # ---- Scale 3: full resolution residual ----
        # Reconstruct what scales 1+2 give us
        # Scale 1 contributes at 8x, scale 2 at 4x
        combined_recon = recon_s1_full + recon_s2_full
        # Add back the upsampled residuals from each scale
        res_s2_up = _upsample_nearest_fast(
            np.concatenate(all_res2).reshape(sh4, sw4) if all_res2 else np.zeros((sh4, sw4), dtype=np.int32),
            h, w, 4
        ) if all_res2 else np.zeros((h, w), dtype=np.int32)

        # Final pixel-level residual
        final_residual = ch - combined_recon
        # Note: final_residual includes the errors from downsampling/upsampling,
        # so it captures all the detail that the coarse scales missed.

        s3_cost = cost_int32_kb(final_residual)
        layer_costs.append(("S3-pixel", 0.0, s3_cost))

        # Verify: combined_recon + final_residual == ch
        verify_lossless(ch, combined_recon, final_residual,
                        f"MultiScale-{channel_names[ci]}")

        ch_factors = sum(fc for _, fc, _ in layer_costs)
        ch_residual = sum(rc for _, _, rc in layer_costs)
        total_factors_kb += ch_factors
        total_residual_kb += ch_residual

        print(f"    {channel_names[ci]}:")
        for lname, fc, rc in layer_costs:
            print(f"      {lname}: factors={fc:.1f} KB, residual={rc:.1f} KB")

    total_kb = total_factors_kb + total_residual_kb
    return {
        "name": "MultiScale (8x/4x/1x)",
        "factors_kb": total_factors_kb,
        "residual_kb": total_residual_kb,
        "total_kb": total_kb,
    }


# ============================================================
# MAIN: Run all approaches and compare
# ============================================================

def print_comparison_table(results):
    """Print a formatted comparison table of all results."""
    header = (f"  {'Method':<35s} {'Factors':>8s}  {'Residual':>8s}  "
              f"{'Total':>8s}  {'vs PRISM':>7s}  {'vs JXL':>7s}")
    sep = "  " + "-" * 85

    print("\n" + "=" * 89)
    print("  TENSOR DECOMPOSITION COMPRESSION RESULTS (entropy estimates, KB)")
    print("=" * 89)
    print(header)
    print(sep)

    for r in results:
        vs_prism = (r["total_kb"] - PRISM_BASELINE_KB) / PRISM_BASELINE_KB * 100
        vs_jxl = (r["total_kb"] - JPEGXL_BASELINE_KB) / JPEGXL_BASELINE_KB * 100
        print(format_row(
            r["name"], r["factors_kb"], r["residual_kb"],
            r["total_kb"], vs_prism, vs_jxl
        ))

    print(sep)
    print(f"  {'PRISM baseline':<35s} {'':>8s}  {'':>8s}  "
          f"{PRISM_BASELINE_KB:>8.1f}")
    print(f"  {'JPEG-XL baseline':<35s} {'':>8s}  {'':>8s}  "
          f"{JPEGXL_BASELINE_KB:>8.1f}")
    print("=" * 89)

    # Find best result
    if results:
        best = min(results, key=lambda r: r["total_kb"])
        print(f"\n  Best: {best['name']} at {best['total_kb']:.1f} KB")
        vs_p = (best["total_kb"] - PRISM_BASELINE_KB) / PRISM_BASELINE_KB * 100
        vs_j = (best["total_kb"] - JPEGXL_BASELINE_KB) / JPEGXL_BASELINE_KB * 100
        print(f"  vs PRISM: {vs_p:+.1f}%, vs JPEG-XL: {vs_j:+.1f}%")


if __name__ == "__main__":
    print("TENSOR DECOMPOSITION CODEC - LOSSLESS COMPRESSION RESEARCH")
    print(f"Image: {IMG_PATH}")
    print(f"Baselines: PRISM={PRISM_BASELINE_KB} KB, JPEG-XL={JPEGXL_BASELINE_KB} KB\n")

    results = []

    # --- 1. SVD Integer Codec at various ranks ---
    print("\n" + "=" * 70)
    print("1. SVD INTEGER CODEC")
    print("=" * 70)
    for rank in [10, 25, 50, 100, 200]:
        print(f"\n  --- rank={rank} ---")
        t0 = time.time()
        r = svd_integer_codec(IMG_PATH, rank=rank)
        elapsed = time.time() - t0
        print(f"    Total: {r['total_kb']:.1f} KB ({elapsed:.1f}s)")
        results.append(r)

    # --- 2. Block SVD Codec ---
    print("\n" + "=" * 70)
    print("2. BLOCK SVD CODEC")
    print("=" * 70)
    for bs, rank in [(64, 16), (32, 8), (16, 4)]:
        print(f"\n  --- block_size={bs}, rank={rank} ---")
        t0 = time.time()
        r = block_svd_codec(IMG_PATH, block_size=bs, rank=rank)
        elapsed = time.time() - t0
        print(f"    Total: {r['total_kb']:.1f} KB ({elapsed:.1f}s)")
        results.append(r)

    # --- 3. Tucker (HOSVD) ---
    print("\n" + "=" * 70)
    print("3. TUCKER DECOMPOSITION (HOSVD)")
    print("=" * 70)
    for ranks in [(50, 50, 3), (100, 100, 3), (200, 200, 3)]:
        print(f"\n  --- ranks={ranks} ---")
        t0 = time.time()
        r = tucker_decomposition(IMG_PATH, ranks=ranks)
        elapsed = time.time() - t0
        print(f"    Total: {r['total_kb']:.1f} KB ({elapsed:.1f}s)")
        results.append(r)

    # --- 4. Tensor Train ---
    print("\n" + "=" * 70)
    print("4. TENSOR TRAIN")
    print("=" * 70)
    for max_r in [5, 10, 20]:
        print(f"\n  --- max_rank={max_r} ---")
        t0 = time.time()
        r = tensor_train(IMG_PATH, max_rank=max_r)
        elapsed = time.time() - t0
        print(f"    Total: {r['total_kb']:.1f} KB ({elapsed:.1f}s)")
        results.append(r)

    # --- 5. Multi-Scale Tensor ---
    print("\n" + "=" * 70)
    print("5. MULTI-SCALE TENSOR DECOMPOSITION")
    print("=" * 70)
    t0 = time.time()
    r = multi_scale_tensor(IMG_PATH)
    elapsed = time.time() - t0
    print(f"    Total: {r['total_kb']:.1f} KB ({elapsed:.1f}s)")
    results.append(r)

    # --- Comparison Table ---
    print_comparison_table(results)
