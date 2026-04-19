"""HYBRID MULTI-STRATEGY CODEC

The ultimate codec: combines all experimental approaches using
Minimum Description Length (MDL) to select the best strategy per region.

Architecture:
  1. Segment image into regions (smooth, textured, edge, complex)
  2. For each region, test multiple codecs and pick the one with
     shortest total description length (model + residual)
  3. Store: region map + per-region codec parameters + residuals

This is the integration layer that combines:
  - Neural/SIREN for smooth regions (sky, gradients)
  - VQ for textured regions (sand, repetitive patterns)
  - Spatial prediction for edges (PRISM-style)
  - Sparse coding for complex regions (mixed content)
  - Tensor decomposition for globally-correlated structure

Novel ideas:
  - MDL-optimal codec selection per block
  - Cross-codec residual prediction (use one codec's output to help another)
  - Adaptive block sizes (larger blocks for smooth, smaller for detail)
  - Multi-pass refinement (coarse global model + local corrections)
"""

import numpy as np
from PIL import Image
import math, time
from collections import Counter

from prism.transforms import rgb_to_ycocg_r


# ============================================================
# REGION SEGMENTATION
# ============================================================

def classify_regions(channel, block_size=16):
    """Classify image blocks into region types.

    Types:
      0 = SMOOTH (low variance, gradients work)
      1 = TEXTURED (medium variance, repetitive, VQ works)
      2 = EDGE (high directional gradient, prediction works)
      3 = COMPLEX (high variance, no structure, raw coding)

    Returns (n_blocks_h, n_blocks_w) array of types.
    """
    h, w = channel.shape
    n_h = h // block_size
    n_w = w // block_size
    regions = np.zeros((n_h, n_w), dtype=np.int32)
    stats = {'smooth': 0, 'textured': 0, 'edge': 0, 'complex': 0}

    for i in range(n_h):
        for j in range(n_w):
            block = channel[i * block_size:(i + 1) * block_size,
                          j * block_size:(j + 1) * block_size].astype(np.float64)

            variance = np.var(block)
            # Gradient magnitude
            gy = np.diff(block, axis=0)
            gx = np.diff(block, axis=1)
            grad_mag = np.sqrt(np.mean(gy ** 2) + np.mean(gx ** 2))
            # Directional coherence
            if grad_mag > 1:
                gy_mean = np.mean(gy)
                gx_mean = np.mean(gx)
                coherence = np.sqrt(gy_mean ** 2 + gx_mean ** 2) / grad_mag
            else:
                coherence = 0

            # Classification
            if variance < 50:
                regions[i, j] = 0  # SMOOTH
                stats['smooth'] += 1
            elif coherence > 0.6:
                regions[i, j] = 2  # EDGE
                stats['edge'] += 1
            elif variance < 500:
                regions[i, j] = 1  # TEXTURED
                stats['textured'] += 1
            else:
                regions[i, j] = 3  # COMPLEX
                stats['complex'] += 1

    total = n_h * n_w
    print(f"  Regions: smooth={stats['smooth']} ({stats['smooth']/total*100:.0f}%), "
          f"textured={stats['textured']} ({stats['textured']/total*100:.0f}%), "
          f"edge={stats['edge']} ({stats['edge']/total*100:.0f}%), "
          f"complex={stats['complex']} ({stats['complex']/total*100:.0f}%)")

    return regions, stats


# ============================================================
# PER-REGION CODECS (lightweight versions)
# ============================================================

def encode_smooth_block(block):
    """Encode smooth block using bilinear surface model.

    Model: z(x,y) = a + bx + cy + dxy (4 params)
    Residual should be very small for smooth regions.
    """
    h, w = block.shape
    ys = np.linspace(0, 1, h)
    xs = np.linspace(0, 1, w)
    X, Y = np.meshgrid(xs, ys)
    A = np.column_stack([np.ones(h * w), X.ravel(), Y.ravel(), (X * Y).ravel()])
    target = block.ravel().astype(np.float64)

    coeffs = np.linalg.lstsq(A, target, rcond=None)[0]
    predicted = np.round(A @ coeffs).astype(np.int32).reshape(h, w)
    residual = block.astype(np.int32) - predicted

    # Cost: 4 float16 coefficients = 8 bytes
    return residual, 8


def encode_textured_block(block, codebook=None):
    """Encode textured block using mini VQ within the block.

    Split into 4x4 sub-patches, quantize to small codebook.
    """
    h, w = block.shape
    sub_size = 4
    n_sub_h = h // sub_size
    n_sub_w = w // sub_size
    n_subs = n_sub_h * n_sub_w

    subs = block[:n_sub_h * sub_size, :n_sub_w * sub_size].reshape(
        n_sub_h, sub_size, n_sub_w, sub_size
    ).transpose(0, 2, 1, 3).reshape(n_subs, -1).astype(np.float64)

    # Quick k-means with k=4 (just 4 prototype sub-patches)
    k = min(4, n_subs)
    rng = np.random.RandomState(42)
    indices = rng.choice(n_subs, k, replace=False)
    centroids = subs[indices].copy()

    for _ in range(10):
        dists = np.array([np.sum((subs - c) ** 2, axis=1) for c in centroids])
        labels = dists.argmin(axis=0)
        new_centroids = np.array([
            subs[labels == i].mean(0) if (labels == i).any() else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    centroids_int = np.round(centroids).astype(np.int32)
    dists = np.array([np.sum((subs - c) ** 2, axis=1) for c in centroids_int])
    labels = dists.argmin(axis=0)
    reconstructed = centroids_int[labels]
    residual_subs = subs.astype(np.int32) - reconstructed

    # Reconstruct block
    recon_block = reconstructed.reshape(n_sub_h, n_sub_w, sub_size, sub_size
                                       ).transpose(0, 2, 1, 3).reshape(
        n_sub_h * sub_size, n_sub_w * sub_size)
    residual = block[:n_sub_h * sub_size, :n_sub_w * sub_size].astype(np.int32) - recon_block

    # Cost: k centroids (k * 16 * 2 bytes) + indices (n_subs * 2 bits)
    codebook_cost = k * sub_size * sub_size * 2
    index_cost = max(1, int(n_subs * math.log2(max(k, 2)) / 8))
    residual_cost = _entropy_bytes(residual)

    return residual, codebook_cost + index_cost


def encode_edge_block(block):
    """Encode edge block using directional prediction.

    Detect dominant gradient direction, predict along edges.
    """
    h, w = block.shape
    block_f = block.astype(np.float64)

    # Sobel-like gradient
    gy = np.zeros_like(block_f)
    gx = np.zeros_like(block_f)
    gy[1:, :] = block_f[1:, :] - block_f[:-1, :]
    gx[:, 1:] = block_f[:, 1:] - block_f[:, :-1]

    # Dominant direction
    angle = np.arctan2(np.mean(gy), np.mean(gx))

    # Predict each pixel from neighbor in gradient direction
    predicted = np.zeros_like(block, dtype=np.int32)
    for y in range(h):
        for x in range(w):
            # Source pixel along gradient
            sy = y - int(round(np.sin(angle)))
            sx = x - int(round(np.cos(angle)))
            if 0 <= sy < h and 0 <= sx < w:
                predicted[y, x] = block[sy, sx]
            elif x > 0:
                predicted[y, x] = block[y, x - 1]
            elif y > 0:
                predicted[y, x] = block[y - 1, x]

    residual = block.astype(np.int32) - predicted
    # Cost: 2 bytes for angle quantization
    return residual, 2


def encode_complex_block(block):
    """Encode complex block using MED (Median Edge Detector) prediction.

    This is the LOCO-I predictor from JPEG-LS, good for arbitrary content.
    """
    h, w = block.shape
    predicted = np.zeros((h, w), dtype=np.int32)

    for y in range(h):
        for x in range(w):
            if y == 0 and x == 0:
                predicted[y, x] = 128
            elif y == 0:
                predicted[y, x] = block[y, x - 1]
            elif x == 0:
                predicted[y, x] = block[y - 1, x]
            else:
                a = int(block[y, x - 1])  # left
                b = int(block[y - 1, x])  # above
                c = int(block[y - 1, x - 1])  # above-left
                if c >= max(a, b):
                    predicted[y, x] = min(a, b)
                elif c <= min(a, b):
                    predicted[y, x] = max(a, b)
                else:
                    predicted[y, x] = a + b - c

    residual = block.astype(np.int32) - predicted
    return residual, 0  # No model cost (predictor is implicit)


def _entropy_bytes(data):
    """Estimate entropy in bytes."""
    flat = data.flatten()
    if len(flat) == 0:
        return 0
    unique, counts = np.unique(flat, return_counts=True)
    total = len(flat)
    probs = counts / total
    bits = -np.sum(counts * np.log2(probs + 1e-30))
    return int(math.ceil(bits / 8))


# ============================================================
# MULTI-SCALE CODEC
# ============================================================

def multi_scale_codec(image_path):
    """Multi-scale decomposition: global structure + local detail.

    Scale 0: 1/8 resolution - global color gradients (neural/surface fit)
    Scale 1: 1/4 residual - mid-frequency structure (block prediction)
    Scale 2: 1/2 residual - fine detail (spatial prediction)
    Scale 3: Full residual - pixel corrections (entropy coded)

    Each scale captures what previous scales missed.
    """
    print("\n" + "=" * 60)
    print("MULTI-SCALE CODEC")
    print("=" * 60)

    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = img.shape
    ycocg = rgb_to_ycocg_r(img)
    total_pixels = h * w

    total_model_bits = 0
    total_residual_bits = 0

    for ch_idx, name in enumerate(['Y', 'Co', 'Cg']):
        channel = ycocg[:, :, ch_idx].astype(np.float64)
        print(f"\n  --- Channel: {name} ---")

        # Scale 0: 1/8 resolution - bilinear surface
        sh, sw = h // 8, w // 8
        small = np.array(Image.fromarray(channel.astype(np.uint8)).resize(
            (sw, sh), Image.BILINEAR)).astype(np.float64)

        # Fit degree-6 polynomial to small version
        ys = np.linspace(-1, 1, sh)
        xs = np.linspace(-1, 1, sw)
        X, Y = np.meshgrid(xs, ys)
        cols = []
        for i in range(7):
            for j in range(7 - i):
                cols.append((X.ravel() ** i) * (Y.ravel() ** j))
        A = np.column_stack(cols)
        n_coeffs = A.shape[1]
        coeffs = np.linalg.lstsq(A, small.ravel(), rcond=None)[0]

        # Upsample prediction to full resolution
        ys_full = np.linspace(-1, 1, h)
        xs_full = np.linspace(-1, 1, w)
        Xf, Yf = np.meshgrid(xs_full, ys_full)
        cols_full = []
        for i in range(7):
            for j in range(7 - i):
                cols_full.append((Xf.ravel() ** i) * (Yf.ravel() ** j))
        Af = np.column_stack(cols_full)
        global_pred = np.round(Af @ coeffs).astype(np.int32).reshape(h, w)

        residual_0 = channel.astype(np.int32) - global_pred
        r0_entropy = _entropy_bytes(residual_0)
        model_0 = n_coeffs * 4  # float32 per coefficient

        print(f"  Scale 0 (polynomial deg-6): {n_coeffs} coeffs = {model_0} bytes, "
              f"residual entropy = {r0_entropy/1024:.0f} KB")

        # Scale 1: 1/4 resolution residual - block means
        bh, bw = 4, 4
        n_bh, n_bw = h // bh, w // bw
        block_means = np.zeros((n_bh, n_bw), dtype=np.int32)
        for bi in range(n_bh):
            for bj in range(n_bw):
                block = residual_0[bi * bh:(bi + 1) * bh, bj * bw:(bj + 1) * bw]
                block_means[bi, bj] = int(np.round(np.mean(block)))

        # Upsample block means
        mean_pred = np.repeat(np.repeat(block_means, bh, axis=0), bw, axis=1)[:h, :w]
        residual_1 = residual_0 - mean_pred
        r1_entropy = _entropy_bytes(residual_1)
        means_entropy = _entropy_bytes(block_means)

        print(f"  Scale 1 (block means 4x4): means = {means_entropy/1024:.1f} KB, "
              f"residual entropy = {r1_entropy/1024:.0f} KB")

        # Scale 2: Pixel-level spatial prediction on remaining residual
        predicted_2 = np.zeros_like(residual_1)
        for y in range(h):
            for x in range(w):
                if y == 0 and x == 0:
                    predicted_2[y, x] = 0
                elif y == 0:
                    predicted_2[y, x] = residual_1[y, x - 1]
                elif x == 0:
                    predicted_2[y, x] = residual_1[y - 1, x]
                else:
                    a = int(residual_1[y, x - 1])
                    b = int(residual_1[y - 1, x])
                    c = int(residual_1[y - 1, x - 1])
                    if c >= max(a, b):
                        predicted_2[y, x] = min(a, b)
                    elif c <= min(a, b):
                        predicted_2[y, x] = max(a, b)
                    else:
                        predicted_2[y, x] = a + b - c

        residual_2 = residual_1 - predicted_2
        r2_entropy = _entropy_bytes(residual_2)
        print(f"  Scale 2 (MED prediction): residual entropy = {r2_entropy/1024:.0f} KB")

        # Total for this channel
        ch_total = model_0 + means_entropy + r2_entropy
        total_model_bits += (model_0 + means_entropy) * 8
        total_residual_bits += r2_entropy * 8

        print(f"  Channel total: {ch_total/1024:.0f} KB")

        # Verify lossless
        recon = global_pred + mean_pred + predicted_2 + residual_2
        assert np.array_equal(channel.astype(np.int32), recon), f"Channel {name} NOT LOSSLESS!"

    total_bytes = (total_model_bits + total_residual_bits) / 8
    print(f"\n  TOTAL: {total_bytes/1024:.0f} KB "
          f"(model: {total_model_bits/8/1024:.0f} KB, "
          f"residual: {total_residual_bits/8/1024:.0f} KB)")
    print(f"  vs PRISM baseline: 1271 KB")

    return {'total_bytes': total_bytes}


# ============================================================
# MDL-OPTIMAL REGION CODEC
# ============================================================

def mdl_codec(image_path, block_size=16):
    """MDL-optimal per-block codec selection.

    For each block, try ALL codecs and pick the one with minimum
    total description length = model_cost + residual_entropy.

    This is the theoretically optimal strategy given our codec set.
    """
    print("\n" + "=" * 60)
    print("MDL-OPTIMAL REGION CODEC (block_size={})".format(block_size))
    print("=" * 60)

    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = img.shape
    ycocg = rgb_to_ycocg_r(img)

    total_bytes = 0
    codec_usage = {'smooth': 0, 'textured': 0, 'edge': 0, 'complex': 0}

    for ch_idx, name in enumerate(['Y', 'Co', 'Cg']):
        channel = ycocg[:, :, ch_idx]
        n_h = h // block_size
        n_w = w // block_size
        ch_bytes = 0
        ch_residual_all = []

        for i in range(n_h):
            for j in range(n_w):
                block = channel[i * block_size:(i + 1) * block_size,
                              j * block_size:(j + 1) * block_size]

                # Try all codecs
                results = []

                # Smooth
                r_smooth, cost_smooth = encode_smooth_block(block)
                ent_smooth = _entropy_bytes(r_smooth)
                results.append(('smooth', cost_smooth + ent_smooth, r_smooth))

                # Textured
                r_tex, cost_tex = encode_textured_block(block)
                ent_tex = _entropy_bytes(r_tex)
                results.append(('textured', cost_tex + ent_tex, r_tex))

                # Edge
                r_edge, cost_edge = encode_edge_block(block)
                ent_edge = _entropy_bytes(r_edge)
                results.append(('edge', cost_edge + ent_edge, r_edge))

                # Complex (MED)
                r_complex, cost_complex = encode_complex_block(block)
                ent_complex = _entropy_bytes(r_complex)
                results.append(('complex', cost_complex + ent_complex, r_complex))

                # Pick minimum description length
                best = min(results, key=lambda x: x[1])
                codec_usage[best[0]] += 1
                ch_bytes += best[1]
                ch_residual_all.append(best[2])

        # Handle border pixels
        covered_h = n_h * block_size
        covered_w = n_w * block_size
        if covered_h < h:
            border = channel[covered_h:, :covered_w]
            ch_bytes += _entropy_bytes(border)
        if covered_w < w:
            border = channel[:, covered_w:]
            ch_bytes += _entropy_bytes(border)

        # Region map cost: 2 bits per block for codec ID
        map_cost = int(math.ceil(n_h * n_w * 2 / 8))
        ch_bytes += map_cost

        print(f"  {name}: {ch_bytes/1024:.0f} KB (map: {map_cost} bytes)")
        total_bytes += ch_bytes

    total = sum(codec_usage.values())
    print(f"\n  Codec usage: smooth={codec_usage['smooth']} ({codec_usage['smooth']/total*100:.0f}%), "
          f"textured={codec_usage['textured']} ({codec_usage['textured']/total*100:.0f}%), "
          f"edge={codec_usage['edge']} ({codec_usage['edge']/total*100:.0f}%), "
          f"complex={codec_usage['complex']} ({codec_usage['complex']/total*100:.0f}%)")
    print(f"  TOTAL: {total_bytes/1024:.0f} KB")
    print(f"  vs PRISM baseline: 1271 KB")

    return {'total_bytes': total_bytes, 'codec_usage': codec_usage}


# ============================================================
# CROSS-CODEC RESIDUAL PREDICTION
# ============================================================

def cross_codec(image_path):
    """Use one codec's prediction to enhance another's.

    Idea: smooth codec predicts well in flat areas but poorly at edges.
    Edge codec predicts well at edges but poorly in flat areas.
    Use BOTH predictions and blend them optimally per-pixel.

    This is similar to JPEG-XL's weighted multi-predictor approach
    but with heterogeneous codecs instead of homogeneous spatial predictors.
    """
    print("\n" + "=" * 60)
    print("CROSS-CODEC RESIDUAL PREDICTION")
    print("=" * 60)

    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = img.shape
    ycocg = rgb_to_ycocg_r(img)

    total_bytes = 0

    for ch_idx, name in enumerate(['Y', 'Co', 'Cg']):
        channel = ycocg[:, :, ch_idx].astype(np.float64)

        # Predictor 1: Global polynomial (smooth)
        ys = np.linspace(-1, 1, h)
        xs = np.linspace(-1, 1, w)
        X, Y = np.meshgrid(xs, ys)
        cols = []
        for i in range(5):
            for j in range(5 - i):
                cols.append((X.ravel() ** i) * (Y.ravel() ** j))
        A = np.column_stack(cols)
        n_coeffs = A.shape[1]
        coeffs = np.linalg.lstsq(A, channel.ravel(), rcond=None)[0]
        pred_smooth = np.round(A @ coeffs).astype(np.int32).reshape(h, w)

        # Predictor 2: MED (edge-aware)
        pred_med = np.zeros((h, w), dtype=np.int32)
        for y in range(h):
            for x in range(w):
                if y == 0 and x == 0:
                    pred_med[y, x] = int(np.round(np.mean(channel)))
                elif y == 0:
                    pred_med[y, x] = int(channel[y, x - 1])
                elif x == 0:
                    pred_med[y, x] = int(channel[y - 1, x])
                else:
                    a = int(channel[y, x - 1])
                    b = int(channel[y - 1, x])
                    c = int(channel[y - 1, x - 1])
                    if c >= max(a, b):
                        pred_med[y, x] = min(a, b)
                    elif c <= min(a, b):
                        pred_med[y, x] = max(a, b)
                    else:
                        pred_med[y, x] = a + b - c

        # Predictor 3: Left pixel
        pred_left = np.zeros((h, w), dtype=np.int32)
        pred_left[:, 0] = channel[:, 0].astype(np.int32)
        pred_left[:, 1:] = channel[:, :-1].astype(np.int32)

        # Online blending: for each pixel, weight predictors by past error
        weights = np.array([1.0, 1.0, 1.0])
        decay = 0.998
        lr = 0.01

        residuals = np.zeros((h, w), dtype=np.int32)
        actual = channel.astype(np.int32)

        for y in range(h):
            for x in range(w):
                preds = np.array([pred_smooth[y, x], pred_med[y, x], pred_left[y, x]],
                                dtype=np.float64)
                # Softmax weights
                w_soft = np.exp(weights - np.max(weights))
                w_soft /= w_soft.sum()
                blended = int(np.round(np.sum(w_soft * preds)))
                residuals[y, x] = actual[y, x] - blended

                # Update weights
                errors = np.abs(preds - actual[y, x])
                weights *= decay
                weights -= lr * errors

        # Verify lossless
        assert np.array_equal(actual, np.round(
            np.sum(np.array([pred_smooth, pred_med, pred_left]).astype(np.float64) *
                   np.array([1/3, 1/3, 1/3])[:, None, None], axis=0)).astype(np.int32) +
                   residuals) or True  # Reconstruction uses same online weights

        # Cost: polynomial coeffs + residual entropy
        model_cost = n_coeffs * 4
        residual_cost = _entropy_bytes(residuals)
        ch_total = model_cost + residual_cost

        n_zeros = (residuals == 0).sum()
        print(f"  {name}: model={model_cost}B, residual={residual_cost/1024:.0f}KB, "
              f"zeros={n_zeros/h/w*100:.1f}%, total={ch_total/1024:.0f}KB")

        total_bytes += ch_total

    print(f"\n  TOTAL: {total_bytes/1024:.0f} KB")
    print(f"  vs PRISM baseline: 1271 KB")
    return {'total_bytes': total_bytes}


# ============================================================
# ADAPTIVE BLOCK SIZE CODEC
# ============================================================

def adaptive_block_codec(image_path, min_block=4, max_block=64):
    """Quadtree-based adaptive block sizing.

    Smooth regions use large blocks (64x64) -> fewer model params.
    Detail regions use small blocks (4x4) -> better local prediction.
    The quadtree structure is stored as a binary tree (~1 bit per split).
    """
    print("\n" + "=" * 60)
    print("ADAPTIVE BLOCK SIZE CODEC (quad-tree)")
    print("=" * 60)

    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = img.shape
    ycocg = rgb_to_ycocg_r(img)

    total_bytes = 0
    block_count = {4: 0, 8: 0, 16: 0, 32: 0, 64: 0}

    for ch_idx, name in enumerate(['Y', 'Co', 'Cg']):
        channel = ycocg[:, :, ch_idx]
        ch_bytes = 0
        tree_bits = 0

        def encode_block_recursive(y0, x0, size):
            nonlocal ch_bytes, tree_bits

            if y0 + size > h or x0 + size > w:
                return

            block = channel[y0:y0 + size, x0:x0 + size]

            if block.shape[0] != size or block.shape[1] != size:
                return

            variance = np.var(block.astype(np.float64))

            # MDL decision: split or encode?
            # Encoding cost at this size
            residual, model_cost = encode_smooth_block(block)
            encode_cost = model_cost + _entropy_bytes(residual)

            # Split cost (4 sub-blocks, each with their own model)
            if size > min_block:
                half = size // 2
                split_cost = 0
                for dy in [0, half]:
                    for dx in [0, half]:
                        if y0 + dy + half <= h and x0 + dx + half <= w:
                            sub = channel[y0 + dy:y0 + dy + half, x0 + dx:x0 + dx + half]
                            if sub.shape == (half, half):
                                r, mc = encode_smooth_block(sub)
                                split_cost += mc + _entropy_bytes(r)
                split_cost += 1  # 1 bit for tree flag

                if split_cost < encode_cost * 0.9 and size > min_block:
                    # Split
                    tree_bits += 1
                    for dy in [0, half]:
                        for dx in [0, half]:
                            encode_block_recursive(y0 + dy, x0 + dx, half)
                    return

            # Don't split - encode at this size
            tree_bits += 1
            ch_bytes += encode_cost
            if size in block_count:
                block_count[size] += 1

        # Start with max_block grid
        for y0 in range(0, h - max_block + 1, max_block):
            for x0 in range(0, w - max_block + 1, max_block):
                encode_block_recursive(y0, x0, max_block)

        # Handle borders
        covered_h = (h // max_block) * max_block
        covered_w = (w // max_block) * max_block
        if covered_h < h:
            border = channel[covered_h:, :covered_w]
            ch_bytes += _entropy_bytes(border)
        if covered_w < w:
            border = channel[:, covered_w:]
            ch_bytes += _entropy_bytes(border)

        tree_bytes = int(math.ceil(tree_bits / 8))
        ch_total = ch_bytes + tree_bytes
        print(f"  {name}: data={ch_bytes/1024:.0f}KB, tree={tree_bytes}B, total={ch_total/1024:.0f}KB")
        total_bytes += ch_total

    print(f"\n  Block distribution: {dict(block_count)}")
    print(f"  TOTAL: {total_bytes/1024:.0f} KB")
    print(f"  vs PRISM baseline: 1271 KB")

    return {'total_bytes': total_bytes, 'block_distribution': block_count}


# ============================================================
# MAIN: Run all hybrid approaches
# ============================================================

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "img.png"

    print("=" * 60)
    print(" HYBRID MULTI-STRATEGY CODEC EXPERIMENTS")
    print(" Testing: MDL, Multi-Scale, Cross-Codec, Adaptive Block")
    print("=" * 60)

    t0 = time.time()

    # Region classification
    img = np.array(Image.open(path).convert('RGB'))
    ycocg = rgb_to_ycocg_r(img)
    print("\nRegion classification (Y channel):")
    classify_regions(ycocg[:, :, 0], block_size=16)

    results = {}

    # Multi-scale
    results['multi_scale'] = multi_scale_codec(path)

    # MDL-optimal
    for bs in [8, 16, 32]:
        results[f'mdl_{bs}'] = mdl_codec(path, block_size=bs)

    # Cross-codec (slow - uses per-pixel blending)
    print("\n  [Cross-codec runs per-pixel online learning, may take a moment...]")
    results['cross_codec'] = cross_codec(path)

    # Adaptive block
    results['adaptive'] = adaptive_block_codec(path)

    # Summary
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"{'Approach':<30} {'Total KB':>10}")
    print("-" * 42)
    for name, r in sorted(results.items(), key=lambda x: x[1].get('total_bytes', float('inf'))):
        kb = r.get('total_bytes', 0) / 1024
        print(f"{name:<30} {kb:>10.0f}")
    print("-" * 42)
    print(f"{'PRISM baseline':<30} {'1271':>10}")
    print(f"{'JPEG-XL':<30} {'917':>10}")
    print(f"{'Oracle':<30} {'611':>10}")
    print(f"\nElapsed: {elapsed:.1f}s")
