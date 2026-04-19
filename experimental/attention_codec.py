#!/usr/bin/env python3
"""SELF-ATTENTION TRANSFORMER PREDICTOR FOR LOSSLESS IMAGE COMPRESSION

Pure numpy implementation -- no PyTorch, no TensorFlow.

Core idea: Instead of fixed spatial predictors (left, above, diagonal),
learn WHICH pixels in the causal context window matter most for each
prediction. The attention weights adapt to image content via online
gradient descent, so encoder and decoder stay synchronized.

Approaches:
  1. Single-head attention -- one set of Q/K/V weights
  2. Multi-head attention -- independent heads capture different patterns
  3. Cross-channel attention -- use luma to predict chroma
  4. Trie-enhanced attention -- blend ContextTrie (sequential) + attention (spatial)

PRISM baseline on img.png: 4.82 bps, ~1271 KB.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prism.transforms import rgb_to_ycocg_r
from experimental.data_structures import ContextTrie, entropy_bits, zigzag_encode


# ============================================================
# Constants
# ============================================================

PRISM_BPS = 4.82
PRISM_KB = 1271
IMAGE_PATH = str(Path(__file__).resolve().parent.parent / "img.png")
SUBSET_ROWS = 200  # process first N rows for speed
EPS = 1e-8


# ============================================================
# Utility: load and transform image
# ============================================================

def load_image_ycocg(image_path, max_rows=None):
    """Load image, apply YCoCg-R transform, optionally truncate rows."""
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    ycocg = rgb_to_ycocg_r(rgb)
    h, w, _ = ycocg.shape
    if max_rows is not None and max_rows < h:
        ycocg = ycocg[:max_rows, :, :]
    return ycocg, (h, w)


def compute_bps_and_kb(residuals, full_h, full_w, n_channels=3):
    """Compute bits-per-pixel from residual entropy, extrapolate to full image."""
    subset_h, subset_w = residuals.shape
    zz = zigzag_encode(residuals.astype(np.int32))
    total_bits = entropy_bits(zz)
    n_pixels_subset = subset_h * subset_w
    bps = total_bits / max(n_pixels_subset, 1)

    full_pixels = full_h * full_w
    estimated_total_bits = bps * full_pixels * n_channels
    estimated_kb = estimated_total_bits / 8 / 1024
    return bps, estimated_kb


def softmax(logits):
    """Numerically stable softmax over the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / (np.sum(exp_vals, axis=-1, keepdims=True) + EPS)


def xavier_init(fan_in, fan_out, rng):
    """Glorot/Xavier uniform initialization."""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


# ============================================================
# Feature extraction for each pixel
# ============================================================

def pixel_features(channel, y, x):
    """Build feature vector for pixel at (y, x).

    Features: [value, grad_h, grad_v, local_variance, pos_x_norm, pos_y_norm]
    All features computed from causal (already-seen) neighbors only.
    """
    h, w = channel.shape
    val = float(channel[y, x]) if y < h and x < w else 128.0
    left = float(channel[y, x - 1]) if x > 0 else 128.0
    above = float(channel[y - 1, x]) if y > 0 else 128.0
    upper_left = float(channel[y - 1, x - 1]) if y > 0 and x > 0 else 128.0

    grad_h = val - left
    grad_v = val - above

    # Local variance from causal neighbors
    neighbors = []
    if x > 0:
        neighbors.append(left)
    if y > 0:
        neighbors.append(above)
    if y > 0 and x > 0:
        neighbors.append(upper_left)
    if y > 0 and x < w - 1:
        neighbors.append(float(channel[y - 1, x + 1]))

    if len(neighbors) >= 2:
        n_arr = np.array(neighbors)
        local_var = float(np.var(n_arr))
    else:
        local_var = 0.0

    pos_x = x / max(w - 1, 1)
    pos_y = y / max(h - 1, 1)

    return np.array([val / 255.0, grad_h / 255.0, grad_v / 255.0,
                     local_var / (255.0 * 255.0), pos_x, pos_y],
                    dtype=np.float64)


def context_features(channel, y, x):
    """Same as pixel_features but using only causal data for a context pixel.

    For context pixels that are already decoded, we can use their actual value.
    """
    return pixel_features(channel, y, x)


# ============================================================
# Gather L-shaped causal context
# ============================================================

def gather_causal_context(channel, y, x, context_size):
    """Gather L-shaped causal context: pixels above and to the left.

    Returns list of (cy, cx) coordinates of context pixels,
    ordered by distance (closest first). At most context_size entries.
    """
    h, w = channel.shape
    candidates = []

    # Row above: full row segment centered on x
    if y > 0:
        for dx in range(-context_size, context_size + 1):
            cx = x + dx
            if 0 <= cx < w:
                dist = abs(dx) + 1
                candidates.append((dist, y - 1, cx))
        # Two rows above for deeper context
        if y > 1:
            for dx in range(-context_size // 2, context_size // 2 + 1):
                cx = x + dx
                if 0 <= cx < w:
                    dist = abs(dx) + 2
                    candidates.append((dist, y - 2, cx))

    # Same row, to the left
    for dx in range(1, context_size + 1):
        cx = x - dx
        if cx >= 0:
            candidates.append((dx, y, cx))

    # Sort by distance, take closest
    candidates.sort(key=lambda t: t[0])
    coords = [(cy, cx) for (_, cy, cx) in candidates[:context_size]]
    return coords


# ============================================================
# APPROACH 1: Single-Head Self-Attention
# ============================================================

def single_head_attention(image_path=IMAGE_PATH, context_size=16):
    """Single-head attention predictor for lossless compression.

    For each pixel:
      1. Gather causal context (L-shaped neighborhood)
      2. Compute Q from current pixel features, K/V from context
      3. Attention = softmax(Q @ K^T / sqrt(d_k))
      4. Prediction = Attention @ V
      5. Online gradient descent on W_q, W_k, W_v
    """
    print("\n" + "=" * 70)
    print("APPROACH 1: Single-Head Self-Attention Predictor")
    print("=" * 70)

    ycocg, (full_h, full_w) = load_image_ycocg(image_path, max_rows=SUBSET_ROWS)
    channel = ycocg[:, :, 0].astype(np.int32)  # Y channel
    h, w = channel.shape
    print(f"  Processing: {h}x{w} Y channel (subset of {full_h}x{full_w})")

    d_feat = 6     # feature dimension
    d_k = 8        # key/query projection dimension
    d_v = 1        # value dimension (predicting scalar pixel value)
    lr = 0.01
    lr_decay = 0.9999

    rng = np.random.RandomState(42)
    W_q = xavier_init(d_feat, d_k, rng)
    W_k = xavier_init(d_feat, d_k, rng)
    W_v = xavier_init(d_feat, d_v, rng)

    residuals = np.zeros((h, w), dtype=np.int32)
    recon = np.full((h, w), 128, dtype=np.int32)

    total_sq_err = 0.0
    n_processed = 0
    attn_entropy_sum = 0.0
    convergence_log = []
    t0 = time.time()

    for y in range(h):
        for x in range(w):
            actual = int(channel[y, x])

            # Gather causal context
            ctx_coords = gather_causal_context(recon, y, x, context_size)

            if len(ctx_coords) < 2:
                # Not enough context -- fallback to simple prediction
                left = int(recon[y, x - 1]) if x > 0 else 128
                above = int(recon[y - 1, x]) if y > 0 else 128
                pred = (left + above) // 2
            else:
                # Build feature vectors
                q_feat = pixel_features(recon, y, x)
                k_feats = np.array([context_features(recon, cy, cx)
                                    for (cy, cx) in ctx_coords])
                v_vals = np.array([[float(recon[cy, cx]) / 255.0]
                                   for (cy, cx) in ctx_coords])

                # Project: Q = q_feat @ W_q, K = k_feats @ W_k
                Q = q_feat @ W_q                       # (d_k,)
                K = k_feats @ W_k                       # (n_ctx, d_k)
                V = v_vals                              # (n_ctx, d_v)

                # Scaled dot-product attention
                scores = K @ Q / math.sqrt(d_k)        # (n_ctx,)
                attn_weights = softmax(scores)          # (n_ctx,)

                # Track attention entropy (how concentrated the attention is)
                aw_clipped = np.clip(attn_weights, EPS, 1.0)
                attn_ent = -np.sum(aw_clipped * np.log2(aw_clipped))
                attn_entropy_sum += attn_ent

                # Prediction
                pred_normed = float(attn_weights @ V)   # scalar in [0, 1]
                pred = int(np.clip(round(pred_normed * 255.0), 0, 511))

                # --- Online gradient descent ---
                error = (actual / 255.0) - pred_normed
                # Gradient through attention softmax is complex;
                # use simplified gradient: dL/dW_q via chain rule
                # dL/d(scores) = error * (V - pred_normed) * attn_weights
                d_scores = error * attn_weights * (V.flatten() - pred_normed)
                # dL/dW_q = q_feat^T @ d_scores @ K / sqrt(d_k) (outer product form)
                d_Q = (d_scores @ K) / math.sqrt(d_k)    # (d_k,)
                dW_q = np.outer(q_feat, d_Q)              # (d_feat, d_k)

                d_K = np.outer(d_scores, Q) / math.sqrt(d_k)  # (n_ctx, d_k)
                dW_k = k_feats.T @ d_K                         # (d_feat, d_k)

                d_V = (attn_weights * error).reshape(-1, 1)    # (n_ctx, 1)
                # For V = v_vals (raw values), we update W_v projection
                # But v_vals are fixed pixel values; the gradient goes to W_v
                # V_proj = k_feats @ W_v, but we use raw values for V
                # Skip W_v update since we use raw pixel values as V

                W_q = W_q + lr * dW_q
                W_k = W_k + lr * dW_k
                lr *= lr_decay

            residual = actual - pred
            residuals[y, x] = residual
            recon[y, x] = actual

            total_sq_err += residual ** 2
            n_processed += 1

        # Log convergence every 20 rows
        if (y + 1) % 20 == 0:
            rmse = math.sqrt(total_sq_err / n_processed)
            avg_attn_ent = attn_entropy_sum / max(n_processed, 1)
            convergence_log.append((y + 1, rmse, avg_attn_ent))
            elapsed = time.time() - t0
            rows_per_sec = (y + 1) / elapsed
            print(f"    Row {y+1:4d}/{h}: RMSE={rmse:.2f}, "
                  f"attn_entropy={avg_attn_ent:.2f} bits, "
                  f"{rows_per_sec:.1f} rows/s")

    bps, est_kb = compute_bps_and_kb(residuals, full_h, full_w)
    elapsed = time.time() - t0

    print(f"\n  --- Single-Head Attention Results ---")
    print(f"  Residual bps:      {bps:.3f}")
    print(f"  PRISM baseline:    {PRISM_BPS:.3f} bps")
    print(f"  Delta:             {bps - PRISM_BPS:+.3f} bps")
    print(f"  Est. full image:   {est_kb:.0f} KB (PRISM: {PRISM_KB} KB)")
    print(f"  Final RMSE:        {math.sqrt(total_sq_err / n_processed):.2f}")
    print(f"  Time:              {elapsed:.1f}s")

    # Convergence analysis
    if convergence_log:
        print(f"\n  Convergence (RMSE over rows):")
        for row, rmse, aent in convergence_log[:5]:
            print(f"    Row {row:4d}: RMSE={rmse:.2f}, attn_entropy={aent:.2f}")
        if len(convergence_log) > 5:
            row, rmse, aent = convergence_log[-1]
            print(f"    ...  {row:4d}: RMSE={rmse:.2f}, attn_entropy={aent:.2f}")

    return {"bps": bps, "est_kb": est_kb, "method": "single_head_attention"}


# ============================================================
# APPROACH 2: Multi-Head Self-Attention
# ============================================================

def multi_head_attention(image_path=IMAGE_PATH, n_heads=4, context_size=16):
    """Multi-head attention predictor.

    Each head independently learns different attention patterns:
      - Head 0 might focus on horizontal neighbors (texture)
      - Head 1 might focus on vertical neighbors (edges)
      - Head 2 might focus on diagonal (gradient direction)
      - Head 3 might focus on distant context (long-range correlation)

    Outputs are concatenated and projected through W_o.
    """
    print("\n" + "=" * 70)
    print(f"APPROACH 2: Multi-Head Attention (n_heads={n_heads})")
    print("=" * 70)

    ycocg, (full_h, full_w) = load_image_ycocg(image_path, max_rows=SUBSET_ROWS)
    channel = ycocg[:, :, 0].astype(np.int32)
    h, w = channel.shape
    print(f"  Processing: {h}x{w} Y channel (subset of {full_h}x{full_w})")

    d_feat = 6
    d_k = 8       # per head
    d_head_v = 4   # value dim per head
    d_concat = n_heads * d_head_v
    lr = 0.008
    lr_decay = 0.9999

    rng = np.random.RandomState(123)

    # Per-head weight matrices
    W_q_heads = [xavier_init(d_feat, d_k, rng) for _ in range(n_heads)]
    W_k_heads = [xavier_init(d_feat, d_k, rng) for _ in range(n_heads)]
    W_v_heads = [xavier_init(d_feat, d_head_v, rng) for _ in range(n_heads)]

    # Output projection
    W_o = xavier_init(d_concat, 1, rng)

    residuals = np.zeros((h, w), dtype=np.int32)
    recon = np.full((h, w), 128, dtype=np.int32)

    total_sq_err = 0.0
    n_processed = 0
    head_attn_focus = [[] for _ in range(n_heads)]  # track top attended position
    t0 = time.time()

    for y in range(h):
        for x in range(w):
            actual = int(channel[y, x])
            ctx_coords = gather_causal_context(recon, y, x, context_size)

            if len(ctx_coords) < 2:
                left = int(recon[y, x - 1]) if x > 0 else 128
                above = int(recon[y - 1, x]) if y > 0 else 128
                pred = (left + above) // 2
            else:
                q_feat = pixel_features(recon, y, x)
                k_feats = np.array([context_features(recon, cy, cx)
                                    for (cy, cx) in ctx_coords])
                n_ctx = len(ctx_coords)

                head_outputs = []
                head_attn_all = []

                for head_i in range(n_heads):
                    Q = q_feat @ W_q_heads[head_i]            # (d_k,)
                    K = k_feats @ W_k_heads[head_i]            # (n_ctx, d_k)
                    V = k_feats @ W_v_heads[head_i]            # (n_ctx, d_head_v)

                    scores = K @ Q / math.sqrt(d_k)
                    attn = softmax(scores)                     # (n_ctx,)
                    head_out = attn @ V                        # (d_head_v,)
                    head_outputs.append(head_out)
                    head_attn_all.append(attn)

                    # Track which context position each head attends to most
                    if n_processed < 5000:
                        top_idx = int(np.argmax(attn))
                        head_attn_focus[head_i].append(top_idx)

                # Concatenate and project
                concat = np.concatenate(head_outputs)          # (d_concat,)
                pred_raw = float(concat @ W_o)                 # scalar
                pred = int(np.clip(round(pred_raw * 255.0), 0, 511))

                # --- Online gradient descent ---
                error = (actual / 255.0) - pred_raw
                # Gradient through W_o
                d_concat = error * W_o.flatten()               # (d_concat,)

                # Gradient per head
                for head_i in range(n_heads):
                    d_head = d_concat[head_i * d_head_v:(head_i + 1) * d_head_v]
                    attn = head_attn_all[head_i]
                    V = k_feats @ W_v_heads[head_i]

                    # dL/dW_v: k_feats^T @ diag(attn) @ d_head
                    d_V = np.outer(attn, d_head)               # (n_ctx, d_head_v)
                    dW_v = k_feats.T @ d_V                     # (d_feat, d_head_v)
                    W_v_heads[head_i] = W_v_heads[head_i] + lr * dW_v

                    # Simplified Q/K updates
                    Q = q_feat @ W_q_heads[head_i]
                    K = k_feats @ W_k_heads[head_i]
                    # d_scores approx
                    head_pred = float(attn @ V @ d_head) if d_head.size > 0 else 0.0
                    d_scores = error * attn * 0.1              # damped for stability
                    d_Q = (d_scores @ K) / math.sqrt(d_k)
                    dW_q = np.outer(q_feat, d_Q)
                    W_q_heads[head_i] = W_q_heads[head_i] + lr * dW_q

                    d_K = np.outer(d_scores, Q) / math.sqrt(d_k)
                    dW_k = k_feats.T @ d_K
                    W_k_heads[head_i] = W_k_heads[head_i] + lr * dW_k

                # Update W_o
                dW_o = (error * concat).reshape(-1, 1)
                W_o = W_o + lr * dW_o
                lr *= lr_decay

            residual = actual - pred
            residuals[y, x] = residual
            recon[y, x] = actual

            total_sq_err += residual ** 2
            n_processed += 1

        if (y + 1) % 40 == 0:
            rmse = math.sqrt(total_sq_err / n_processed)
            elapsed = time.time() - t0
            print(f"    Row {y+1:4d}/{h}: RMSE={rmse:.2f}, "
                  f"{(y+1)/elapsed:.1f} rows/s")

    bps, est_kb = compute_bps_and_kb(residuals, full_h, full_w)
    elapsed = time.time() - t0

    print(f"\n  --- Multi-Head Attention Results ---")
    print(f"  Residual bps:      {bps:.3f}")
    print(f"  PRISM baseline:    {PRISM_BPS:.3f} bps")
    print(f"  Delta:             {bps - PRISM_BPS:+.3f} bps")
    print(f"  Est. full image:   {est_kb:.0f} KB (PRISM: {PRISM_KB} KB)")
    print(f"  Final RMSE:        {math.sqrt(total_sq_err / n_processed):.2f}")
    print(f"  Time:              {elapsed:.1f}s")

    # Head specialization analysis
    print(f"\n  Head Specialization (top attended context position):")
    for head_i in range(n_heads):
        if head_attn_focus[head_i]:
            focus = np.array(head_attn_focus[head_i])
            unique, counts = np.unique(focus, return_counts=True)
            top3 = unique[np.argsort(-counts)[:3]]
            top3_pct = counts[np.argsort(-counts)[:3]] / len(focus) * 100
            desc_parts = [f"pos {p}({pct:.0f}%)" for p, pct in zip(top3, top3_pct)]
            print(f"    Head {head_i}: top positions = {', '.join(desc_parts)}")
            # Interpret position meaning
            median_pos = float(np.median(focus))
            print(f"             median focus position = {median_pos:.1f} "
                  f"({'near' if median_pos < 4 else 'mid' if median_pos < 10 else 'far'} context)")

    return {"bps": bps, "est_kb": est_kb, "method": "multi_head_attention"}


# ============================================================
# APPROACH 3: Cross-Channel Attention
# ============================================================

def cross_channel_attention(image_path=IMAGE_PATH):
    """Cross-channel attention: use decoded Y to predict Co and Cg.

    Cross-attention mechanism:
      Q = chroma_features @ W_q  (what the chroma pixel needs)
      K = luma_features @ W_k    (what the luma pixels offer)
      V = luma_values @ W_v      (luma information to transfer)

    Exploits strong luma-chroma correlation in natural images.
    """
    print("\n" + "=" * 70)
    print("APPROACH 3: Cross-Channel Attention (Y -> Co, Cg)")
    print("=" * 70)

    ycocg, (full_h, full_w) = load_image_ycocg(image_path, max_rows=SUBSET_ROWS)
    y_ch = ycocg[:, :, 0].astype(np.int32)
    co_ch = ycocg[:, :, 1].astype(np.int32)
    cg_ch = ycocg[:, :, 2].astype(np.int32)
    h, w = y_ch.shape
    print(f"  Processing: {h}x{w} Co+Cg channels (subset of {full_h}x{full_w})")

    d_feat = 6
    d_k = 8
    context_size = 12
    lr = 0.01
    lr_decay = 0.9999

    results_per_channel = {}

    for ch_name, target_ch in [("Co", co_ch), ("Cg", cg_ch)]:
        print(f"\n  --- Channel: {ch_name} ---")
        rng = np.random.RandomState(77)
        W_q = xavier_init(d_feat, d_k, rng)
        W_k = xavier_init(d_feat, d_k, rng)
        # Bias term for the prediction
        bias = np.zeros(1, dtype=np.float64)
        current_lr = lr

        residuals = np.zeros((h, w), dtype=np.int32)
        recon_target = np.full((h, w), 0, dtype=np.int32)

        total_sq_err = 0.0
        n_processed = 0
        t0 = time.time()

        for yi in range(h):
            for xi in range(w):
                actual = int(target_ch[yi, xi])
                ctx_coords = gather_causal_context(y_ch, yi, xi, context_size)

                if len(ctx_coords) < 2:
                    # Simple prediction from co-located luma
                    pred = 0  # chroma centered around 0
                else:
                    # Q from chroma context (what do we need?)
                    q_feat = pixel_features(recon_target, yi, xi)
                    # K from luma context (what does luma offer?)
                    k_feats = np.array([pixel_features(y_ch, cy, cx)
                                        for (cy, cx) in ctx_coords])
                    # V from already-decoded chroma at context positions
                    v_vals = np.array([float(recon_target[cy, cx])
                                       for (cy, cx) in ctx_coords])

                    Q = q_feat @ W_q                           # (d_k,)
                    K = k_feats @ W_k                          # (n_ctx, d_k)

                    scores = K @ Q / math.sqrt(d_k)
                    attn = softmax(scores)

                    # Weighted sum of chroma values at context positions
                    pred_raw = float(attn @ v_vals) + float(bias[0])
                    pred = int(np.clip(round(pred_raw), -255, 511))

                    # Online update
                    error = actual - pred_raw
                    d_scores = error * attn * 0.1
                    d_Q = (d_scores @ K) / math.sqrt(d_k)
                    dW_q = np.outer(q_feat, d_Q)
                    d_K = np.outer(d_scores, Q) / math.sqrt(d_k)
                    dW_k = k_feats.T @ d_K

                    W_q = W_q + current_lr * dW_q
                    W_k = W_k + current_lr * dW_k
                    bias = bias + np.array([current_lr * error * 0.01])
                    current_lr *= lr_decay

                residual = actual - pred
                residuals[yi, xi] = residual
                recon_target[yi, xi] = actual

                total_sq_err += residual ** 2
                n_processed += 1

            if (yi + 1) % 50 == 0:
                rmse = math.sqrt(total_sq_err / n_processed)
                print(f"    Row {yi+1:4d}/{h}: RMSE={rmse:.2f}")

        bps_ch, _ = compute_bps_and_kb(residuals, full_h, full_w, n_channels=1)
        rmse = math.sqrt(total_sq_err / n_processed)
        print(f"    {ch_name} bps: {bps_ch:.3f}, RMSE: {rmse:.2f}")
        results_per_channel[ch_name] = bps_ch

    # Also measure Y channel with simple spatial attention for fair comparison
    print(f"\n  --- Channel: Y (spatial attention, for total estimate) ---")
    y_result = single_head_attention_channel(y_ch, context_size=12)
    results_per_channel["Y"] = y_result["bps"]

    # Total estimate
    total_bps = sum(results_per_channel.values()) / 3.0
    # Weighted: Y has more pixels-worth of entropy typically
    weighted_bps = (results_per_channel["Y"] * 1.0 +
                    results_per_channel["Co"] * 1.0 +
                    results_per_channel["Cg"] * 1.0) / 3.0
    full_pixels = full_h * full_w
    est_kb = weighted_bps * full_pixels * 3 / 8 / 1024

    elapsed = time.time() - t0
    print(f"\n  --- Cross-Channel Attention Results ---")
    print(f"  Y  bps: {results_per_channel['Y']:.3f}")
    print(f"  Co bps: {results_per_channel['Co']:.3f}")
    print(f"  Cg bps: {results_per_channel['Cg']:.3f}")
    print(f"  Avg bps:           {weighted_bps:.3f}")
    print(f"  PRISM baseline:    {PRISM_BPS:.3f} bps")
    print(f"  Delta:             {weighted_bps - PRISM_BPS:+.3f} bps")
    print(f"  Est. full image:   {est_kb:.0f} KB (PRISM: {PRISM_KB} KB)")

    return {"bps": weighted_bps, "est_kb": est_kb, "method": "cross_channel_attention",
            "per_channel": results_per_channel}


def single_head_attention_channel(channel, context_size=12):
    """Run single-head attention on a single channel (helper for cross-channel)."""
    h, w = channel.shape
    d_feat = 6
    d_k = 8
    lr = 0.01
    lr_decay = 0.9999

    rng = np.random.RandomState(42)
    W_q = xavier_init(d_feat, d_k, rng)
    W_k = xavier_init(d_feat, d_k, rng)

    residuals = np.zeros((h, w), dtype=np.int32)
    recon = np.full((h, w), 128, dtype=np.int32)

    total_sq_err = 0.0
    n_processed = 0

    for y in range(h):
        for x in range(w):
            actual = int(channel[y, x])
            ctx_coords = gather_causal_context(recon, y, x, context_size)

            if len(ctx_coords) < 2:
                left = int(recon[y, x - 1]) if x > 0 else 128
                above = int(recon[y - 1, x]) if y > 0 else 128
                pred = (left + above) // 2
            else:
                q_feat = pixel_features(recon, y, x)
                k_feats = np.array([context_features(recon, cy, cx)
                                    for (cy, cx) in ctx_coords])
                v_vals = np.array([float(recon[cy, cx])
                                   for (cy, cx) in ctx_coords])

                Q = q_feat @ W_q
                K = k_feats @ W_k
                scores = K @ Q / math.sqrt(d_k)
                attn = softmax(scores)

                pred_raw = float(attn @ v_vals)
                pred = int(np.clip(round(pred_raw), 0, 511))

                error = actual - pred_raw
                d_scores = error * attn * 0.1
                d_Q = (d_scores @ K) / math.sqrt(d_k)
                dW_q = np.outer(q_feat, d_Q)
                d_K = np.outer(d_scores, Q) / math.sqrt(d_k)
                dW_k = k_feats.T @ d_K

                W_q = W_q + lr * dW_q
                W_k = W_k + lr * dW_k
                lr *= lr_decay

            residual = actual - pred
            residuals[y, x] = residual
            recon[y, x] = actual

            total_sq_err += residual ** 2
            n_processed += 1

        if (y + 1) % 50 == 0:
            rmse = math.sqrt(total_sq_err / n_processed)
            print(f"    Row {y+1:4d}/{h}: RMSE={rmse:.2f}")

    zz = zigzag_encode(residuals.astype(np.int32))
    total_bits = entropy_bits(zz)
    bps = total_bits / max(n_processed, 1)
    print(f"    Y bps: {bps:.3f}")
    return {"bps": bps}


# ============================================================
# APPROACH 4: Trie-Enhanced Attention
# ============================================================

def trie_enhanced_attention(image_path=IMAGE_PATH):
    """Blend ContextTrie (sequential context) with attention (spatial context).

    The trie captures sequential patterns in raster-scan order:
      "after seeing values [a, b, c], the next value is likely d"

    Attention captures spatial patterns:
      "this pixel is most similar to the one 2 pixels above"

    A learned gate blends them:
      prediction = gate * trie_pred + (1 - gate) * attn_pred

    The gate adapts online, learning when sequential vs. spatial context
    is more informative (e.g., smooth gradients favor trie, edges favor attention).
    """
    print("\n" + "=" * 70)
    print("APPROACH 4: Trie-Enhanced Attention (Sequential + Spatial)")
    print("=" * 70)

    ycocg, (full_h, full_w) = load_image_ycocg(image_path, max_rows=SUBSET_ROWS)
    channel = ycocg[:, :, 0].astype(np.int32)
    h, w = channel.shape
    print(f"  Processing: {h}x{w} Y channel (subset of {full_h}x{full_w})")

    # Trie config
    trie_depth = 6
    n_quant = 64  # quantize pixel values to reduce trie alphabet

    # Attention config
    d_feat = 6
    d_k = 8
    context_size = 12
    lr = 0.01
    lr_decay = 0.9999

    rng = np.random.RandomState(99)
    W_q = xavier_init(d_feat, d_k, rng)
    W_k = xavier_init(d_feat, d_k, rng)

    trie = ContextTrie(max_depth=trie_depth)

    # Gate parameter (logit space): sigmoid(gate_logit) = blend weight for trie
    gate_logit = 0.0  # start at 0.5 blend
    gate_lr = 0.005

    residuals = np.zeros((h, w), dtype=np.int32)
    recon = np.full((h, w), 128, dtype=np.int32)

    total_sq_err = 0.0
    n_processed = 0
    gate_history = []
    trie_better_count = 0
    attn_better_count = 0
    t0 = time.time()

    # Sequential buffer for trie context
    seq_buffer = []

    for y in range(h):
        for x in range(w):
            actual = int(channel[y, x])
            actual_q = min(actual * n_quant // 256, n_quant - 1)

            # --- Trie prediction ---
            trie_context = seq_buffer[-trie_depth:]
            trie_probs = trie.predict(trie_context, n_symbols=n_quant)
            # Expected value from trie distribution
            trie_pred_q = float(np.dot(np.arange(n_quant), trie_probs))
            trie_pred = trie_pred_q * 256.0 / n_quant

            # --- Attention prediction ---
            ctx_coords = gather_causal_context(recon, y, x, context_size)

            if len(ctx_coords) < 2:
                left = int(recon[y, x - 1]) if x > 0 else 128
                above = int(recon[y - 1, x]) if y > 0 else 128
                attn_pred = float((left + above) // 2)
            else:
                q_feat = pixel_features(recon, y, x)
                k_feats = np.array([context_features(recon, cy, cx)
                                    for (cy, cx) in ctx_coords])
                v_vals = np.array([float(recon[cy, cx])
                                   for (cy, cx) in ctx_coords])

                Q = q_feat @ W_q
                K = k_feats @ W_k
                scores = K @ Q / math.sqrt(d_k)
                attn = softmax(scores)
                attn_pred = float(attn @ v_vals)

                # Update attention weights
                error_attn = actual - attn_pred
                d_scores = error_attn * attn * 0.1
                d_Q = (d_scores @ K) / math.sqrt(d_k)
                dW_q = np.outer(q_feat, d_Q)
                d_K = np.outer(d_scores, Q) / math.sqrt(d_k)
                dW_k = k_feats.T @ d_K

                W_q = W_q + lr * dW_q
                W_k = W_k + lr * dW_k
                lr *= lr_decay

            # --- Blend with learned gate ---
            gate = 1.0 / (1.0 + math.exp(-gate_logit))  # sigmoid
            blended_pred = gate * trie_pred + (1.0 - gate) * attn_pred
            pred = int(np.clip(round(blended_pred), 0, 511))

            residual = actual - pred
            residuals[y, x] = residual
            recon[y, x] = actual

            # --- Update gate ---
            trie_err = abs(actual - trie_pred)
            attn_err = abs(actual - attn_pred)
            # Move gate toward whichever predictor had lower error
            # d(gate_logit) = lr * (attn_err - trie_err) * gate * (1 - gate)
            gate_grad = (attn_err - trie_err) * gate * (1.0 - gate)
            gate_logit = gate_logit + gate_lr * gate_grad

            if trie_err < attn_err:
                trie_better_count += 1
            else:
                attn_better_count += 1

            # --- Update trie ---
            trie.update(trie_context, actual_q)
            seq_buffer.append(actual_q)

            total_sq_err += residual ** 2
            n_processed += 1

            if n_processed % 10000 == 0:
                gate_history.append(gate)

        if (y + 1) % 40 == 0:
            gate = 1.0 / (1.0 + math.exp(-gate_logit))
            rmse = math.sqrt(total_sq_err / n_processed)
            elapsed = time.time() - t0
            print(f"    Row {y+1:4d}/{h}: RMSE={rmse:.2f}, "
                  f"gate={gate:.3f} (trie weight), "
                  f"{(y+1)/elapsed:.1f} rows/s")

    bps, est_kb = compute_bps_and_kb(residuals, full_h, full_w)
    elapsed = time.time() - t0

    final_gate = 1.0 / (1.0 + math.exp(-gate_logit))
    total_comparisons = trie_better_count + attn_better_count

    print(f"\n  --- Trie-Enhanced Attention Results ---")
    print(f"  Residual bps:      {bps:.3f}")
    print(f"  PRISM baseline:    {PRISM_BPS:.3f} bps")
    print(f"  Delta:             {bps - PRISM_BPS:+.3f} bps")
    print(f"  Est. full image:   {est_kb:.0f} KB (PRISM: {PRISM_KB} KB)")
    print(f"  Final RMSE:        {math.sqrt(total_sq_err / n_processed):.2f}")
    print(f"  Time:              {elapsed:.1f}s")
    print(f"\n  Gate Analysis:")
    print(f"    Final gate value:  {final_gate:.3f} "
          f"({'trie-dominant' if final_gate > 0.6 else 'attention-dominant' if final_gate < 0.4 else 'balanced'})")
    print(f"    Trie better:       {trie_better_count}/{total_comparisons} "
          f"({100*trie_better_count/max(total_comparisons,1):.1f}%)")
    print(f"    Attention better:  {attn_better_count}/{total_comparisons} "
          f"({100*attn_better_count/max(total_comparisons,1):.1f}%)")

    if gate_history:
        print(f"    Gate evolution:    {gate_history[0]:.3f} -> {gate_history[-1]:.3f}")

    return {"bps": bps, "est_kb": est_kb, "method": "trie_enhanced_attention",
            "gate": final_gate}


# ============================================================
# Comparison Table
# ============================================================

def print_comparison_table(results):
    """Print formatted comparison table of all approaches."""
    print("\n" + "=" * 70)
    print("COMPARISON TABLE: Attention-Based Prediction vs PRISM Baseline")
    print("=" * 70)

    header = f"  {'Method':<30s} {'BPS':>8s} {'Est KB':>8s} {'vs PRISM':>10s}"
    print(header)
    print("  " + "-" * 58)

    # Baseline
    print(f"  {'PRISM (baseline)':<30s} {PRISM_BPS:>8.3f} {PRISM_KB:>8.0f} {'---':>10s}")

    for r in results:
        delta = r["bps"] - PRISM_BPS
        delta_str = f"{delta:+.3f}"
        marker = " **" if delta < 0 else ""
        print(f"  {r['method']:<30s} {r['bps']:>8.3f} {r['est_kb']:>8.0f} {delta_str:>10s}{marker}")

    print()

    # Find best
    best = min(results, key=lambda r: r["bps"])
    if best["bps"] < PRISM_BPS:
        improvement = (PRISM_BPS - best["bps"]) / PRISM_BPS * 100
        print(f"  BEST: {best['method']} ({improvement:.1f}% better than PRISM)")
    else:
        gap = (best["bps"] - PRISM_BPS) / PRISM_BPS * 100
        print(f"  BEST: {best['method']} ({gap:.1f}% worse than PRISM)")
        print(f"  Note: PRISM uses 7 hand-tuned predictors + neural mixer + "
              f"cross-channel.")
        print(f"  Attention must learn everything from scratch via online SGD.")
        print(f"  This is a proof-of-concept -- a larger context window,")
        print(f"  more heads, or pre-training on image patches could close the gap.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SELF-ATTENTION TRANSFORMER PREDICTOR FOR LOSSLESS COMPRESSION")
    print("Pure numpy -- no PyTorch, no TensorFlow")
    print(f"Image: {IMAGE_PATH}")
    print(f"Subset: first {SUBSET_ROWS} rows for speed")
    print(f"PRISM baseline: {PRISM_BPS} bps, ~{PRISM_KB} KB")
    print("=" * 70)

    results = []

    # Run all approaches
    r1 = single_head_attention()
    results.append(r1)

    r2 = multi_head_attention(n_heads=4)
    results.append(r2)

    r3 = cross_channel_attention()
    results.append(r3)

    r4 = trie_enhanced_attention()
    results.append(r4)

    # Final comparison
    print_comparison_table(results)
