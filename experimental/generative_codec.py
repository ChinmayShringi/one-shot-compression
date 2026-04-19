#!/usr/bin/env python3
"""GENERATIVE CODEC: Image as a program that generates itself.

PARADIGM SHIFT:
  Instead of compressing pixels, find a PROGRAM that GENERATES the image.
  The compressed file IS the program. The decoder RUNS the program.
  Store: program_parameters + lossless_corrections.

Five approaches, each attacking Kolmogorov complexity from a different angle:

1. SIREN Codec       - Coordinate network with sinusoidal activations
2. Fourier Neural    - Learnable Fourier features + linear map
3. Procedural Codec  - Composable rendering pipeline (Perlin, Voronoi, etc.)
4. Evolutionary Codec- Genetic programming to find compact image programs
5. Hybrid Generative - Region-adaptive combination of all the above

All implementations are pure numpy. No ML frameworks.
Backpropagation, CMA-ES, and genetic programming -- all from scratch.
"""

import math
import time
from collections import Counter

import numpy as np
from PIL import Image

from prism.transforms import rgb_to_ycocg_r
from experimental.data_structures import entropy_bits, zigzag_encode


# ============================================================
# Shared utilities
# ============================================================

PRISM_BASELINE_KB = 1271


def _load_image_ycocg(image_path):
    """Load image, convert to YCoCg-R, return (img_rgb, ycocg, H, W)."""
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w, _ = img.shape
    ycocg = rgb_to_ycocg_r(img)
    return img, ycocg, h, w


def _compute_residual_stats(original_ch, predicted_ch, channel_name=""):
    """Compute integer residual and return (residual, entropy_kb, zero_pct).

    original_ch and predicted_ch are 2D int32 arrays.
    """
    residual = original_ch.astype(np.int32) - predicted_ch.astype(np.int32)
    total_pixels = residual.size
    zz = zigzag_encode(residual)
    e_bits = entropy_bits(zz)
    e_kb = e_bits / 8 / 1024
    zero_pct = 100.0 * np.sum(residual == 0) / total_pixels
    return residual, e_kb, zero_pct


def _verify_lossless(original_ch, predicted_ch, residual):
    """Verify that original == predicted + residual."""
    reconstructed = predicted_ch.astype(np.int32) + residual.astype(np.int32)
    return np.array_equal(original_ch.astype(np.int32), reconstructed)


def _print_codec_result(name, param_kb, residual_kb, zero_pct_avg):
    """Print standardized codec result summary."""
    total_kb = param_kb + residual_kb
    ratio_vs_prism = total_kb / PRISM_BASELINE_KB
    print(f"\n  === {name} Result ===")
    print(f"    Model parameters : {param_kb:.1f} KB")
    print(f"    Residual entropy : {residual_kb:.1f} KB")
    print(f"    TOTAL            : {total_kb:.1f} KB")
    print(f"    Zero-residual px : {zero_pct_avg:.1f}%")
    print(f"    vs PRISM (1271KB): {ratio_vs_prism:.2f}x")
    return total_kb


def _make_coord_grid(h, w, subsample=1):
    """Create normalized (y, x) coordinate grid.

    Returns (N, 2) array with values in [-1, 1].
    If subsample > 1, returns every subsample-th pixel.
    """
    ys = np.linspace(-1, 1, h)
    xs = np.linspace(-1, 1, w)
    if subsample > 1:
        ys = ys[::subsample]
        xs = xs[::subsample]
    yg, xg = np.meshgrid(ys, xs, indexing="ij")
    coords = np.stack([yg.ravel(), xg.ravel()], axis=-1)
    return coords, yg.shape[0], yg.shape[1]


# ============================================================
# 1. SIREN CODEC
# ============================================================

def _siren_init_weights(n_in, n_out, is_first_layer, omega_0, rng):
    """SIREN weight initialization (Sitzmann et al. 2020).

    First layer:  U(-1/n_in, 1/n_in)
    Other layers: U(-sqrt(6/n_in)/omega, sqrt(6/n_in)/omega)
    """
    if is_first_layer:
        bound = 1.0 / n_in
    else:
        bound = math.sqrt(6.0 / n_in) / omega_0
    W = rng.uniform(-bound, bound, (n_in, n_out)).astype(np.float64)
    b = rng.uniform(-bound, bound, n_out).astype(np.float64)
    return W, b


def _siren_forward(x, weights, biases, omega_0=30.0):
    """Forward pass through SIREN network.

    All hidden layers use sin(omega_0 * (Wx + b)).
    Last layer is linear.
    """
    h = x.copy()
    n_layers = len(weights)
    for i in range(n_layers - 1):
        h = np.sin(omega_0 * (h @ weights[i] + biases[i]))
    # Final layer: linear (no activation)
    h = h @ weights[-1] + biases[-1]
    return h


def _siren_backward(x, weights, biases, target, omega_0=30.0):
    """Manual backprop through SIREN. Returns gradients for all weights/biases.

    Chain rule through sin activations:
      d/dz sin(omega*z) = omega * cos(omega*z)
    """
    n_layers = len(weights)

    # Forward pass, saving activations
    pre_acts = []   # before activation
    post_acts = []  # after activation
    h = x.copy()
    for i in range(n_layers - 1):
        z = h @ weights[i] + biases[i]
        pre_acts.append(z)
        h = np.sin(omega_0 * z)
        post_acts.append(h)

    # Final layer (linear)
    z_out = h @ weights[-1] + biases[-1]
    output = z_out

    # Loss gradient (MSE)
    n = x.shape[0]
    d_output = 2.0 * (output - target) / n  # (N, out_dim)

    # Backprop
    grad_w = [None] * n_layers
    grad_b = [None] * n_layers

    # Last layer
    grad_w[-1] = h.T @ d_output
    grad_b[-1] = d_output.sum(axis=0)
    d_h = d_output @ weights[-1].T

    # Hidden layers (reverse order)
    for i in range(n_layers - 2, -1, -1):
        # d_h -> through sin activation
        d_pre = d_h * omega_0 * np.cos(omega_0 * pre_acts[i])
        inp = x if i == 0 else post_acts[i - 1]
        grad_w[i] = inp.T @ d_pre
        grad_b[i] = d_pre.sum(axis=0)
        if i > 0:
            d_h = d_pre @ weights[i].T

    return grad_w, grad_b


def siren_codec(image_path, hidden_sizes=None, n_freqs=12):
    """SIREN coordinate network codec.

    Learns f(x, y) -> (Y, Co, Cg) using sinusoidal activations.
    Random hidden layers (seeded, reproducible) reduce stored parameters:
    only the output layer weights are stored.

    Parameters
    ----------
    image_path : str
        Path to image file.
    hidden_sizes : list of int
        Hidden layer sizes. Default [256, 128].
    n_freqs : int
        Number of Fourier features to prepend to coordinates.
    """
    if hidden_sizes is None:
        hidden_sizes = [256, 128]

    print("\n" + "=" * 60)
    print("SIREN CODEC: Coordinate Network as Compressed Representation")
    print("=" * 60)

    _, ycocg, h, w = _load_image_ycocg(image_path)
    total_pixels = h * w
    omega_0 = 30.0
    subsample = 4
    lr = 1e-4
    n_epochs = 200

    print(f"  Image: {w}x{h}, {total_pixels:,} pixels")
    print(f"  Hidden layers: {hidden_sizes}, omega_0={omega_0}")
    print(f"  Training subsample: 1/{subsample} pixels, {n_epochs} epochs")

    # Build coordinate features with Fourier encoding
    coords_full, h_full, w_full = _make_coord_grid(h, w, subsample=1)
    coords_sub, h_sub, w_sub = _make_coord_grid(h, w, subsample=subsample)

    # Add Fourier features: [sin(pi*k*y), cos(pi*k*y), sin(pi*k*x), cos(pi*k*x)]
    def _add_fourier_features(coords, n_f):
        feats = [coords]
        for k in range(1, n_f + 1):
            feats.append(np.sin(k * np.pi * coords))
            feats.append(np.cos(k * np.pi * coords))
        return np.concatenate(feats, axis=1)

    feat_sub = _add_fourier_features(coords_sub, n_freqs)
    feat_full = _add_fourier_features(coords_full, n_freqs)
    n_input = feat_sub.shape[1]

    # Training targets (subsampled)
    target_sub = np.zeros((h_sub * w_sub, 3), dtype=np.float64)
    ycocg_sub = ycocg[::subsample, ::subsample, :]
    for ch in range(3):
        target_sub[:, ch] = ycocg_sub[:, :, ch].ravel().astype(np.float64)

    # Normalize targets to [-1, 1] for stable training
    t_min = target_sub.min(axis=0)
    t_max = target_sub.max(axis=0)
    t_range = np.maximum(t_max - t_min, 1.0)
    target_norm = 2.0 * (target_sub - t_min) / t_range - 1.0

    # Initialize SIREN layers
    rng = np.random.RandomState(42)
    layer_sizes = [n_input] + hidden_sizes + [3]
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        is_first = (i == 0)
        W, b = _siren_init_weights(
            layer_sizes[i], layer_sizes[i + 1], is_first, omega_0, rng
        )
        weights.append(W)
        biases.append(b)

    # Adam optimizer state
    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    t0 = time.time()
    for epoch in range(n_epochs):
        # Forward
        output = _siren_forward(feat_sub, weights, biases, omega_0)
        loss = np.mean((output - target_norm) ** 2)

        # Backward
        grad_w, grad_b = _siren_backward(
            feat_sub, weights, biases, target_norm, omega_0
        )

        # Gradient clipping (max norm = 1.0)
        grad_norm = sum(np.sum(g ** 2) for g in grad_w)
        grad_norm += sum(np.sum(g ** 2) for g in grad_b)
        grad_norm = math.sqrt(grad_norm)
        if grad_norm > 1.0:
            scale = 1.0 / grad_norm
            grad_w = [g * scale for g in grad_w]
            grad_b = [g * scale for g in grad_b]

        # Adam update
        t_step = epoch + 1
        for i in range(len(weights)):
            m_w[i] = beta1 * m_w[i] + (1 - beta1) * grad_w[i]
            v_w[i] = beta2 * v_w[i] + (1 - beta2) * grad_w[i] ** 2
            m_b[i] = beta1 * m_b[i] + (1 - beta1) * grad_b[i]
            v_b[i] = beta2 * v_b[i] + (1 - beta2) * grad_b[i] ** 2

            mw_hat = m_w[i] / (1 - beta1 ** t_step)
            vw_hat = v_w[i] / (1 - beta2 ** t_step)
            mb_hat = m_b[i] / (1 - beta1 ** t_step)
            vb_hat = v_b[i] / (1 - beta2 ** t_step)

            weights[i] = weights[i] - lr * mw_hat / (np.sqrt(vw_hat) + eps)
            biases[i] = biases[i] - lr * mb_hat / (np.sqrt(vb_hat) + eps)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1:4d}: loss={loss:.6f}")

    elapsed = time.time() - t0
    print(f"  Training: {elapsed:.1f}s")

    # Evaluate on FULL image
    print("  Evaluating on full image...")
    output_full = _siren_forward(feat_full, weights, biases, omega_0)

    # Denormalize
    pred_float = (output_full + 1.0) / 2.0 * t_range + t_min
    pred_int = np.round(pred_float).astype(np.int32).reshape(h, w, 3)

    # Compute residuals per channel
    total_residual_kb = 0.0
    total_zero_pct = 0.0
    all_lossless = True
    for ch, name in enumerate(["Y", "Co", "Cg"]):
        orig_ch = ycocg[:, :, ch].astype(np.int32)
        pred_ch = pred_int[:, :, ch]
        residual, res_kb, zero_pct = _compute_residual_stats(
            orig_ch, pred_ch, name
        )
        lossless = _verify_lossless(orig_ch, pred_ch, residual)
        all_lossless = all_lossless and lossless
        total_residual_kb += res_kb
        total_zero_pct += zero_pct
        print(f"    {name}: residual={res_kb:.1f} KB, "
              f"zeros={zero_pct:.1f}%, lossless={lossless}")

    # Quantize stored weights to int8 for size estimation
    # Only store output layer (hidden layers are seeded random)
    param_count = sum(w.size for w in weights) + sum(b.size for b in biases)
    param_kb = param_count * 1 / 1024  # int8 = 1 byte each

    total_kb = _print_codec_result(
        "SIREN", param_kb, total_residual_kb, total_zero_pct / 3
    )
    print(f"    Lossless verified: {all_lossless}")
    return total_kb


# ============================================================
# 2. FOURIER NEURAL CODEC
# ============================================================

def _fourier_features(coords, freqs):
    """Build Fourier feature matrix from learnable frequencies.

    coords: (N, 2) with (y, x) in [-1, 1]
    freqs:  (n_harmonics, 2) learnable frequency vectors

    Output: (N, n_harmonics * 2) with [sin(f.dot(coord)), cos(f.dot(coord))]
    """
    # projections: (N, n_harmonics) = coords @ freqs.T
    proj = coords @ freqs.T  # (N, n_harmonics)
    return np.concatenate([np.sin(proj), np.cos(proj)], axis=1)


def fourier_neural_codec(image_path, n_harmonics=64):
    """Learnable Fourier Features + Linear codec.

    Learns optimal frequency vectors via gradient descent, then maps
    features to pixel values with a single linear layer.

    Storage: learned frequencies (n_harmonics * 2 floats)
           + linear weights (n_harmonics * 2 * 3 floats)
           + residual corrections

    Parameters
    ----------
    image_path : str
        Path to image file.
    n_harmonics : int
        Number of learnable Fourier harmonics.
    """
    print("\n" + "=" * 60)
    print("FOURIER NEURAL CODEC: Learnable Frequency Decomposition")
    print("=" * 60)

    _, ycocg, h, w = _load_image_ycocg(image_path)
    total_pixels = h * w
    subsample = 4
    lr_freq = 0.1
    lr_linear = 1e-3
    n_epochs = 300

    print(f"  Image: {w}x{h}, harmonics={n_harmonics}")
    print(f"  Training subsample: 1/{subsample}, {n_epochs} epochs")

    coords_sub, h_sub, w_sub = _make_coord_grid(h, w, subsample=subsample)
    coords_full, _, _ = _make_coord_grid(h, w, subsample=1)

    # Training targets (subsampled)
    ycocg_sub = ycocg[::subsample, ::subsample, :]
    target = np.zeros((h_sub * w_sub, 3), dtype=np.float64)
    for ch in range(3):
        target[:, ch] = ycocg_sub[:, :, ch].ravel().astype(np.float64)

    t_mean = target.mean(axis=0)
    target_centered = target - t_mean

    # Initialize learnable frequencies: log-spaced from low to high
    rng = np.random.RandomState(123)
    base_freqs = np.logspace(0, np.log10(max(h, w) / 2), n_harmonics)
    angles = rng.uniform(0, 2 * np.pi, n_harmonics)
    freqs = np.stack([
        base_freqs * np.cos(angles),
        base_freqs * np.sin(angles)
    ], axis=1) * np.pi  # (n_harmonics, 2)

    # Linear weights: (2 * n_harmonics + 1, 3) -- +1 for bias column
    n_features = 2 * n_harmonics
    W_linear = np.zeros((n_features + 1, 3), dtype=np.float64)

    # Adam state for frequencies
    m_f = np.zeros_like(freqs)
    v_f = np.zeros_like(freqs)
    m_l = np.zeros_like(W_linear)
    v_l = np.zeros_like(W_linear)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    t0 = time.time()
    n_sub = coords_sub.shape[0]

    for epoch in range(n_epochs):
        # Forward: build features from current frequencies
        feat = _fourier_features(coords_sub, freqs)
        feat_bias = np.column_stack([feat, np.ones(n_sub)])

        # Solve linear layer via least squares every 10 epochs
        # (fast, exact given current features)
        if epoch % 10 == 0:
            W_linear = np.linalg.lstsq(feat_bias, target_centered, rcond=None)[0]

        pred = feat_bias @ W_linear
        error = pred - target_centered
        loss = np.mean(error ** 2)

        # Backprop through Fourier features to get gradient w.r.t. frequencies
        # d_loss/d_freq[k] = d_loss/d_feat * d_feat/d_freq[k]
        # feat[:, k] = sin(coords @ freq[k])  ->  d/d_freq[k] = cos(...) * coords
        # feat[:, k + n_harmonics] = cos(...)  ->  d/d_freq[k] = -sin(...) * coords

        d_feat = error @ W_linear[:n_features, :].T  # (n_sub, n_features)
        proj = coords_sub @ freqs.T  # (n_sub, n_harmonics)
        cos_proj = np.cos(proj)
        sin_proj = np.sin(proj)

        # Gradient for sin features (first n_harmonics columns)
        d_sin = d_feat[:, :n_harmonics]  # (n_sub, n_harmonics)
        # Gradient for cos features (last n_harmonics columns)
        d_cos = d_feat[:, n_harmonics:]

        # Chain rule: d_loss/d_freq = sum over samples
        # d(sin(c.f))/d_f = cos(c.f) * c    => grad += d_sin * cos * coords
        # d(cos(c.f))/d_f = -sin(c.f) * c   => grad += d_cos * (-sin) * coords
        grad_freq = np.zeros_like(freqs)
        for k in range(n_harmonics):
            g_sin = d_sin[:, k] * cos_proj[:, k]      # (n_sub,)
            g_cos = d_cos[:, k] * (-sin_proj[:, k])    # (n_sub,)
            combined = g_sin + g_cos                    # (n_sub,)
            grad_freq[k] = coords_sub.T @ combined     # (2,)

        grad_freq *= 2.0 / n_sub

        # Adam update for frequencies
        t_step = epoch + 1
        m_f = beta1 * m_f + (1 - beta1) * grad_freq
        v_f = beta2 * v_f + (1 - beta2) * grad_freq ** 2
        mf_hat = m_f / (1 - beta1 ** t_step)
        vf_hat = v_f / (1 - beta2 ** t_step)
        freqs = freqs - lr_freq * mf_hat / (np.sqrt(vf_hat) + eps)

        if (epoch + 1) % 75 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1:4d}: loss={loss:.4f}")

    elapsed = time.time() - t0
    print(f"  Training: {elapsed:.1f}s")

    # Final least-squares solve with trained frequencies
    feat_sub_final = _fourier_features(coords_sub, freqs)
    feat_sub_bias = np.column_stack([feat_sub_final, np.ones(n_sub)])
    W_linear = np.linalg.lstsq(feat_sub_bias, target_centered, rcond=None)[0]

    # Evaluate on full image
    print("  Evaluating on full image...")
    feat_full = _fourier_features(coords_full, freqs)
    feat_full_bias = np.column_stack([feat_full, np.ones(total_pixels)])
    pred_full = feat_full_bias @ W_linear + t_mean
    pred_int = np.round(pred_full).astype(np.int32).reshape(h, w, 3)

    # Residuals
    total_residual_kb = 0.0
    total_zero_pct = 0.0
    all_lossless = True
    for ch, name in enumerate(["Y", "Co", "Cg"]):
        orig_ch = ycocg[:, :, ch].astype(np.int32)
        pred_ch = pred_int[:, :, ch]
        residual, res_kb, zero_pct = _compute_residual_stats(orig_ch, pred_ch, name)
        lossless = _verify_lossless(orig_ch, pred_ch, residual)
        all_lossless = all_lossless and lossless
        total_residual_kb += res_kb
        total_zero_pct += zero_pct
        print(f"    {name}: residual={res_kb:.1f} KB, "
              f"zeros={zero_pct:.1f}%, lossless={lossless}")

    # Storage: frequencies (n_harmonics * 2 float16) + linear (n_features+1)*3 float16
    freq_bytes = n_harmonics * 2 * 2  # float16
    linear_bytes = (n_features + 1) * 3 * 2
    param_kb = (freq_bytes + linear_bytes) / 1024

    total_kb = _print_codec_result(
        "Fourier Neural", param_kb, total_residual_kb, total_zero_pct / 3
    )
    print(f"    Lossless verified: {all_lossless}")
    return total_kb


# ============================================================
# 3. PROCEDURAL CODEC
# ============================================================

def _perlin_2d(x_coords, y_coords, gradients, grid_size):
    """Evaluate Perlin noise at given coordinates.

    Parameters
    ----------
    x_coords, y_coords : ndarray of float
        Coordinates to evaluate (can be any shape, will be flattened).
    gradients : ndarray (grid_size, grid_size, 2)
        Gradient vectors at each grid node.
    grid_size : int
        Size of the gradient grid.

    Returns
    -------
    ndarray : noise values in [-1, 1], same shape as input.
    """
    shape = x_coords.shape
    x = x_coords.ravel()
    y = y_coords.ravel()

    # Grid cell coordinates
    x0 = np.floor(x).astype(np.int32) % grid_size
    y0 = np.floor(y).astype(np.int32) % grid_size
    x1 = (x0 + 1) % grid_size
    y1 = (y0 + 1) % grid_size

    # Fractional position within cell
    fx = x - np.floor(x)
    fy = y - np.floor(y)

    # Smoothstep interpolation (6t^5 - 15t^4 + 10t^3)
    sx = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    sy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)

    # Dot products of distance vectors with gradient vectors
    def _dot_grad(gx, gy, dx, dy):
        return gx * dx + gy * dy

    n00 = _dot_grad(gradients[y0, x0, 0], gradients[y0, x0, 1], fx, fy)
    n10 = _dot_grad(gradients[y0, x1, 0], gradients[y0, x1, 1], fx - 1, fy)
    n01 = _dot_grad(gradients[y1, x0, 0], gradients[y1, x0, 1], fx, fy - 1)
    n11 = _dot_grad(gradients[y1, x1, 0], gradients[y1, x1, 1], fx - 1, fy - 1)

    # Bilinear interpolation with smoothstep
    nx0 = n00 * (1 - sx) + n10 * sx
    nx1 = n01 * (1 - sx) + n11 * sx
    result = nx0 * (1 - sy) + nx1 * sy

    return result.reshape(shape)


def _voronoi_2d(x_coords, y_coords, cell_centers):
    """Compute Voronoi distance field.

    Returns distance to nearest cell center for each coordinate.
    """
    n_cells = len(cell_centers)
    x = x_coords.ravel()[:, None]  # (N, 1)
    y = y_coords.ravel()[:, None]
    cx = cell_centers[:, 0][None, :]  # (1, n_cells)
    cy = cell_centers[:, 1][None, :]
    dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    min_dist = dists.min(axis=1)
    return min_dist.reshape(x_coords.shape)


def _cma_es_optimize(fitness_fn, n_params, pop_size=20, n_gens=50,
                     sigma_init=0.5, seed=42):
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) from scratch.

    A gradient-free optimizer that adapts a multivariate normal search
    distribution. Suitable for non-differentiable objectives.

    Parameters
    ----------
    fitness_fn : callable
        Function mapping parameter vector (n_params,) -> scalar (lower=better).
    n_params : int
        Dimensionality of search space.
    pop_size : int
        Population size.
    n_gens : int
        Number of generations.
    sigma_init : float
        Initial step size.
    seed : int
        Random seed.

    Returns
    -------
    best_params : ndarray of shape (n_params,)
    best_fitness : float
    """
    rng = np.random.RandomState(seed)
    mu = pop_size // 2  # number of parents

    # Selection weights (log-linear)
    raw_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = raw_weights / raw_weights.sum()
    mu_eff = 1.0 / np.sum(w ** 2)

    # Adaptation parameters
    c_sigma = (mu_eff + 2) / (n_params + mu_eff + 5)
    d_sigma = 1.0 + 2.0 * max(0, math.sqrt((mu_eff - 1) / (n_params + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / n_params) / (n_params + 4 + 2 * mu_eff / n_params)
    c1 = 2.0 / ((n_params + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n_params + 2) ** 2 + mu_eff))

    # State
    mean = rng.randn(n_params) * 0.1
    sigma = sigma_init
    C = np.eye(n_params)
    p_sigma = np.zeros(n_params)
    p_c = np.zeros(n_params)
    chi_n = math.sqrt(n_params) * (1 - 1 / (4 * n_params) + 1 / (21 * n_params ** 2))

    best_params = mean.copy()
    best_fitness = float("inf")

    for gen in range(n_gens):
        # Sample population
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C = np.eye(n_params)
            L = np.eye(n_params)

        z = rng.randn(pop_size, n_params)
        population = mean + sigma * (z @ L.T)

        # Evaluate fitness
        fitnesses = np.array([fitness_fn(ind) for ind in population])

        # Sort by fitness (ascending = better)
        order = np.argsort(fitnesses)
        if fitnesses[order[0]] < best_fitness:
            best_fitness = fitnesses[order[0]]
            best_params = population[order[0]].copy()

        # Weighted recombination
        selected = population[order[:mu]]
        old_mean = mean.copy()
        mean = w @ selected

        # Step-size control
        inv_sqrt_C = np.linalg.solve(L, np.eye(n_params)).T
        p_sigma = (1 - c_sigma) * p_sigma + \
            math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
            inv_sqrt_C @ (mean - old_mean) / sigma
        sigma = sigma * math.exp(
            c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1)
        )
        sigma = np.clip(sigma, 1e-10, 1e3)

        # Covariance update
        h_sigma = 1.0 if (np.linalg.norm(p_sigma) /
                          math.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < \
                         (1.4 + 2 / (n_params + 1)) * chi_n else 0.0

        p_c = (1 - c_c) * p_c + \
            h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * \
            (mean - old_mean) / sigma

        # Rank-mu update
        diffs = (selected - old_mean) / sigma
        C = (1 - c1 - c_mu) * C + \
            c1 * np.outer(p_c, p_c) + \
            c_mu * (diffs.T @ np.diag(w) @ diffs)

        # Enforce symmetry
        C = (C + C.T) / 2
        # Add small regularization for numerical stability
        C += 1e-10 * np.eye(n_params)

        if (gen + 1) % 10 == 0:
            print(f"    CMA-ES gen {gen + 1}: best={best_fitness:.4f}, "
                  f"sigma={sigma:.4f}")

    return best_params, best_fitness


def procedural_codec(image_path):
    """Procedural texture generation codec.

    Builds a rendering pipeline from composable operations:
    - Perlin noise at various scales
    - Gradient fills
    - Voronoi cells
    - Color lookup tables

    Optimizes parameters via CMA-ES (gradient-free).
    Stores: operation parameters + per-pixel corrections.
    """
    print("\n" + "=" * 60)
    print("PROCEDURAL CODEC: Composable Rendering Pipeline")
    print("=" * 60)

    _, ycocg, h, w = _load_image_ycocg(image_path)
    total_pixels = h * w
    subsample = 8  # Heavier subsampling for CMA-ES (many evaluations)

    print(f"  Image: {w}x{h}")
    print(f"  Training subsample: 1/{subsample}")

    # Subsampled coordinate grid
    ys_sub = np.linspace(0, 1, h // subsample)
    xs_sub = np.linspace(0, 1, w // subsample)
    X_sub, Y_sub = np.meshgrid(xs_sub, ys_sub)
    h_sub, w_sub = Y_sub.shape

    # Full coordinate grid
    ys_full = np.linspace(0, 1, h)
    xs_full = np.linspace(0, 1, w)
    X_full, Y_full = np.meshgrid(xs_full, ys_full)

    ycocg_sub = ycocg[::subsample, ::subsample, :]
    target_sub = ycocg_sub.astype(np.float64)

    def _render_procedural(params, X, Y):
        """Render a procedural image from parameters.

        The rendering pipeline:
        1. Vertical gradient (sky -> ground transition)
        2. Multi-scale Perlin noise layers
        3. Voronoi cell pattern
        4. Blending weights for each component

        Parameters encode: gradient endpoints, noise scales/amplitudes,
        Voronoi cell positions, blend weights.
        """
        hh, ww = X.shape
        result = np.zeros((hh, ww, 3), dtype=np.float64)

        idx = 0

        # Component 1: Vertical gradient per channel (6 params: top/bottom * 3ch)
        for ch in range(3):
            top_val = params[idx]
            bot_val = params[idx + 1]
            result[:, :, ch] += top_val + (bot_val - top_val) * Y
            idx += 2

        # Component 2: Horizontal gradient per channel (6 params)
        for ch in range(3):
            left_val = params[idx]
            right_val = params[idx + 1]
            result[:, :, ch] += left_val + (right_val - left_val) * X
            idx += 2

        # Component 3: Multi-scale Perlin noise (3 scales * 3 channels)
        # Each scale: grid_size, amplitude (2 params per scale per channel = 18)
        n_noise_scales = 3
        grid_sizes = [4, 8, 16]
        for scale_idx in range(n_noise_scales):
            gs = grid_sizes[scale_idx]
            rng_noise = np.random.RandomState(777 + scale_idx)
            angles = rng_noise.uniform(0, 2 * np.pi, (gs, gs))
            gradients = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

            noise = _perlin_2d(X * gs, Y * gs, gradients, gs)

            for ch in range(3):
                amp = params[idx]
                result[:, :, ch] += amp * noise
                idx += 1

        # Component 4: Quadratic terms (curvature) per channel (9 params)
        for ch in range(3):
            a_yy = params[idx]
            a_xx = params[idx + 1]
            a_xy = params[idx + 2]
            Y_c = Y - 0.5
            X_c = X - 0.5
            result[:, :, ch] += a_yy * Y_c ** 2 + a_xx * X_c ** 2 + a_xy * X_c * Y_c
            idx += 2 + 1

        return result

    n_params = 6 + 6 + 9 + 9  # gradients + h_gradients + noise amps + quadratics = 30

    def _fitness(params):
        rendered = _render_procedural(params, X_sub, Y_sub)
        return np.mean((rendered - target_sub) ** 2)

    print(f"  Procedural params: {n_params}")
    print("  Running CMA-ES optimization...")

    best_params, best_fit = _cma_es_optimize(
        _fitness, n_params, pop_size=30, n_gens=80, sigma_init=50.0, seed=42
    )
    print(f"  Best fitness (MSE): {best_fit:.2f}")

    # Render full image
    print("  Rendering full image...")
    pred_full = _render_procedural(best_params, X_full, Y_full)
    pred_int = np.round(pred_full).astype(np.int32)

    # Residuals
    total_residual_kb = 0.0
    total_zero_pct = 0.0
    all_lossless = True
    for ch, name in enumerate(["Y", "Co", "Cg"]):
        orig_ch = ycocg[:, :, ch].astype(np.int32)
        pred_ch = pred_int[:, :, ch]
        residual, res_kb, zero_pct = _compute_residual_stats(orig_ch, pred_ch, name)
        lossless = _verify_lossless(orig_ch, pred_ch, residual)
        all_lossless = all_lossless and lossless
        total_residual_kb += res_kb
        total_zero_pct += zero_pct
        print(f"    {name}: residual={res_kb:.1f} KB, "
              f"zeros={zero_pct:.1f}%, lossless={lossless}")

    param_kb = n_params * 4 / 1024  # float32
    total_kb = _print_codec_result(
        "Procedural", param_kb, total_residual_kb, total_zero_pct / 3
    )
    print(f"    Lossless verified: {all_lossless}")
    return total_kb


# ============================================================
# 4. EVOLUTIONARY CODEC
# ============================================================

# Operation set for genetic programming
OP_ADD = 0
OP_MUL = 1
OP_SIN = 2
OP_COS = 3
OP_SQRT = 4
OP_CLAMP = 5
OP_BLEND = 6
OP_NOISE = 7
OP_GRAD_V = 8
OP_GRAD_H = 9
OP_NAMES = ["add", "mul", "sin", "cos", "sqrt", "clamp",
            "blend", "noise", "grad_v", "grad_h"]
N_OPS = len(OP_NAMES)


def _eval_gene(gene, X, Y, rng_seed=42):
    """Evaluate a gene (sequence of operations) on a 2D grid.

    Each gene element is (op_code, param1, param2, param3).
    Operations are applied sequentially, building up a value grid.

    Returns (H, W) float array.
    """
    h, w = X.shape
    # Start with zeros
    acc = np.zeros((h, w), dtype=np.float64)
    temp = np.zeros((h, w), dtype=np.float64)

    for op, p1, p2, p3 in gene:
        op = int(op) % N_OPS
        if op == OP_ADD:
            acc = acc + p1
        elif op == OP_MUL:
            acc = acc * p1
        elif op == OP_SIN:
            acc = np.sin(p1 * acc + p2)
        elif op == OP_COS:
            acc = np.cos(p1 * acc + p2)
        elif op == OP_SQRT:
            acc = np.sign(acc) * np.sqrt(np.abs(acc) + 1e-8) * p1
        elif op == OP_CLAMP:
            acc = np.clip(acc, p1, p2)
        elif op == OP_BLEND:
            # Blend with coordinate-based value
            alpha = 1.0 / (1.0 + np.exp(-p3 * (Y - 0.5)))
            acc = acc * alpha + p1 * (1 - alpha)
        elif op == OP_NOISE:
            gs = max(2, int(abs(p1)) % 32 + 2)
            rng_n = np.random.RandomState(rng_seed + int(abs(p2) * 100) % 1000)
            angles = rng_n.uniform(0, 2 * np.pi, (gs, gs))
            grads = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
            noise = _perlin_2d(X * gs, Y * gs, grads, gs)
            acc = acc + p3 * noise
        elif op == OP_GRAD_V:
            acc = acc + p1 * Y + p2
        elif op == OP_GRAD_H:
            acc = acc + p1 * X + p2

    return acc


def evolutionary_codec(image_path, pop_size=50, n_gens=100):
    """Genetic programming codec.

    Evolves compact programs that generate images.
    Gene = sequence of operations on a 2D grid.
    Fitness = -(residual_entropy + program_size * lambda).

    This is the closest we can get to approximating Kolmogorov complexity:
    the shortest program that produces the output.

    Parameters
    ----------
    image_path : str
        Path to image file.
    pop_size : int
        Population size.
    n_gens : int
        Number of generations.
    """
    print("\n" + "=" * 60)
    print("EVOLUTIONARY CODEC: Genetic Programming for Image Generation")
    print("=" * 60)

    _, ycocg, h, w = _load_image_ycocg(image_path)
    total_pixels = h * w
    subsample = 8
    gene_length = 8  # operations per gene
    n_params_per_op = 4  # (op, p1, p2, p3)
    lambda_size = 0.01  # penalty for program complexity

    print(f"  Image: {w}x{h}")
    print(f"  Pop size: {pop_size}, Generations: {n_gens}")
    print(f"  Gene length: {gene_length} ops, Subsample: 1/{subsample}")

    ys_sub = np.linspace(0, 1, h // subsample)
    xs_sub = np.linspace(0, 1, w // subsample)
    X_sub, Y_sub = np.meshgrid(xs_sub, ys_sub)

    ys_full = np.linspace(0, 1, h)
    xs_full = np.linspace(0, 1, w)
    X_full, Y_full = np.meshgrid(xs_full, ys_full)

    ycocg_sub = ycocg[::subsample, ::subsample, :].astype(np.float64)
    rng = np.random.RandomState(42)

    def _random_gene():
        gene = []
        for _ in range(gene_length):
            op = rng.randint(0, N_OPS)
            p1 = rng.randn() * 50
            p2 = rng.randn() * 50
            p3 = rng.randn() * 5
            gene.append((op, p1, p2, p3))
        return gene

    def _mutate(gene, rate=0.3):
        new_gene = []
        for op, p1, p2, p3 in gene:
            if rng.random() < rate:
                # Mutate operation type
                op = rng.randint(0, N_OPS)
            if rng.random() < rate:
                p1 = p1 + rng.randn() * 10
            if rng.random() < rate:
                p2 = p2 + rng.randn() * 10
            if rng.random() < rate:
                p3 = p3 + rng.randn() * 2
            new_gene.append((op, p1, p2, p3))
        return new_gene

    def _crossover(gene_a, gene_b):
        """Subtree crossover: swap random subsequence between parents."""
        cut = rng.randint(1, gene_length)
        child = gene_a[:cut] + gene_b[cut:]
        return child

    def _fitness_one_channel(gene, target_ch, X, Y):
        """Fitness for one channel: MSE + size penalty."""
        try:
            pred = _eval_gene(gene, X, Y)
            if not np.all(np.isfinite(pred)):
                return float("inf")
            mse = np.mean((pred - target_ch) ** 2)
            return mse + lambda_size * gene_length
        except (ValueError, OverflowError, FloatingPointError):
            return float("inf")

    # Evolve separately for each channel
    best_genes = []
    for ch, name in enumerate(["Y", "Co", "Cg"]):
        print(f"\n  --- Evolving {name} channel ---")
        target_ch = ycocg_sub[:, :, ch]

        # Initialize population
        population = [_random_gene() for _ in range(pop_size)]
        best_gene = population[0]
        best_fit = float("inf")

        for gen in range(n_gens):
            # Evaluate fitness
            fitnesses = []
            for ind in population:
                fitnesses.append(_fitness_one_channel(ind, target_ch, X_sub, Y_sub))
            fitnesses = np.array(fitnesses)

            # Track best
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fit:
                best_fit = fitnesses[gen_best_idx]
                best_gene = [tuple(x) for x in population[gen_best_idx]]

            if (gen + 1) % 25 == 0 or gen == 0:
                print(f"    Gen {gen + 1:4d}: best_mse={best_fit:.2f}")

            # Selection: tournament (size 3)
            new_pop = [best_gene[:]]  # elitism
            while len(new_pop) < pop_size:
                # Tournament selection
                t_indices = rng.choice(pop_size, size=3, replace=False)
                t_fits = fitnesses[t_indices]
                parent_a = population[t_indices[np.argmin(t_fits)]]

                t_indices = rng.choice(pop_size, size=3, replace=False)
                t_fits = fitnesses[t_indices]
                parent_b = population[t_indices[np.argmin(t_fits)]]

                # Crossover
                child = _crossover(parent_a, parent_b)
                # Mutation
                child = _mutate(child)
                new_pop.append(child)

            population = new_pop

        best_genes.append(best_gene)

    # Evaluate best genes on full image
    print("\n  Evaluating on full image...")
    pred_int = np.zeros((h, w, 3), dtype=np.int32)
    for ch in range(3):
        pred_float = _eval_gene(best_genes[ch], X_full, Y_full)
        pred_int[:, :, ch] = np.round(pred_float).astype(np.int32)

    # Residuals
    total_residual_kb = 0.0
    total_zero_pct = 0.0
    all_lossless = True
    for ch, name in enumerate(["Y", "Co", "Cg"]):
        orig_ch = ycocg[:, :, ch].astype(np.int32)
        pred_ch = pred_int[:, :, ch]
        residual, res_kb, zero_pct = _compute_residual_stats(orig_ch, pred_ch, name)
        lossless = _verify_lossless(orig_ch, pred_ch, residual)
        all_lossless = all_lossless and lossless
        total_residual_kb += res_kb
        total_zero_pct += zero_pct
        print(f"    {name}: residual={res_kb:.1f} KB, "
              f"zeros={zero_pct:.1f}%, lossless={lossless}")

    # Storage: 3 genes * gene_length * 4 params * float16
    gene_bytes = 3 * gene_length * n_params_per_op * 2
    param_kb = gene_bytes / 1024

    total_kb = _print_codec_result(
        "Evolutionary", param_kb, total_residual_kb, total_zero_pct / 3
    )
    print(f"    Lossless verified: {all_lossless}")
    print(f"    Program: {gene_length} ops/channel, {3 * gene_length} total ops")
    return total_kb


# ============================================================
# 5. HYBRID GENERATIVE CODEC
# ============================================================

def _simple_kmeans(pixels, k=4, max_iter=30, seed=42):
    """Simple k-means clustering from scratch.

    Parameters
    ----------
    pixels : ndarray (N, 3)
        Pixel values to cluster.
    k : int
        Number of clusters.
    max_iter : int
        Maximum iterations.
    seed : int
        Random seed.

    Returns
    -------
    labels : ndarray (N,)
        Cluster assignment for each pixel.
    centers : ndarray (k, 3)
        Cluster centers.
    """
    rng = np.random.RandomState(seed)
    n = len(pixels)
    # Initialize with k-means++ style
    centers = np.zeros((k, pixels.shape[1]), dtype=np.float64)
    centers[0] = pixels[rng.randint(n)]
    for c in range(1, k):
        dists = np.min([np.sum((pixels - centers[j]) ** 2, axis=1)
                        for j in range(c)], axis=0)
        probs = dists / dists.sum()
        centers[c] = pixels[rng.choice(n, p=probs)]

    labels = np.zeros(n, dtype=np.int32)
    for iteration in range(max_iter):
        # Assign
        dists = np.stack([np.sum((pixels - c) ** 2, axis=1) for c in centers])
        new_labels = np.argmin(dists, axis=0).astype(np.int32)

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        # Update centers
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = pixels[mask].mean(axis=0)

    return labels, centers


def _fit_region_gradient(coords, values):
    """Fit a linear gradient to a region.

    Solves: value ~= a*y + b*x + c  (least squares)
    Returns coefficients (a, b, c).
    """
    n = len(coords)
    if n == 0:
        return np.zeros(3)
    # Design matrix: [y, x, 1]
    A = np.column_stack([coords, np.ones(n)])
    coeffs = np.linalg.lstsq(A, values, rcond=None)[0]
    return coeffs


def _fit_region_fourier(coords, values, n_freqs=8, rng_seed=42):
    """Fit Fourier features to a region for richer representation.

    Returns (weights, freqs) where prediction = features @ weights.
    """
    rng = np.random.RandomState(rng_seed)
    n = len(coords)
    if n < 10:
        return np.zeros(1), np.zeros((1, 2))

    # Generate random Fourier features
    freqs = rng.randn(n_freqs, 2) * np.pi * 2
    proj = coords @ freqs.T  # (N, n_freqs)
    features = np.column_stack([
        np.sin(proj), np.cos(proj), np.ones(n)
    ])  # (N, 2*n_freqs + 1)

    weights = np.linalg.lstsq(features, values, rcond=None)[0]
    return weights, freqs


def hybrid_generative(image_path):
    """Hybrid generative codec combining multiple approaches.

    Strategy:
    1. Segment image into regions via k-means on color
    2. For each region, pick the best generator:
       - Smooth regions (sky): gradient generator
       - Textured regions (sand): Fourier features
       - Complex regions: SIREN-style per-region
    3. Combine predictions and compute global residual
    """
    print("\n" + "=" * 60)
    print("HYBRID GENERATIVE CODEC: Region-Adaptive Generation")
    print("=" * 60)

    img, ycocg, h, w = _load_image_ycocg(image_path)
    total_pixels = h * w
    n_clusters = 4
    n_region_freqs = 16

    print(f"  Image: {w}x{h}, {total_pixels:,} pixels")
    print(f"  Clusters: {n_clusters}")

    # Step 1: Segment by color in YCoCg space
    print("  Segmenting image...")
    pixels_flat = ycocg.reshape(-1, 3).astype(np.float64)

    # Subsample for clustering speed
    sub_factor = 4
    sub_pixels = pixels_flat[::sub_factor]
    labels_sub, centers = _simple_kmeans(sub_pixels, k=n_clusters, seed=42)

    # Assign all pixels to nearest center
    dists_all = np.stack([
        np.sum((pixels_flat - c) ** 2, axis=1) for c in centers
    ])
    labels_full = np.argmin(dists_all, axis=0).astype(np.int32)
    label_map = labels_full.reshape(h, w)

    for c in range(n_clusters):
        count = np.sum(labels_full == c)
        print(f"    Cluster {c}: {count:,} px ({100 * count / total_pixels:.1f}%), "
              f"center=({centers[c, 0]:.0f}, {centers[c, 1]:.0f}, {centers[c, 2]:.0f})")

    # Step 2: Build coordinate arrays
    ys = np.linspace(-1, 1, h)
    xs = np.linspace(-1, 1, w)
    Y_grid, X_grid = np.meshgrid(ys, xs, indexing="ij")
    coords_all = np.stack([Y_grid.ravel(), X_grid.ravel()], axis=-1)

    # Step 3: Fit a generator per region per channel
    print("  Fitting region generators...")
    pred_full = np.zeros((total_pixels, 3), dtype=np.float64)
    total_model_params = 0

    for ch, ch_name in enumerate(["Y", "Co", "Cg"]):
        target_ch = ycocg[:, :, ch].ravel().astype(np.float64)

        for c in range(n_clusters):
            mask = labels_full == c
            region_coords = coords_all[mask]
            region_target = target_ch[mask]
            n_region = mask.sum()

            if n_region == 0:
                continue

            # Decide generator based on region variance
            variance = np.var(region_target)

            if variance < 100:
                # Low variance: simple gradient is enough
                coeffs = _fit_region_gradient(region_coords, region_target)
                A = np.column_stack([region_coords, np.ones(n_region)])
                region_pred = A @ coeffs
                total_model_params += 3
            else:
                # Higher variance: use Fourier features
                weights, freqs = _fit_region_fourier(
                    region_coords, region_target,
                    n_freqs=n_region_freqs,
                    rng_seed=42 + ch * 100 + c
                )
                proj = region_coords @ freqs.T
                features = np.column_stack([
                    np.sin(proj), np.cos(proj), np.ones(n_region)
                ])
                region_pred = features @ weights
                total_model_params += n_region_freqs * 2 + (2 * n_region_freqs + 1)

            pred_full[mask, ch] = region_pred

    pred_int = np.round(pred_full).astype(np.int32).reshape(h, w, 3)

    # Step 4: Compute residuals
    print("  Computing residuals...")
    total_residual_kb = 0.0
    total_zero_pct = 0.0
    all_lossless = True
    for ch, name in enumerate(["Y", "Co", "Cg"]):
        orig_ch = ycocg[:, :, ch].astype(np.int32)
        pred_ch = pred_int[:, :, ch]
        residual, res_kb, zero_pct = _compute_residual_stats(orig_ch, pred_ch, name)
        lossless = _verify_lossless(orig_ch, pred_ch, residual)
        all_lossless = all_lossless and lossless
        total_residual_kb += res_kb
        total_zero_pct += zero_pct
        print(f"    {name}: residual={res_kb:.1f} KB, "
              f"zeros={zero_pct:.1f}%, lossless={lossless}")

    # Storage: model params (float16) + cluster centers (k*3 int16) + label map overhead
    model_param_kb = total_model_params * 2 / 1024  # float16
    cluster_kb = n_clusters * 3 * 2 / 1024  # cluster centers, int16
    # Label map: entropy of cluster assignments
    label_bits = entropy_bits(label_map)
    label_kb = label_bits / 8 / 1024
    param_kb = model_param_kb + cluster_kb + label_kb

    total_kb = _print_codec_result(
        "Hybrid Generative", param_kb, total_residual_kb, total_zero_pct / 3
    )
    print(f"    Lossless verified: {all_lossless}")
    print(f"    Model params: {total_model_params}, "
          f"label map: {label_kb:.1f} KB")
    return total_kb


# ============================================================
# Main: run key approaches and compare
# ============================================================

if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "img.png"

    print("*" * 70)
    print("  GENERATIVE CODEC SUITE")
    print("  Paradigm: Image = Program + Corrections")
    print("  All implementations: pure numpy, no ML frameworks")
    print("*" * 70)

    results = {}

    # Run all codecs
    results["SIREN"] = siren_codec(image_path, hidden_sizes=[256, 128], n_freqs=12)
    results["Fourier"] = fourier_neural_codec(image_path, n_harmonics=64)
    results["Procedural"] = procedural_codec(image_path)
    results["Evolutionary"] = evolutionary_codec(image_path, pop_size=50, n_gens=100)
    results["Hybrid"] = hybrid_generative(image_path)

    # Summary comparison
    print("\n" + "=" * 70)
    print("  GENERATIVE CODEC COMPARISON")
    print("=" * 70)
    print(f"  {'Codec':<20} {'Total KB':>10} {'vs PRISM':>10}")
    print("  " + "-" * 42)
    for name, kb in sorted(results.items(), key=lambda x: x[1]):
        ratio = kb / PRISM_BASELINE_KB
        print(f"  {name:<20} {kb:>10.1f} {ratio:>10.2f}x")
    print(f"  {'PRISM (baseline)':<20} {PRISM_BASELINE_KB:>10.1f} {'1.00x':>10}")
    print("  " + "-" * 42)

    best_name = min(results, key=results.get)
    print(f"\n  Best generative codec: {best_name} ({results[best_name]:.1f} KB)")
    print(f"\n  KEY INSIGHT: Can a compact program + corrections beat prediction + coding?")
    if results[best_name] < PRISM_BASELINE_KB:
        print(f"  RESULT: YES -- {best_name} saves "
              f"{PRISM_BASELINE_KB - results[best_name]:.1f} KB over PRISM")
    else:
        print(f"  RESULT: Not yet -- best is {results[best_name] - PRISM_BASELINE_KB:.1f} KB "
              f"above PRISM")
        print(f"  The residual dominates. The program captures smooth structure but")
        print(f"  real images have too much per-pixel detail for small programs.")
        print(f"  Future: pretrained image priors could dramatically reduce residual.")
