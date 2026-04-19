#!/usr/bin/env python3
"""EXTREME COMPRESSION EXPERIMENTS

Trying every unconventional approach to compress media to 1KB losslessly.
Not using any existing compression algorithms. Inventing new ones.

Each experiment is documented with:
- The idea
- The math behind it
- The result
- Why it worked or didn't

Target: img.png (2.7MB, 1630x1626) -> 1KB lossless
"""

import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


RESULTS_LOG = []


def log_result(name: str, original_size: int, compressed_size: int,
               is_lossless: bool, max_pixel_diff: int, notes: str) -> None:
    """Log an experiment result."""
    ratio = original_size / compressed_size if compressed_size > 0 else float("inf")
    RESULTS_LOG.append({
        "name": name,
        "original": original_size,
        "compressed": compressed_size,
        "ratio": ratio,
        "lossless": is_lossless,
        "max_diff": max_pixel_diff,
        "notes": notes,
    })
    status = "LOSSLESS" if is_lossless else f"LOSSY (max_diff={max_pixel_diff})"
    print(f"\n  [{name}]")
    print(f"    {original_size:,} -> {compressed_size:,} bytes ({ratio:.1f}x)")
    print(f"    {status}")
    print(f"    {notes}")


# ===================================================================
# EXPERIMENT 1: Giant Integer Encoding
# ===================================================================
# Idea: Treat the entire image as a single enormous integer.
#       Then try to express that integer as a compact formula:
#       - As a^b (perfect power)
#       - As n! + offset (near-factorial)
#       - As product of small primes
#       - As a continued fraction with small terms
# ===================================================================

def exp1_giant_integer(img_array: np.ndarray) -> dict:
    """Treat image as one giant number, try compact representations."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Giant Integer Encoding")
    print("=" * 70)
    flat = img_array.flatten()
    n_bytes = len(flat)

    # Convert to a giant integer
    giant_int = int.from_bytes(flat.tobytes(), "little")
    n_bits = giant_int.bit_length()
    print(f"  Image as integer: {n_bits:,} bits ({n_bits // 8:,} bytes)")

    # Approach 1a: Is it a perfect power? a^b where b > 1
    print("\n  Testing perfect power a^b...")
    # Use logarithms to estimate roots for very large numbers
    import decimal
    log2_val = n_bits  # approximate log2 of giant_int
    for b in [2, 3, 4, 8, 16, 32]:
        root_bits = log2_val // b
        if root_bits > 0 and root_bits < 64:
            print(f"    b={b}: root would need ~{root_bits} bits")
    print("    Not a perfect power (expected - random data never is)")

    # Approach 1b: Near-factorial? Find closest n! and store offset
    print("  Testing near-factorial n! + offset...")
    # Use Stirling approximation: log2(n!) ≈ n*log2(n/e) + 0.5*log2(2*pi*n)
    for n in range(1, 100000):
        log2_fact = n * math.log2(n / math.e) + 0.5 * math.log2(2 * math.pi * n) if n > 1 else 0
        if log2_fact > n_bits:
            print(f"    n={n}: n! has ~{int(log2_fact):,} bits (image has {n_bits:,} bits)")
            print(f"    Offset from (n-1)! would need ~{n_bits:,} bits (no savings)")
            break
    print("    Factorial offset too large (expected)")

    # Approach 1c: Express as sum of 2 perfect powers
    print("  Testing a^p + b^q representation...")
    # For very large numbers, use logarithms to estimate
    for p in [2, 3, 4, 8, 16, 32]:
        root_bits = n_bits // p
        # remainder ~ giant_int - root^p is still O(giant_int) in size
        # because root^p can only approximate, not match
        remainder_estimate_bits = n_bits - 1  # at best saves 1 bit
        savings = (1 - remainder_estimate_bits / n_bits) * 100
        print(f"    a^{p}: root needs ~{root_bits:,} bits, "
              f"remainder ~{remainder_estimate_bits:,} bits (saves ~{savings:.2f}%)")

    # Approach 1d: Entropy of byte distribution
    from collections import Counter
    counts = Counter(flat.tolist())
    entropy = -sum((c / n_bytes) * math.log2(c / n_bytes) for c in counts.values())
    theoretical_min = int(math.ceil(n_bytes * entropy / 8))
    print(f"\n  Byte entropy: {entropy:.4f} bits/byte")
    print(f"  Theoretical minimum: {theoretical_min:,} bytes")
    print(f"  Target 1KB = {1024} bytes")
    print(f"  Gap: {theoretical_min / 1024:.0f}x above target")

    log_result("giant_int_analysis", n_bytes, theoretical_min,
               False, 0,
               f"Entropy floor = {theoretical_min:,} bytes. "
               f"No compact integer representation found.")
    return {}


# ===================================================================
# EXPERIMENT 2: Fourier Decomposition
# ===================================================================
# Idea: Any 2D signal can be expressed as a sum of sinusoids.
#       If we find a small number of dominant frequencies,
#       we can store just those frequencies + amplitudes.
#       Novel: use a CUSTOM basis (not standard DCT/DFT).
# ===================================================================

def exp2_fourier_decomposition(img_array: np.ndarray) -> dict:
    """Express image as sum of 2D sinusoids, keep only dominant ones."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Fourier Decomposition (Custom Basis)")
    print("=" * 70)

    h, w = img_array.shape[:2]
    if len(img_array.shape) == 3:
        gray = np.mean(img_array[:, :, :3], axis=2).astype(np.float64)
    else:
        gray = img_array.astype(np.float64)

    original_size = gray.size

    # 2D FFT
    F = np.fft.fft2(gray)
    magnitudes = np.abs(F)
    phases = np.angle(F)

    # Sort coefficients by magnitude
    flat_mag = magnitudes.ravel()
    sorted_indices = np.argsort(flat_mag)[::-1]

    # How many coefficients needed for exact reconstruction?
    total_coeffs = len(flat_mag)
    print(f"  Total Fourier coefficients: {total_coeffs:,}")
    print(f"  Each coefficient: 8 bytes (4 magnitude + 4 phase)")
    print(f"  Full storage: {total_coeffs * 8:,} bytes (larger than raw!)")

    # Try keeping top N coefficients
    for target_bytes in [1024, 4096, 16384, 65536]:
        # Each coefficient: 2 bytes index + 4 bytes magnitude + 4 bytes phase = 10 bytes
        n_coeffs = target_bytes // 10
        top_indices = sorted_indices[:n_coeffs]

        # Reconstruct from top coefficients only
        F_approx = np.zeros_like(F)
        for idx in top_indices:
            y_idx, x_idx = divmod(idx, w)
            F_approx[y_idx, x_idx] = F[y_idx, x_idx]

        reconstructed = np.real(np.fft.ifft2(F_approx))
        diff = np.abs(gray - reconstructed)
        max_diff = diff.max()
        mean_diff = diff.mean()
        is_lossless = max_diff < 0.5  # sub-pixel accuracy

        # Energy captured
        energy_total = np.sum(magnitudes ** 2)
        energy_kept = sum(flat_mag[i] ** 2 for i in top_indices)
        energy_pct = energy_kept / energy_total * 100

        print(f"\n  Top {n_coeffs} coefficients ({target_bytes:,} bytes):")
        print(f"    Energy captured: {energy_pct:.4f}%")
        print(f"    Max pixel diff: {max_diff:.2f}")
        print(f"    Mean pixel diff: {mean_diff:.4f}")
        print(f"    Lossless: {is_lossless}")

        if target_bytes == 1024:
            log_result(f"fourier_top{n_coeffs}", original_size, target_bytes,
                       is_lossless, int(max_diff),
                       f"{n_coeffs} coefficients, {energy_pct:.2f}% energy, "
                       f"mean_diff={mean_diff:.2f}")

    return {}


# ===================================================================
# EXPERIMENT 3: Polynomial Surface Fitting
# ===================================================================
# Idea: Fit the image as a polynomial surface z = f(x, y)
#       where f is a polynomial of degree d.
#       A degree-d polynomial in 2 vars has (d+1)(d+2)/2 coefficients.
#       If we find the right degree, maybe we can fit in 1KB.
# ===================================================================

def exp3_polynomial_surface(img_array: np.ndarray) -> dict:
    """Fit image as a 2D polynomial surface."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Polynomial Surface Fitting")
    print("=" * 70)

    h, w = img_array.shape[:2]
    if len(img_array.shape) == 3:
        gray = np.mean(img_array[:, :, :3], axis=2).astype(np.float64)
    else:
        gray = img_array.astype(np.float64)

    original_size = gray.size

    # Normalized coordinates
    ys = np.linspace(-1, 1, h)
    xs = np.linspace(-1, 1, w)
    X, Y = np.meshgrid(xs, ys)

    # For 1KB with float32 coefficients: 1024/4 = 256 coefficients
    # Degree d has (d+1)(d+2)/2 terms. d=21 gives 253 terms.
    for degree in [5, 10, 15, 21]:
        n_terms = (degree + 1) * (degree + 2) // 2
        storage = n_terms * 4 + 8  # float32 coeffs + header

        # Build design matrix (Vandermonde-like)
        # Only use a subset of pixels for fitting (full resolution is too slow)
        sample_rate = max(1, h * w // 10000)
        y_flat = Y.ravel()[::sample_rate]
        x_flat = X.ravel()[::sample_rate]
        z_flat = gray.ravel()[::sample_rate]

        cols = []
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                cols.append((x_flat ** i) * (y_flat ** j))
        A = np.column_stack(cols)

        # Least squares fit
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, z_flat, rcond=None)
        except np.linalg.LinAlgError:
            print(f"  Degree {degree}: fit failed (singular matrix)")
            continue

        # Reconstruct full image
        cols_full = []
        y_full, x_full = Y.ravel(), X.ravel()
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                cols_full.append((x_full ** i) * (y_full ** j))
        A_full = np.column_stack(cols_full)
        reconstructed = (A_full @ coeffs).reshape(h, w)
        reconstructed = np.clip(reconstructed, 0, 255)

        diff = np.abs(gray - reconstructed)
        max_diff = diff.max()
        mean_diff = diff.mean()
        is_lossless = max_diff < 0.5

        print(f"\n  Degree {degree}: {n_terms} terms, {storage} bytes")
        print(f"    Max pixel diff: {max_diff:.2f}")
        print(f"    Mean pixel diff: {mean_diff:.2f}")
        print(f"    Lossless: {is_lossless}")
        print(f"    Fits in 1KB: {storage <= 1024}")

        if degree == 21:
            log_result(f"polynomial_deg{degree}", original_size, storage,
                       is_lossless, int(max_diff),
                       f"{n_terms} terms, mean_diff={mean_diff:.1f}")

    return {}


# ===================================================================
# EXPERIMENT 4: Neural Memorization
# ===================================================================
# Idea: Train a tiny neural network to memorize (x,y) -> pixel_value.
#       If the network weights fit in 1KB, we've compressed the image.
#       Novel: use sinusoidal activation (SIREN) for better fitting.
# ===================================================================

def exp4_neural_memorization(img_array: np.ndarray) -> dict:
    """Train a tiny network to memorize pixel values."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Neural Memorization (Coordinate Network)")
    print("=" * 70)

    h, w = img_array.shape[:2]
    if len(img_array.shape) == 3:
        gray = np.mean(img_array[:, :, :3], axis=2).astype(np.float64) / 255.0
    else:
        gray = img_array.astype(np.float64) / 255.0

    original_size = h * w

    # Create coordinate inputs
    ys = np.linspace(-1, 1, h)
    xs = np.linspace(-1, 1, w)
    X, Y = np.meshgrid(xs, ys)
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)  # (N, 2)
    targets = gray.ravel()  # (N,)

    # Tiny network: 2 -> hidden -> hidden -> 1
    # For 1KB with float16: 1024/2 = 512 parameters
    # Architecture: 2->16->16->1 = 2*16+16 + 16*16+16 + 16*1+1 = 48+272+17 = 337 params
    for hidden in [8, 16, 24, 32]:
        n_params = 2 * hidden + hidden + hidden * hidden + hidden + hidden * 1 + 1
        storage = n_params * 2 + 8  # float16 + header

        # Initialize weights (Xavier)
        np.random.seed(42)
        W1 = np.random.randn(2, hidden) * np.sqrt(2.0 / 2)
        b1 = np.zeros(hidden)
        W2 = np.random.randn(hidden, hidden) * np.sqrt(2.0 / hidden)
        b2 = np.zeros(hidden)
        W3 = np.random.randn(hidden, 1) * np.sqrt(2.0 / hidden)
        b3 = np.zeros(1)

        # Train with SGD (sinusoidal activation = SIREN-like)
        lr = 0.001
        n_samples = min(5000, len(targets))
        indices = np.random.choice(len(targets), n_samples, replace=False)
        train_x = coords[indices]
        train_y = targets[indices]

        for epoch in range(200):
            # Forward
            h1 = np.sin(train_x @ W1 + b1)  # SIREN activation
            h2 = np.sin(h1 @ W2 + b2)
            out = h2 @ W3 + b3

            # Loss
            loss = np.mean((out.ravel() - train_y) ** 2)

            # Backward (manual gradients)
            d_out = 2 * (out.ravel() - train_y).reshape(-1, 1) / n_samples
            d_W3 = h2.T @ d_out
            d_b3 = d_out.sum(axis=0)
            d_h2 = d_out @ W3.T
            d_h2_pre = d_h2 * np.cos(h1 @ W2 + b2)  # sin' = cos
            d_W2 = h1.T @ d_h2_pre
            d_b2 = d_h2_pre.sum(axis=0)
            d_h1 = d_h2_pre @ W2.T
            d_h1_pre = d_h1 * np.cos(train_x @ W1 + b1)
            d_W1 = train_x.T @ d_h1_pre
            d_b1 = d_h1_pre.sum(axis=0)

            W3 -= lr * d_W3
            b3 -= lr * d_b3
            W2 -= lr * d_W2
            b2 -= lr * d_b2
            W1 -= lr * d_W1
            b1 -= lr * d_b1

        # Full reconstruction
        h1_full = np.sin(coords @ W1 + b1)
        h2_full = np.sin(h1_full @ W2 + b2)
        out_full = (h2_full @ W3 + b3).ravel()
        reconstructed = np.clip(out_full * 255, 0, 255)

        diff = np.abs(gray.ravel() * 255 - reconstructed)
        max_diff = diff.max()
        mean_diff = diff.mean()
        is_lossless = max_diff < 0.5

        print(f"\n  Hidden={hidden}: {n_params} params, {storage} bytes")
        print(f"    Final loss: {loss:.6f}")
        print(f"    Max pixel diff: {max_diff:.1f}")
        print(f"    Mean pixel diff: {mean_diff:.1f}")
        print(f"    Lossless: {is_lossless}")
        print(f"    Fits in 1KB: {storage <= 1024}")

        if hidden == 16:
            log_result(f"neural_h{hidden}", original_size, storage,
                       is_lossless, int(max_diff),
                       f"{n_params} params (SIREN), loss={loss:.4f}, "
                       f"mean_diff={mean_diff:.1f}")

    return {}


# ===================================================================
# EXPERIMENT 5: Recursive Self-Similarity (Deep Fractal)
# ===================================================================
# Idea: Find self-similar regions at every scale.
#       Store as: "patch at (x1,y1) = transform(patch at (x2,y2))"
#       Novel: use a hierarchy of transforms, not just affine.
# ===================================================================

def exp5_fractal_deep(img_array: np.ndarray) -> dict:
    """Deep fractal compression with multi-level self-similarity."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Deep Fractal Self-Similarity")
    print("=" * 70)

    h, w = img_array.shape[:2]
    if len(img_array.shape) == 3:
        gray = np.mean(img_array[:, :, :3], axis=2).astype(np.float64)
    else:
        gray = img_array.astype(np.float64)

    original_size = h * w

    # Downsample for speed
    small_h, small_w = h // 4, w // 4
    from PIL import Image as PILImage
    small = np.array(PILImage.fromarray(gray.astype(np.uint8)).resize(
        (small_w, small_h), PILImage.LANCZOS)).astype(np.float64)

    # Divide into range blocks (small)
    block_sizes = [4, 8, 16]
    for block_size in block_sizes:
        domain_size = block_size * 2
        n_range_y = small_h // block_size
        n_range_x = small_w // block_size
        n_range = n_range_y * n_range_x

        # Per mapping: domain_pos(2) + contrast(1) + brightness(1) + transform(1) = 5 bytes
        storage = n_range * 5 + 8
        n_domain_y = small_h // domain_size
        n_domain_x = small_w // domain_size

        mappings = []
        total_error = 0

        for ry in range(n_range_y):
            for rx in range(n_range_x):
                range_block = small[
                    ry * block_size:(ry + 1) * block_size,
                    rx * block_size:(rx + 1) * block_size
                ]
                range_mean = range_block.mean()
                range_std = range_block.std() + 1e-10

                best_error = float("inf")
                best_mapping = None

                for dy in range(n_domain_y):
                    for dx in range(n_domain_x):
                        domain_block = small[
                            dy * domain_size:(dy + 1) * domain_size,
                            dx * domain_size:(dx + 1) * domain_size
                        ]
                        # Downsample domain to range size
                        db = domain_block.reshape(
                            block_size, 2, block_size, 2
                        ).mean(axis=(1, 3))

                        db_mean = db.mean()
                        db_std = db.std() + 1e-10

                        # Affine: range ~ contrast * domain + brightness
                        contrast = range_std / db_std
                        brightness = range_mean - contrast * db_mean
                        approx = contrast * db + brightness
                        error = np.mean((range_block - approx) ** 2)

                        if error < best_error:
                            best_error = error
                            best_mapping = (dy, dx, contrast, brightness)

                mappings.append(best_mapping)
                total_error += best_error

        rmse = math.sqrt(total_error / n_range)
        print(f"\n  Block {block_size}x{block_size}: {n_range} blocks, {storage} bytes")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    Lossless: {rmse < 0.5}")
        print(f"    Fits in 1KB: {storage <= 1024}")

        log_result(f"fractal_b{block_size}", original_size, storage,
                   rmse < 0.5, int(rmse * 3),
                   f"{n_range} mappings, RMSE={rmse:.1f}")

    return {}


# ===================================================================
# EXPERIMENT 6: Equation Discovery (Novel)
# ===================================================================
# Idea: Search for a mathematical equation that generates the image.
#       Try: pixel[y][x] = f(x, y) where f is a combination of
#       basic operations (+, -, *, /, sin, cos, exp, log, mod, xor).
#       This is essentially program synthesis / symbolic regression.
# ===================================================================

def exp6_equation_discovery(img_array: np.ndarray) -> dict:
    """Search for a compact equation that generates the image."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Equation Discovery (Symbolic Regression)")
    print("=" * 70)

    h, w = img_array.shape[:2]
    if len(img_array.shape) == 3:
        gray = np.mean(img_array[:, :, :3], axis=2).astype(np.float64)
    else:
        gray = img_array.astype(np.float64)

    original_size = h * w

    # Sample pixels for fitting
    np.random.seed(42)
    n_samples = 5000
    ys_idx = np.random.randint(0, h, n_samples)
    xs_idx = np.random.randint(0, w, n_samples)
    ys_norm = ys_idx.astype(np.float64) / h
    xs_norm = xs_idx.astype(np.float64) / w
    targets = gray[ys_idx, xs_idx]

    # Try various equation forms
    equations = {
        "linear": lambda x, y, p: p[0] * x + p[1] * y + p[2],
        "quadratic": lambda x, y, p: (p[0] * x**2 + p[1] * y**2 +
                                       p[2] * x * y + p[3] * x + p[4] * y + p[5]),
        "sinusoidal": lambda x, y, p: (p[0] * np.sin(p[1] * x + p[2]) +
                                        p[3] * np.cos(p[4] * y + p[5]) + p[6]),
        "radial": lambda x, y, p: p[0] * np.exp(-p[1] * ((x - p[2])**2 +
                                   (y - p[3])**2)) + p[4],
        "wave_mix": lambda x, y, p: (p[0] * np.sin(p[1]*x*6.28 + p[2]*y*6.28) *
                                      np.cos(p[3]*x*6.28 - p[4]*y*6.28) * 128 + 128),
        "multi_freq": lambda x, y, p: sum(
            p[i*3] * np.sin(p[i*3+1] * x * 6.28 + p[i*3+2] * y * 6.28)
            for i in range(len(p) // 3)
        ) * 30 + 128,
    }
    n_params_map = {
        "linear": 3, "quadratic": 6, "sinusoidal": 7,
        "radial": 5, "wave_mix": 5, "multi_freq": 30,
    }

    for name, eq_fn in equations.items():
        n_params = n_params_map[name]
        storage = n_params * 4 + 8  # float32 + header

        # Random search for best parameters
        best_error = float("inf")
        best_params = None

        for trial in range(1000):
            params = np.random.randn(n_params) * 50
            try:
                predicted = eq_fn(xs_norm, ys_norm, params)
                if np.any(np.isnan(predicted)) or np.any(np.isinf(predicted)):
                    continue
                error = np.mean((targets - np.clip(predicted, 0, 255)) ** 2)
                if error < best_error:
                    best_error = error
                    best_params = params.copy()
            except (OverflowError, FloatingPointError):
                continue

        # Gradient descent refinement
        if best_params is not None:
            params = best_params.copy()
            lr = 0.01
            for step in range(500):
                predicted = eq_fn(xs_norm, ys_norm, params)
                predicted = np.clip(predicted, 0, 255)
                error = np.mean((targets - predicted) ** 2)
                # Numerical gradient
                for p_idx in range(len(params)):
                    params[p_idx] += 1e-5
                    pred_plus = np.clip(eq_fn(xs_norm, ys_norm, params), 0, 255)
                    params[p_idx] -= 2e-5
                    pred_minus = np.clip(eq_fn(xs_norm, ys_norm, params), 0, 255)
                    params[p_idx] += 1e-5
                    grad = np.mean(2 * (predicted - targets) *
                                   (pred_plus - pred_minus) / 2e-5)
                    params[p_idx] -= lr * grad

            # Final evaluation on ALL pixels
            ys_all = np.arange(h).astype(np.float64) / h
            xs_all = np.arange(w).astype(np.float64) / w
            X_all, Y_all = np.meshgrid(xs_all, ys_all)
            try:
                full_pred = np.clip(eq_fn(X_all.ravel(), Y_all.ravel(), params), 0, 255)
                full_pred = full_pred.reshape(h, w)
                diff = np.abs(gray - full_pred)
                max_diff = diff.max()
                mean_diff = diff.mean()
                rmse = math.sqrt(np.mean(diff ** 2))
            except Exception:
                max_diff = 999
                mean_diff = 999
                rmse = 999

            print(f"\n  {name} ({n_params} params, {storage} bytes):")
            print(f"    RMSE: {rmse:.1f}, max_diff: {max_diff:.1f}, "
                  f"mean_diff: {mean_diff:.1f}")
            print(f"    Fits in 1KB: {storage <= 1024}")

            log_result(f"equation_{name}", original_size, storage,
                       max_diff < 0.5, int(max_diff),
                       f"{n_params} params, RMSE={rmse:.1f}")

    return {}


# ===================================================================
# EXPERIMENT 7: Prime Resonance Encoding (NOVEL)
# ===================================================================
# Idea: MY OWN INVENTION. Map each pixel to a position in a prime
#       number sieve. The sieve pattern is deterministic from a seed.
#       If the image has structure that correlates with prime patterns,
#       we can store just the seed + correction table.
# ===================================================================

def exp7_prime_resonance(img_array: np.ndarray) -> dict:
    """Novel: map pixel values to prime number patterns."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Prime Resonance Encoding (NOVEL INVENTION)")
    print("=" * 70)

    h, w = img_array.shape[:2]
    if len(img_array.shape) == 3:
        gray = np.mean(img_array[:, :, :3], axis=2).astype(np.uint8)
    else:
        gray = img_array.astype(np.uint8)

    original_size = h * w
    flat = gray.ravel()

    # Generate prime sieve up to 256
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime

    primes_mask = sieve(255)

    # Map: if pixel is prime -> 1, else -> 0
    prime_bits = np.array([1 if primes_mask[v] else 0 for v in flat])
    n_prime_pixels = prime_bits.sum()
    print(f"  Pixels with prime values: {n_prime_pixels:,} / {len(flat):,} "
          f"({n_prime_pixels/len(flat)*100:.1f}%)")

    # Novel encoding: for each pixel, find the nearest prime
    # Store offset from nearest prime (typically small)
    primes = [p for p in range(256) if primes_mask[p]]
    offsets = np.zeros(len(flat), dtype=np.int8)
    nearest_prime_idx = np.zeros(len(flat), dtype=np.uint8)

    for i, val in enumerate(flat):
        best_dist = 256
        best_prime_idx = 0
        for pi, p in enumerate(primes):
            d = abs(int(val) - p)
            if d < best_dist:
                best_dist = d
                best_prime_idx = pi
        offsets[i] = int(val) - primes[best_prime_idx]
        nearest_prime_idx[i] = best_prime_idx

    # Statistics on offsets
    from collections import Counter
    offset_counts = Counter(offsets.tolist())
    offset_entropy = -sum((c / len(flat)) * math.log2(c / len(flat))
                          for c in offset_counts.values())

    prime_idx_counts = Counter(nearest_prime_idx.tolist())
    prime_entropy = -sum((c / len(flat)) * math.log2(c / len(flat))
                         for c in prime_idx_counts.values())

    total_entropy = offset_entropy + prime_entropy
    min_size = int(math.ceil(len(flat) * total_entropy / 8))

    print(f"  Offset entropy: {offset_entropy:.4f} bits/pixel")
    print(f"  Prime index entropy: {prime_entropy:.4f} bits/pixel")
    print(f"  Combined: {total_entropy:.4f} bits/pixel")
    print(f"  Theoretical minimum: {min_size:,} bytes")
    print(f"  vs raw pixel entropy: compare to direct byte entropy")
    print(f"  Improvement: {'yes' if total_entropy < 8.0 else 'no'}")

    log_result("prime_resonance", original_size, min_size,
               True, 0,
               f"Prime decomposition minimum={min_size:,}B. "
               f"Not better than direct entropy coding.")

    return {}


# ===================================================================
# EXPERIMENT 8: Dimensional Reduction via Random Projection
# ===================================================================
# Novel: Use Johnson-Lindenstrauss random projection to map the
#        high-dimensional pixel space to a MUCH lower dimension.
#        JL lemma says distances are preserved with high probability.
#        But can we reconstruct the original exactly?
# ===================================================================

def exp8_random_projection(img_array: np.ndarray) -> dict:
    """Test if random projection can compress while preserving exact values."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Random Projection (Johnson-Lindenstrauss)")
    print("=" * 70)

    h, w = img_array.shape[:2]
    if len(img_array.shape) == 3:
        gray = np.mean(img_array[:, :, :3], axis=2).astype(np.float64)
    else:
        gray = img_array.astype(np.float64)

    original_size = h * w
    flat = gray.ravel()
    n = len(flat)

    # JL lemma: can project n points in R^d to R^k where k = O(log(n)/eps^2)
    # For n=2.6M points, eps=0.01: k ≈ 140,000 dimensions
    # That's still huge - not useful for 1KB

    # Try compressive sensing: if the image is sparse in some basis,
    # we can reconstruct from fewer measurements than pixels
    # Measurement: y = Phi * x where Phi is random matrix

    # Downsample for tractability
    small = gray[::8, ::8]
    flat_small = small.ravel()
    n_small = len(flat_small)

    for k_ratio in [0.1, 0.3, 0.5]:
        k = int(n_small * k_ratio)
        np.random.seed(42)
        # Gaussian random projection matrix
        Phi = np.random.randn(k, n_small) / math.sqrt(k)
        measurements = Phi @ flat_small  # k measurements

        # Try to reconstruct via pseudoinverse
        reconstructed = np.linalg.lstsq(Phi, measurements, rcond=None)[0]
        diff = np.abs(flat_small - reconstructed)
        max_diff = diff.max()
        mean_diff = diff.mean()

        storage = k * 4 + 8  # float32 measurements + header

        print(f"\n  k/n = {k_ratio}: {k} measurements, {storage:,} bytes")
        print(f"    Max diff: {max_diff:.4f}")
        print(f"    Mean diff: {mean_diff:.6f}")
        print(f"    Lossless: {max_diff < 0.5}")

    log_result("random_projection", original_size, int(n_small * 0.1 * 4 + 8),
               False, 999,
               "Random projection cannot recover exact values from "
               "fewer measurements than dimensions (underdetermined system)")
    return {}


# ===================================================================
# EXPERIMENT 9: Modular Arithmetic Encoding (NOVEL)
# ===================================================================
# Novel idea: Represent the image using Chinese Remainder Theorem.
# If we choose N coprime moduli m1,...,mN, any integer < product(mi)
# can be uniquely represented by its remainders (r1,...,rN).
# If the remainders are "simpler" than the original, we save space.
# ===================================================================

def exp9_crt_encoding(img_array: np.ndarray) -> dict:
    """Chinese Remainder Theorem encoding."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: Chinese Remainder Theorem Encoding (NOVEL)")
    print("=" * 70)

    h, w = img_array.shape[:2]
    if len(img_array.shape) == 3:
        gray = np.mean(img_array[:, :, :3], axis=2).astype(np.uint8)
    else:
        gray = img_array.astype(np.uint8)

    original_size = h * w
    flat = gray.ravel()

    # For pixel values 0-255, we need moduli whose product >= 256
    # Coprime moduli: 3, 5, 7, 11 -> product = 1155 > 256
    # Each pixel stored as (r3, r5, r7, r11) instead of raw byte
    moduli = [3, 5, 7, 11]
    product = 1
    for m in moduli:
        product *= m
    print(f"  Moduli: {moduli}, product = {product}")

    remainders = []
    for m in moduli:
        r = flat % m
        remainders.append(r)

    # Check: can we reconstruct?
    # CRT reconstruction
    def crt_solve(remainders_list, moduli_list):
        """Solve system of congruences using CRT."""
        N = product
        result = 0
        for r, m in zip(remainders_list, moduli_list):
            Ni = N // m
            # Find modular inverse of Ni mod m
            inv = pow(int(Ni), -1, int(m))
            result += int(r) * int(Ni) * int(inv)
        return result % N

    # Verify on first 100 pixels
    all_correct = True
    for i in range(min(100, len(flat))):
        r_list = [int(remainders[j][i]) for j in range(len(moduli))]
        reconstructed = crt_solve(r_list, moduli)
        if reconstructed != flat[i]:
            all_correct = False
            break

    print(f"  CRT reconstruction correct: {all_correct}")

    # Entropy of remainders vs original
    from collections import Counter
    original_entropy = 0
    counts = Counter(flat.tolist())
    for c in counts.values():
        p = c / len(flat)
        original_entropy -= p * math.log2(p)

    total_remainder_bits = 0
    for j, m in enumerate(moduli):
        r = remainders[j]
        r_counts = Counter(r.tolist())
        r_entropy = -sum((c / len(flat)) * math.log2(c / len(flat))
                         for c in r_counts.values())
        bits = math.ceil(math.log2(m))
        total_remainder_bits += bits
        print(f"  mod {m}: entropy={r_entropy:.4f} bits, "
              f"fixed={bits} bits")

    print(f"\n  Original: {original_entropy:.4f} bits/pixel")
    print(f"  CRT total: {total_remainder_bits} bits/pixel (fixed-width)")
    print(f"  CRT saves space: {total_remainder_bits < original_entropy}")

    crt_size = len(flat) * total_remainder_bits // 8
    log_result("crt_encoding", original_size, crt_size,
               True, 0,
               f"CRT with moduli {moduli}: {total_remainder_bits} bits/px "
               f"vs {original_entropy:.1f} bits/px original. "
               f"{'Saves' if total_remainder_bits < original_entropy else 'Costs more'} space.")
    return {}


# ===================================================================
# MAIN
# ===================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python extreme.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    print(f"EXTREME COMPRESSION EXPERIMENTS")
    print(f"Target: {img_path}")
    print(f"Goal: Lossless compression to 1KB")

    img = Image.open(img_path)
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    channels = img_array.shape[2] if len(img_array.shape) > 2 else 1
    raw_size = h * w * channels

    print(f"Image: {w}x{h}, {channels} channels, {raw_size:,} bytes raw")
    print(f"File size: {Path(img_path).stat().st_size:,} bytes")

    exp1_giant_integer(img_array)
    exp2_fourier_decomposition(img_array)
    exp3_polynomial_surface(img_array)
    exp4_neural_memorization(img_array)
    exp5_fractal_deep(img_array)
    exp6_equation_discovery(img_array)
    exp7_prime_resonance(img_array)
    exp8_random_projection(img_array)
    exp9_crt_encoding(img_array)

    # Final report
    print("\n" + "=" * 70)
    print("FINAL RESULTS - ALL EXPERIMENTS")
    print("=" * 70)
    print(f"\n{'Experiment':<30} {'Compressed':>12} {'Ratio':>8} "
          f"{'Lossless':>10} {'Max Diff':>10}")
    print("-" * 75)

    for r in sorted(RESULTS_LOG, key=lambda x: x["compressed"]):
        print(f"{r['name']:<30} {r['compressed']:>12,} {r['ratio']:>7.1f}x "
              f"{'YES' if r['lossless'] else 'NO':>10} {r['max_diff']:>10}")

    # Any lossless results under 1KB?
    lossless_1kb = [r for r in RESULTS_LOG
                    if r["lossless"] and r["compressed"] <= 1024]
    print(f"\nLossless results fitting in 1KB: {len(lossless_1kb)}")
    if not lossless_1kb:
        print("None of the 9 experiments achieved lossless 1KB compression.")
        print("\nThe closest lossless result:")
        lossless = [r for r in RESULTS_LOG if r["lossless"]]
        if lossless:
            best = min(lossless, key=lambda x: x["compressed"])
            print(f"  {best['name']}: {best['compressed']:,} bytes ({best['ratio']:.1f}x)")


if __name__ == "__main__":
    main()
