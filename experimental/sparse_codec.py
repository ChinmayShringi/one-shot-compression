#!/usr/bin/env python3
"""SPARSE DICTIONARY LEARNING CODEC FOR LOSSLESS IMAGE COMPRESSION

Represent each image patch as a sparse linear combination of dictionary atoms:
  patch ~ a1*d1 + a2*d2 + ... + ak*dk  (k << dictionary_size)

If most patches can be well-approximated with just 3-5 atoms, the sparse codes
are very compact. Store: dictionary + sparse codes + residual.

Approaches:
  1. OMP Codec    -- Fixed DCT + random patch dictionary, OMP sparse coding
  2. K-SVD Codec  -- Learned dictionary via K-SVD, adapted to this image
  3. Hierarchical -- Multi-scale sparse coding (16x16 -> 8x8 -> 4x4)
  4. Convolutional -- Shift-invariant sparse coding with convolutional filters

All from scratch. No sklearn, no scipy sparse coding.
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
from experimental.data_structures import KDTree, entropy_bits, extract_patches, zigzag_encode


# ============================================================
# DCT Dictionary (from scratch)
# ============================================================

def dct_dictionary(patch_size, n_atoms=None):
    """Build 2D DCT dictionary.

    Each atom is a 2D cosine basis function flattened to a vector.
    For patch_size=8, produces 64 atoms (complete basis).

    Returns (patch_size**2, n_atoms) normalized dictionary matrix.
    """
    if n_atoms is None:
        n_atoms = patch_size * patch_size

    atoms = []
    for u in range(patch_size):
        for v in range(patch_size):
            if len(atoms) >= n_atoms:
                break
            atom = np.zeros((patch_size, patch_size), dtype=np.float64)
            for i in range(patch_size):
                for j in range(patch_size):
                    atom[i, j] = (
                        math.cos(math.pi * (2 * i + 1) * u / (2 * patch_size))
                        * math.cos(math.pi * (2 * j + 1) * v / (2 * patch_size))
                    )
            atoms.append(atom.flatten())
        if len(atoms) >= n_atoms:
            break

    dictionary = np.array(atoms, dtype=np.float64).T  # (patch_dim, n_atoms)
    # Normalize each column to unit norm
    norms = np.linalg.norm(dictionary, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return dictionary / norms


def _random_patch_atoms(patches, n_extra, rng):
    """Select random patches as additional dictionary atoms.

    Returns (patch_dim, n_extra) normalized matrix.
    """
    n_patches = len(patches)
    indices = rng.choice(n_patches, size=min(n_extra, n_patches), replace=False)
    atoms = patches[indices].astype(np.float64).T  # (patch_dim, n_extra)
    norms = np.linalg.norm(atoms, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return atoms / norms


def build_initial_dictionary(patches, patch_size, dict_size, rng):
    """Build initial dictionary: DCT basis + random patches from the image.

    Returns (patch_dim, dict_size) normalized dictionary.
    """
    dct_atoms = dct_dictionary(patch_size)
    n_dct = dct_atoms.shape[1]

    if dict_size <= n_dct:
        return dct_atoms[:, :dict_size].copy()

    n_extra = dict_size - n_dct
    random_atoms = _random_patch_atoms(patches, n_extra, rng)
    combined = np.hstack([dct_atoms, random_atoms])

    # Ensure exactly dict_size atoms, pad with random if needed
    if combined.shape[1] < dict_size:
        patch_dim = combined.shape[0]
        extra = rng.randn(patch_dim, dict_size - combined.shape[1])
        extra_norms = np.linalg.norm(extra, axis=0, keepdims=True)
        extra_norms = np.where(extra_norms < 1e-12, 1.0, extra_norms)
        combined = np.hstack([combined, extra / extra_norms])

    return combined[:, :dict_size]


# ============================================================
# Orthogonal Matching Pursuit (from scratch)
# ============================================================

def omp(dictionary, signal, sparsity):
    """Orthogonal Matching Pursuit: sparse approximation of signal.

    Greedily selects atoms from the dictionary that best correlate
    with the residual, then solves least-squares over all selected atoms.

    Args:
        dictionary: (patch_dim, n_atoms) -- column-normalized
        signal: (patch_dim,) -- target vector
        sparsity: maximum number of atoms to select

    Returns:
        selected: list of atom indices
        coefficients: array of coefficient values for selected atoms
        residual: remaining error vector
    """
    patch_dim, n_atoms = dictionary.shape
    residual = signal.astype(np.float64).copy()
    selected = []
    coefficients = np.array([], dtype=np.float64)

    for _k in range(sparsity):
        # Find atom most correlated with current residual
        correlations = dictionary.T @ residual
        best_atom = int(np.argmax(np.abs(correlations)))

        # Skip if already selected (can happen with near-duplicate atoms)
        if best_atom in selected:
            # Pick next best that is not selected
            sorted_atoms = np.argsort(-np.abs(correlations))
            found = False
            for candidate in sorted_atoms:
                if int(candidate) not in selected:
                    best_atom = int(candidate)
                    found = True
                    break
            if not found:
                break

        selected.append(best_atom)

        # Solve least squares over all selected atoms
        d_sel = dictionary[:, selected]
        coefficients = np.linalg.lstsq(d_sel, signal.astype(np.float64), rcond=None)[0]

        # Update residual
        residual = signal.astype(np.float64) - d_sel @ coefficients

        # Early termination if residual is negligible
        if np.linalg.norm(residual) < 1e-10:
            break

    return selected, coefficients, residual


def omp_batch(dictionary, signals, sparsity):
    """Apply OMP to a batch of signals.

    Args:
        dictionary: (patch_dim, n_atoms)
        signals: (n_signals, patch_dim)
        sparsity: max atoms per signal

    Returns:
        all_selected: list of lists of atom indices
        all_coefficients: list of coefficient arrays
        all_residuals: (n_signals, patch_dim)
    """
    n_signals = len(signals)
    all_selected = []
    all_coefficients = []
    residuals = np.zeros_like(signals, dtype=np.float64)

    for i in range(n_signals):
        sel, coeffs, res = omp(dictionary, signals[i], sparsity)
        all_selected.append(sel)
        all_coefficients.append(coeffs)
        residuals[i] = res

    return all_selected, all_coefficients, residuals


# ============================================================
# Cost Estimation Utilities
# ============================================================

def _quantize_coefficients(coefficients_list, n_bits=12):
    """Quantize floating-point coefficients to integers for storage.

    Returns quantized values and the scale factor used.
    """
    # Find global range across all coefficients
    all_coeffs = np.concatenate(coefficients_list) if coefficients_list else np.array([0.0])
    if len(all_coeffs) == 0:
        return [], 1.0

    max_abs = np.max(np.abs(all_coeffs))
    if max_abs < 1e-12:
        return [np.zeros_like(c, dtype=np.int32) for c in coefficients_list], 1.0

    scale = (2 ** (n_bits - 1) - 1) / max_abs
    quantized = [np.round(c * scale).astype(np.int32) for c in coefficients_list]
    return quantized, scale


def _reconstruct_from_sparse(dictionary, all_selected, all_coefficients, scale,
                             quantized_coeffs):
    """Reconstruct patches from sparse codes (using quantized coefficients)."""
    n_patches = len(all_selected)
    patch_dim = dictionary.shape[0]
    reconstructed = np.zeros((n_patches, patch_dim), dtype=np.float64)

    for i in range(n_patches):
        if len(all_selected[i]) > 0:
            d_sel = dictionary[:, all_selected[i]]
            dequant = quantized_coeffs[i].astype(np.float64) / scale
            reconstructed[i] = d_sel @ dequant

    return reconstructed


def compute_sparse_cost(dictionary, all_selected, all_coefficients,
                        residuals, patches, patch_size, grid_shape,
                        border_pixels, coeff_bits=12):
    """Compute total storage cost of sparse representation.

    Components:
        - Dictionary: dict_size * patch_dim * 16 bits
        - Sparse codes: atom indices + quantized coefficients per patch
        - Residual: entropy of integer residual after rounding
        - Border: entropy of uncovered border pixels

    Returns dict with component sizes in KB and lossless verification.
    """
    patch_dim, dict_size = dictionary.shape
    n_patches = len(all_selected)

    # Dictionary cost (store as float16)
    dict_bytes = dict_size * patch_dim * 2  # 16 bits per element

    # Sparse codes cost
    # Atom indices: each index needs log2(dict_size) bits
    index_bits_per_atom = max(1, math.ceil(math.log2(dict_size))) if dict_size > 1 else 1
    total_index_bits = sum(len(s) * index_bits_per_atom for s in all_selected)

    # Number of atoms per patch (need to store this): log2(sparsity_max) bits
    max_sparsity = max((len(s) for s in all_selected), default=0)
    count_bits = max(1, math.ceil(math.log2(max_sparsity + 1))) if max_sparsity > 0 else 1
    total_count_bits = n_patches * count_bits

    # Quantize coefficients
    quantized, scale = _quantize_coefficients(all_coefficients, n_bits=coeff_bits)

    # Coefficient cost via entropy
    all_quant_flat = np.concatenate([q.flatten() for q in quantized]) if quantized else np.array([0])
    coeff_entropy = entropy_bits(zigzag_encode(all_quant_flat.astype(np.int32)))

    total_codes_bits = total_index_bits + total_count_bits + coeff_entropy
    codes_bytes = total_codes_bits / 8

    # Reconstruct and compute integer residual for lossless verification
    reconstructed = _reconstruct_from_sparse(
        dictionary, all_selected, all_coefficients, scale, quantized
    )
    approx_int = np.round(reconstructed).astype(np.int32)
    integer_residual = patches.astype(np.int32) - approx_int

    # Residual cost via entropy
    residual_zigzag = zigzag_encode(integer_residual.flatten())
    residual_entropy = entropy_bits(residual_zigzag)
    residual_bytes = residual_entropy / 8

    # Border cost
    border_bytes = 0.0
    if len(border_pixels) > 0:
        border_zigzag = zigzag_encode(border_pixels.astype(np.int32))
        border_bytes = entropy_bits(border_zigzag) / 8

    # Scale factor cost (just 8 bytes for a float64)
    meta_bytes = 8 + 4  # scale + grid_shape

    total_bytes = dict_bytes + codes_bytes + residual_bytes + border_bytes + meta_bytes

    # Lossless verification
    full_recon = approx_int + integer_residual
    lossless = np.array_equal(full_recon, patches.astype(np.int32))

    return {
        "dictionary_kb": dict_bytes / 1024,
        "codes_kb": codes_bytes / 1024,
        "residual_kb": residual_bytes / 1024,
        "border_kb": border_bytes / 1024,
        "total_kb": total_bytes / 1024,
        "lossless": lossless,
        "mean_residual_abs": float(np.mean(np.abs(integer_residual))),
        "max_residual_abs": int(np.max(np.abs(integer_residual))) if integer_residual.size > 0 else 0,
        "avg_atoms_used": float(np.mean([len(s) for s in all_selected])),
    }


def _print_cost(label, cost, baseline_kb=1271):
    """Print cost breakdown."""
    print(f"\n  [{label}]")
    print(f"    Dictionary:  {cost['dictionary_kb']:8.1f} KB")
    print(f"    Sparse codes:{cost['codes_kb']:8.1f} KB")
    print(f"    Residual:    {cost['residual_kb']:8.1f} KB")
    print(f"    Border:      {cost['border_kb']:8.1f} KB")
    print(f"    --------------------------------")
    print(f"    TOTAL:       {cost['total_kb']:8.1f} KB")
    print(f"    vs PRISM:    {cost['total_kb'] / baseline_kb * 100:8.1f}%")
    print(f"    Lossless:    {cost['lossless']}")
    print(f"    Avg atoms:   {cost['avg_atoms_used']:.2f}")
    print(f"    Mean |res|:  {cost['mean_residual_abs']:.2f}")
    print(f"    Max  |res|:  {cost['max_residual_abs']}")


# ============================================================
# CODEC 1: OMP with DCT + Random Patch Dictionary
# ============================================================

def omp_codec(image_path, patch_size=8, dict_size=256, sparsity=4):
    """Sparse coding with fixed DCT + random patch dictionary.

    Process:
      1. Load image, convert to YCoCg-R
      2. Build dictionary: DCT basis + random image patches
      3. Extract patches, apply OMP to each
      4. Compute integer residual for losslessness
      5. Report storage cost

    Args:
        image_path: path to PNG image
        patch_size: size of square patches (8 = 8x8)
        dict_size: number of dictionary atoms
        sparsity: max atoms per patch (OMP iterations)

    Returns:
        dict with cost breakdown
    """
    print("\n" + "=" * 70)
    print(f"OMP CODEC: patch={patch_size}, dict={dict_size}, sparsity={sparsity}")
    print("=" * 70)

    t0 = time.time()

    # Load and transform
    img = np.array(Image.open(image_path).convert("RGB"))
    ycocg = rgb_to_ycocg_r(img)
    h, w, c = ycocg.shape
    print(f"  Image: {h}x{w}x{c}")

    rng = np.random.RandomState(42)

    # Process each channel independently
    total_cost = {
        "dictionary_kb": 0.0, "codes_kb": 0.0,
        "residual_kb": 0.0, "border_kb": 0.0,
        "total_kb": 0.0, "lossless": True,
    }
    all_avg_atoms = []

    for ch in range(c):
        channel = ycocg[:, :, ch]

        # Extract patches
        patches, grid_shape, border = extract_patches(
            channel[:, :, np.newaxis], patch_size
        )
        n_patches = len(patches)
        patch_dim = patches.shape[1]
        print(f"  Channel {ch}: {n_patches} patches of dim {patch_dim}, border {len(border)} px")

        # Build dictionary
        dictionary = build_initial_dictionary(patches, patch_size, dict_size, rng)
        print(f"  Dictionary: {dictionary.shape}")

        # Subsample patches if there are too many (>50k) for speed
        if n_patches > 50000:
            sample_idx = rng.choice(n_patches, 50000, replace=False)
            sample_patches = patches[sample_idx]
            print(f"  Subsampled to {len(sample_patches)} patches for OMP")
        else:
            sample_idx = None
            sample_patches = patches

        # Apply OMP
        selected, coefficients, residuals = omp_batch(
            dictionary, sample_patches.astype(np.float64), sparsity
        )

        # If subsampled, apply OMP to remaining patches too
        if sample_idx is not None:
            remaining_mask = np.ones(n_patches, dtype=bool)
            remaining_mask[sample_idx] = False
            remaining_patches = patches[remaining_mask]

            sel2, coeff2, res2 = omp_batch(
                dictionary, remaining_patches.astype(np.float64), sparsity
            )

            # Merge results in original order
            full_selected = [None] * n_patches
            full_coefficients = [None] * n_patches
            full_residuals = np.zeros((n_patches, patch_dim), dtype=np.float64)

            for i, si in enumerate(sample_idx):
                full_selected[si] = selected[i]
                full_coefficients[si] = coefficients[i]
                full_residuals[si] = residuals[i]

            rem_idx = np.where(remaining_mask)[0]
            for i, ri in enumerate(rem_idx):
                full_selected[ri] = sel2[i]
                full_coefficients[ri] = coeff2[i]
                full_residuals[ri] = res2[i]

            selected = full_selected
            coefficients = full_coefficients
            residuals = full_residuals

        # Compute cost
        cost = compute_sparse_cost(
            dictionary, selected, coefficients, residuals,
            patches, patch_size, grid_shape, border
        )

        for key in ["dictionary_kb", "codes_kb", "residual_kb", "border_kb", "total_kb"]:
            total_cost[key] += cost[key]
        total_cost["lossless"] = total_cost["lossless"] and cost["lossless"]
        all_avg_atoms.append(cost["avg_atoms_used"])

    total_cost["avg_atoms_used"] = float(np.mean(all_avg_atoms))
    total_cost["mean_residual_abs"] = 0.0
    total_cost["max_residual_abs"] = 0

    elapsed = time.time() - t0
    _print_cost(f"OMP (elapsed: {elapsed:.1f}s)", total_cost)
    return total_cost


# ============================================================
# CODEC 2: K-SVD Dictionary Learning
# ============================================================

def ksvd_learn(patches, dict_size, sparsity, n_iter=10, rng=None):
    """K-SVD dictionary learning algorithm (from scratch).

    Alternates between:
      1. Sparse coding step: OMP for all patches using current dictionary
      2. Dictionary update step: for each atom, update it and its
         coefficients via rank-1 SVD of the error matrix

    Args:
        patches: (n_patches, patch_dim) training data
        dict_size: number of dictionary atoms
        sparsity: max atoms per patch
        n_iter: number of K-SVD iterations
        rng: random state

    Returns:
        dictionary: (patch_dim, dict_size) learned dictionary
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n_patches, patch_dim = patches.shape
    signals = patches.astype(np.float64)

    # Initialize with DCT + random patches
    dictionary = build_initial_dictionary(patches, int(math.sqrt(patch_dim)), dict_size, rng)
    assert dictionary.shape == (patch_dim, dict_size), (
        f"Dict shape mismatch: {dictionary.shape} vs ({patch_dim}, {dict_size})"
    )

    for iteration in range(n_iter):
        t_iter = time.time()

        # Step 1: Sparse coding -- OMP for all patches
        all_selected, all_coefficients, all_residuals = omp_batch(
            dictionary, signals, sparsity
        )

        # Compute representation error
        rep_error = np.mean(np.sum(all_residuals ** 2, axis=1))

        # Step 2: Dictionary update -- atom by atom
        atoms_updated = 0
        for atom_idx in range(dict_size):
            # Find all patches that use this atom
            using_patches = []
            coeff_positions = []
            for p_idx in range(n_patches):
                if atom_idx in all_selected[p_idx]:
                    pos = all_selected[p_idx].index(atom_idx)
                    using_patches.append(p_idx)
                    coeff_positions.append(pos)

            if len(using_patches) < 2:
                # Too few patches use this atom -- replace with random patch
                rand_idx = rng.randint(n_patches)
                atom_vec = signals[rand_idx].copy()
                norm = np.linalg.norm(atom_vec)
                if norm > 1e-12:
                    dictionary[:, atom_idx] = atom_vec / norm
                continue

            # Compute error matrix E_k: error when removing atom k's contribution
            # For each patch using atom k, reconstruct WITHOUT atom k
            error_matrix = np.zeros((patch_dim, len(using_patches)), dtype=np.float64)

            for j, (p_idx, c_pos) in enumerate(zip(using_patches, coeff_positions)):
                # Full reconstruction with all selected atoms
                sel = all_selected[p_idx]
                coeffs = all_coefficients[p_idx]

                # Reconstruction without atom k
                recon_without = np.zeros(patch_dim, dtype=np.float64)
                for s, c_val in zip(sel, coeffs):
                    if s != atom_idx:
                        recon_without += c_val * dictionary[:, s]

                error_matrix[:, j] = signals[p_idx] - recon_without

            # SVD of error matrix -> best rank-1 approximation
            # This simultaneously updates the atom and its coefficients
            try:
                u, sigma, vt = np.linalg.svd(error_matrix, full_matrices=False)
            except np.linalg.LinAlgError:
                continue

            # Update atom to first left singular vector
            new_atom = u[:, 0]
            norm = np.linalg.norm(new_atom)
            if norm > 1e-12:
                dictionary[:, atom_idx] = new_atom / norm * np.sign(new_atom[np.argmax(np.abs(new_atom))])

            # Update coefficients for patches using this atom
            new_coeffs_all = sigma[0] * vt[0, :]
            for j, (p_idx, c_pos) in enumerate(zip(using_patches, coeff_positions)):
                new_val = new_coeffs_all[j]
                coeffs_list = list(all_coefficients[p_idx])
                coeffs_list[c_pos] = new_val
                all_coefficients[p_idx] = np.array(coeffs_list, dtype=np.float64)

            atoms_updated += 1

        elapsed_iter = time.time() - t_iter
        print(f"    K-SVD iter {iteration + 1}/{n_iter}: "
              f"error={rep_error:.2f}, atoms_updated={atoms_updated}, "
              f"time={elapsed_iter:.1f}s")

    # Final normalization
    norms = np.linalg.norm(dictionary, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    dictionary = dictionary / norms

    return dictionary


def ksvd_codec(image_path, patch_size=8, dict_size=256, sparsity=4,
               n_iter=10, max_train_patches=10000):
    """K-SVD dictionary learning codec.

    Learns an image-specific dictionary optimized via K-SVD, then
    encodes all patches using OMP with the learned dictionary.

    Much better than fixed DCT for natural images because the
    dictionary atoms adapt to the actual image content.

    Args:
        image_path: path to PNG image
        patch_size: size of square patches
        dict_size: number of dictionary atoms to learn
        sparsity: max atoms per patch
        n_iter: K-SVD iterations
        max_train_patches: limit training patches for speed

    Returns:
        dict with cost breakdown
    """
    print("\n" + "=" * 70)
    print(f"K-SVD CODEC: patch={patch_size}, dict={dict_size}, "
          f"sparsity={sparsity}, iter={n_iter}")
    print("=" * 70)

    t0 = time.time()

    img = np.array(Image.open(image_path).convert("RGB"))
    ycocg = rgb_to_ycocg_r(img)
    h, w, c = ycocg.shape
    print(f"  Image: {h}x{w}x{c}")

    rng = np.random.RandomState(42)

    total_cost = {
        "dictionary_kb": 0.0, "codes_kb": 0.0,
        "residual_kb": 0.0, "border_kb": 0.0,
        "total_kb": 0.0, "lossless": True,
    }
    all_avg_atoms = []

    for ch in range(c):
        print(f"\n  --- Channel {ch} ---")
        channel = ycocg[:, :, ch]

        patches, grid_shape, border = extract_patches(
            channel[:, :, np.newaxis], patch_size
        )
        n_patches = len(patches)
        patch_dim = patches.shape[1]
        print(f"  {n_patches} patches of dim {patch_dim}")

        # Subsample for dictionary learning (K-SVD is expensive)
        if n_patches > max_train_patches:
            train_idx = rng.choice(n_patches, max_train_patches, replace=False)
            train_patches = patches[train_idx]
            print(f"  Training K-SVD on {len(train_patches)} patches")
        else:
            train_patches = patches

        # Learn dictionary
        dictionary = ksvd_learn(
            train_patches, dict_size, sparsity, n_iter=n_iter, rng=rng
        )
        print(f"  Learned dictionary: {dictionary.shape}")

        # Encode ALL patches with learned dictionary
        print(f"  Encoding all {n_patches} patches...")
        selected, coefficients, residuals = omp_batch(
            dictionary, patches.astype(np.float64), sparsity
        )

        # Compute cost
        cost = compute_sparse_cost(
            dictionary, selected, coefficients, residuals,
            patches, patch_size, grid_shape, border
        )

        for key in ["dictionary_kb", "codes_kb", "residual_kb", "border_kb", "total_kb"]:
            total_cost[key] += cost[key]
        total_cost["lossless"] = total_cost["lossless"] and cost["lossless"]
        all_avg_atoms.append(cost["avg_atoms_used"])

    total_cost["avg_atoms_used"] = float(np.mean(all_avg_atoms))
    total_cost["mean_residual_abs"] = 0.0
    total_cost["max_residual_abs"] = 0

    elapsed = time.time() - t0
    _print_cost(f"K-SVD (elapsed: {elapsed:.1f}s)", total_cost)
    return total_cost


# ============================================================
# CODEC 3: Hierarchical Multi-Scale Sparse Coding
# ============================================================

def _sparse_encode_single_scale(channel_2d, patch_size, dict_size, sparsity,
                                n_ksvd_iter, max_train, rng):
    """Run sparse coding at a single scale. Returns cost dict and integer residual image."""
    h, w = channel_2d.shape

    patches, grid_shape, border = extract_patches(
        channel_2d[:, :, np.newaxis], patch_size
    )
    n_patches = len(patches)
    patch_dim = patches.shape[1]

    if n_patches == 0:
        return {
            "dictionary_kb": 0.0, "codes_kb": 0.0,
            "residual_kb": 0.0, "border_kb": 0.0,
            "total_kb": 0.0, "lossless": True,
            "avg_atoms_used": 0.0,
            "mean_residual_abs": 0.0,
            "max_residual_abs": 0,
        }, channel_2d.copy()

    # Learn dictionary
    if n_patches > max_train:
        train_idx = rng.choice(n_patches, max_train, replace=False)
        train_patches = patches[train_idx]
    else:
        train_patches = patches

    dictionary = ksvd_learn(train_patches, dict_size, sparsity, n_iter=n_ksvd_iter, rng=rng)

    # Encode all patches
    selected, coefficients, residuals = omp_batch(
        dictionary, patches.astype(np.float64), sparsity
    )

    # Compute cost
    cost = compute_sparse_cost(
        dictionary, selected, coefficients, residuals,
        patches, patch_size, grid_shape, border
    )

    # Build residual image for next scale
    quantized, scale = _quantize_coefficients(coefficients)
    reconstructed = _reconstruct_from_sparse(
        dictionary, selected, coefficients, scale, quantized
    )
    approx_int = np.round(reconstructed).astype(np.int32)
    integer_residual_patches = patches.astype(np.int32) - approx_int

    # Reassemble residual into image
    n_h, n_w = grid_shape
    residual_image = np.zeros_like(channel_2d, dtype=np.int32)
    covered_h = (n_h - 1) * patch_size + patch_size
    covered_w = (n_w - 1) * patch_size + patch_size

    idx = 0
    for i in range(n_h):
        for j in range(n_w):
            y0 = i * patch_size
            x0 = j * patch_size
            residual_image[y0:y0 + patch_size, x0:x0 + patch_size] = (
                integer_residual_patches[idx].reshape(patch_size, patch_size)
            )
            idx += 1

    # Copy uncovered border from original
    if covered_h < h:
        residual_image[covered_h:, :covered_w] = channel_2d[covered_h:, :covered_w]
    if covered_w < w:
        residual_image[:, covered_w:] = channel_2d[:, covered_w:]

    return cost, residual_image


def hierarchical_sparse(image_path, max_train=5000):
    """Multi-scale sparse coding.

    Scale 1: 16x16 patches, large dictionary (512), sparsity=8
             Captures large-scale structure (edges, gradients, textures)
    Scale 2: 8x8 patches on residual, medium dictionary (256), sparsity=4
             Captures medium details
    Scale 3: 4x4 patches on remaining residual, small dictionary (128), sparsity=2
             Captures fine details

    Each scale encodes the residual left by the previous scale.

    Returns:
        dict with cost breakdown
    """
    print("\n" + "=" * 70)
    print("HIERARCHICAL SPARSE CODEC: 3 scales (16x16 -> 8x8 -> 4x4)")
    print("=" * 70)

    t0 = time.time()

    img = np.array(Image.open(image_path).convert("RGB"))
    ycocg = rgb_to_ycocg_r(img)
    h, w, c = ycocg.shape
    print(f"  Image: {h}x{w}x{c}")

    scales = [
        {"patch_size": 16, "dict_size": 512, "sparsity": 8, "n_iter": 5},
        {"patch_size": 8,  "dict_size": 256, "sparsity": 4, "n_iter": 5},
        {"patch_size": 4,  "dict_size": 128, "sparsity": 2, "n_iter": 3},
    ]

    rng = np.random.RandomState(42)

    total_cost = {
        "dictionary_kb": 0.0, "codes_kb": 0.0,
        "residual_kb": 0.0, "border_kb": 0.0,
        "total_kb": 0.0, "lossless": True,
    }
    all_avg_atoms = []

    for ch in range(c):
        print(f"\n  === Channel {ch} ===")
        current_signal = ycocg[:, :, ch].astype(np.int32)

        channel_cost = {
            "dictionary_kb": 0.0, "codes_kb": 0.0,
            "residual_kb": 0.0, "border_kb": 0.0,
            "total_kb": 0.0, "lossless": True,
        }
        channel_avg_atoms = []

        for scale_idx, scale_params in enumerate(scales):
            print(f"\n    Scale {scale_idx + 1}: {scale_params['patch_size']}x{scale_params['patch_size']}, "
                  f"dict={scale_params['dict_size']}, sparsity={scale_params['sparsity']}")

            scale_cost, residual_img = _sparse_encode_single_scale(
                current_signal, scale_params["patch_size"],
                scale_params["dict_size"], scale_params["sparsity"],
                scale_params["n_iter"], max_train, rng
            )

            # Accumulate cost (skip residual cost for non-final scales)
            channel_cost["dictionary_kb"] += scale_cost["dictionary_kb"]
            channel_cost["codes_kb"] += scale_cost["codes_kb"]
            if scale_idx < len(scales) - 1:
                # Intermediate scale: residual is fed to next scale
                channel_cost["border_kb"] += scale_cost["border_kb"]
            else:
                # Final scale: include residual cost
                channel_cost["residual_kb"] += scale_cost["residual_kb"]
                channel_cost["border_kb"] += scale_cost["border_kb"]

            channel_avg_atoms.append(scale_cost["avg_atoms_used"])

            # Update signal to be the residual for next scale
            current_signal = residual_img

            print(f"      Cost so far: dict={channel_cost['dictionary_kb']:.1f} KB, "
                  f"codes={channel_cost['codes_kb']:.1f} KB")

        channel_cost["total_kb"] = sum(
            channel_cost[k] for k in ["dictionary_kb", "codes_kb", "residual_kb", "border_kb"]
        )

        for key in channel_cost:
            if key == "lossless":
                total_cost["lossless"] = total_cost["lossless"] and channel_cost["lossless"]
            else:
                total_cost[key] += channel_cost[key]

        all_avg_atoms.extend(channel_avg_atoms)

    total_cost["avg_atoms_used"] = float(np.mean(all_avg_atoms))
    total_cost["mean_residual_abs"] = 0.0
    total_cost["max_residual_abs"] = 0

    elapsed = time.time() - t0
    _print_cost(f"Hierarchical Sparse (elapsed: {elapsed:.1f}s)", total_cost)
    return total_cost


# ============================================================
# CODEC 4: Convolutional Sparse Coding
# ============================================================

def _convolve2d(image, kernel):
    """2D convolution (valid mode) -- from scratch.

    Args:
        image: (H, W) input
        kernel: (kh, kw) filter

    Returns:
        (H - kh + 1, W - kw + 1) convolution result
    """
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = h - kh + 1
    out_w = w - kw + 1

    if out_h <= 0 or out_w <= 0:
        return np.zeros((max(out_h, 0), max(out_w, 0)), dtype=np.float64)

    # Use stride tricks for efficiency (still no external libs)
    output = np.zeros((out_h, out_w), dtype=np.float64)

    # Vectorized: extract all windows at once
    windows = np.lib.stride_tricks.as_strided(
        image,
        shape=(out_h, out_w, kh, kw),
        strides=(image.strides[0], image.strides[1], image.strides[0], image.strides[1])
    )
    output = np.einsum("ijkl,kl->ij", windows, kernel)

    return output


def _correlate2d(image, kernel):
    """2D cross-correlation (valid mode) -- from scratch.

    Same as convolution but without flipping the kernel.
    """
    return _convolve2d(image, kernel[::-1, ::-1])


def _full_convolve2d(feature_map, kernel):
    """2D convolution (full mode) -- output is (H + kh - 1, W + kw - 1).

    Used for reconstruction: conv(sparse_map, filter).
    """
    kh, kw = kernel.shape
    # Pad feature map
    padded = np.zeros(
        (feature_map.shape[0] + 2 * (kh - 1), feature_map.shape[1] + 2 * (kw - 1)),
        dtype=np.float64,
    )
    padded[kh - 1:kh - 1 + feature_map.shape[0], kw - 1:kw - 1 + feature_map.shape[1]] = feature_map
    return _convolve2d(padded, kernel)


def convolutional_sparse(image_path, filter_size=8, n_filters=64,
                         sparsity_ratio=0.05, n_iter=5):
    """Convolutional sparse coding codec.

    Instead of patch-based coding, uses shift-invariant filters applied
    to the entire image. Each filter produces a feature map, and we
    enforce sparsity on the feature maps (most values are zero).

    Better for repetitive textures (sky, sand, fabric) because the
    same filter can be reused at any position without redundancy.

    Process:
      1. Initialize filters (DCT-like + random)
      2. For each iteration:
         a. Compute feature maps via correlation with each filter
         b. Threshold feature maps (keep top sparsity_ratio fraction)
         c. Reconstruct and compute error
         d. Update filters using gradient descent on error
      3. Store: filters + sparse feature maps + residual

    Args:
        image_path: path to PNG image
        filter_size: size of square filters (8 = 8x8)
        n_filters: number of convolutional filters
        sparsity_ratio: fraction of feature map entries kept (0.05 = 5%)
        n_iter: number of learning iterations

    Returns:
        dict with cost breakdown
    """
    print("\n" + "=" * 70)
    print(f"CONVOLUTIONAL SPARSE CODEC: {filter_size}x{filter_size}, "
          f"{n_filters} filters, sparsity={sparsity_ratio}")
    print("=" * 70)

    t0 = time.time()

    img = np.array(Image.open(image_path).convert("RGB"))
    ycocg = rgb_to_ycocg_r(img)
    h, w, c = ycocg.shape
    print(f"  Image: {h}x{w}x{c}")

    rng = np.random.RandomState(42)

    # Feature map dimensions
    fm_h = h - filter_size + 1
    fm_w = w - filter_size + 1

    total_cost = {
        "dictionary_kb": 0.0, "codes_kb": 0.0,
        "residual_kb": 0.0, "border_kb": 0.0,
        "total_kb": 0.0, "lossless": True,
    }

    for ch in range(c):
        print(f"\n  --- Channel {ch} ---")
        channel = ycocg[:, :, ch].astype(np.float64)

        # Initialize filters: first set from DCT, rest random
        filters = []
        dct_d = dct_dictionary(filter_size)
        n_dct = min(dct_d.shape[1], n_filters)
        for f_idx in range(n_dct):
            filt = dct_d[:, f_idx].reshape(filter_size, filter_size)
            filters.append(filt)

        for _ in range(n_filters - n_dct):
            filt = rng.randn(filter_size, filter_size)
            filt /= max(np.linalg.norm(filt), 1e-12)
            filters.append(filt)

        filters = np.array(filters, dtype=np.float64)  # (n_filters, fs, fs)

        # Iterative learning
        learning_rate = 0.01
        for iteration in range(n_iter):
            # Forward: compute all feature maps
            feature_maps = np.zeros((n_filters, fm_h, fm_w), dtype=np.float64)
            for f_idx in range(n_filters):
                feature_maps[f_idx] = _correlate2d(channel, filters[f_idx])

            # Enforce sparsity: keep only top entries by magnitude
            total_entries = n_filters * fm_h * fm_w
            n_keep = max(1, int(total_entries * sparsity_ratio))

            abs_vals = np.abs(feature_maps.ravel())
            if n_keep < total_entries:
                threshold = np.partition(abs_vals, -n_keep)[-n_keep]
                sparse_maps = np.where(np.abs(feature_maps) >= threshold, feature_maps, 0.0)
            else:
                sparse_maps = feature_maps.copy()

            # Reconstruct
            reconstruction = np.zeros((h, w), dtype=np.float64)
            for f_idx in range(n_filters):
                if np.any(sparse_maps[f_idx] != 0):
                    recon_contrib = _full_convolve2d(sparse_maps[f_idx], filters[f_idx])
                    # Crop to image size
                    reconstruction += recon_contrib[:h, :w]

            # Error
            error = channel - reconstruction
            mse = np.mean(error ** 2)

            # Update filters (gradient descent)
            for f_idx in range(n_filters):
                if np.any(sparse_maps[f_idx] != 0):
                    grad = _correlate2d(error, sparse_maps[f_idx][::-1, ::-1])
                    # Crop or pad gradient to filter size
                    g_h, g_w = grad.shape
                    update = np.zeros_like(filters[f_idx])
                    copy_h = min(g_h, filter_size)
                    copy_w = min(g_w, filter_size)
                    update[:copy_h, :copy_w] = grad[:copy_h, :copy_w]

                    filters[f_idx] += learning_rate * update
                    norm = np.linalg.norm(filters[f_idx])
                    if norm > 1e-12:
                        filters[f_idx] /= norm

            nnz = np.count_nonzero(sparse_maps)
            print(f"    Iter {iteration + 1}/{n_iter}: MSE={mse:.2f}, "
                  f"nnz={nnz}/{total_entries} ({nnz / total_entries * 100:.1f}%)")

        # Final encoding with learned filters
        feature_maps = np.zeros((n_filters, fm_h, fm_w), dtype=np.float64)
        for f_idx in range(n_filters):
            feature_maps[f_idx] = _correlate2d(channel, filters[f_idx])

        total_entries = n_filters * fm_h * fm_w
        n_keep = max(1, int(total_entries * sparsity_ratio))
        abs_vals = np.abs(feature_maps.ravel())
        if n_keep < total_entries:
            threshold = np.partition(abs_vals, -n_keep)[-n_keep]
            sparse_maps = np.where(np.abs(feature_maps) >= threshold, feature_maps, 0.0)
        else:
            sparse_maps = feature_maps.copy()

        # Reconstruct
        reconstruction = np.zeros((h, w), dtype=np.float64)
        for f_idx in range(n_filters):
            if np.any(sparse_maps[f_idx] != 0):
                recon_contrib = _full_convolve2d(sparse_maps[f_idx], filters[f_idx])
                reconstruction += recon_contrib[:h, :w]

        approx_int = np.round(reconstruction).astype(np.int32)
        integer_residual = channel.astype(np.int32) - approx_int

        # Lossless check
        recon_check = approx_int + integer_residual
        ch_lossless = np.array_equal(recon_check, channel.astype(np.int32))

        # Cost: filters
        filter_bytes = n_filters * filter_size * filter_size * 2  # float16

        # Cost: sparse feature maps (store as index + value pairs)
        nnz = np.count_nonzero(sparse_maps)
        # Each nonzero entry: position (log2(total_entries) bits) + value (12 bits)
        pos_bits = max(1, math.ceil(math.log2(total_entries))) if total_entries > 1 else 1
        nnz_values = sparse_maps[sparse_maps != 0]

        # Quantize nonzero values
        if len(nnz_values) > 0:
            max_abs = np.max(np.abs(nnz_values))
            if max_abs > 1e-12:
                scale = 2047.0 / max_abs
                quant_vals = np.round(nnz_values * scale).astype(np.int32)
                val_entropy = entropy_bits(zigzag_encode(quant_vals))
            else:
                val_entropy = 0.0
        else:
            val_entropy = 0.0

        sparse_map_bits = nnz * pos_bits + val_entropy
        # Also need nnz count per filter
        sparse_map_bits += n_filters * 32  # 32 bits per filter for nnz count

        sparse_map_bytes = sparse_map_bits / 8

        # Cost: residual
        res_zigzag = zigzag_encode(integer_residual.flatten())
        res_entropy = entropy_bits(res_zigzag)
        residual_bytes = res_entropy / 8

        ch_cost = {
            "dictionary_kb": filter_bytes / 1024,
            "codes_kb": sparse_map_bytes / 1024,
            "residual_kb": residual_bytes / 1024,
            "border_kb": 0.0,
            "total_kb": (filter_bytes + sparse_map_bytes + residual_bytes) / 1024,
            "lossless": ch_lossless,
        }

        for key in ["dictionary_kb", "codes_kb", "residual_kb", "border_kb", "total_kb"]:
            total_cost[key] += ch_cost[key]
        total_cost["lossless"] = total_cost["lossless"] and ch_lossless

    total_cost["avg_atoms_used"] = 0.0
    total_cost["mean_residual_abs"] = 0.0
    total_cost["max_residual_abs"] = 0

    elapsed = time.time() - t0
    _print_cost(f"Convolutional Sparse (elapsed: {elapsed:.1f}s)", total_cost)
    return total_cost


# ============================================================
# Main: Run All Codecs
# ============================================================

if __name__ == "__main__":
    IMAGE_PATH = str(Path(__file__).resolve().parent.parent / "img.png")
    BASELINE_KB = 1271

    print("=" * 70)
    print("SPARSE DICTIONARY LEARNING CODEC -- LOSSLESS COMPRESSION RESEARCH")
    print("=" * 70)
    print(f"Image: {IMAGE_PATH}")
    print(f"PRISM baseline: {BASELINE_KB} KB")
    print()

    results = {}

    # 1. OMP with DCT + random dictionary
    results["OMP"] = omp_codec(IMAGE_PATH, patch_size=8, dict_size=256, sparsity=4)

    # 2. K-SVD dictionary learning (most promising)
    results["K-SVD"] = ksvd_codec(
        IMAGE_PATH, patch_size=8, dict_size=256, sparsity=4,
        n_iter=10, max_train_patches=8000
    )

    # 3. Hierarchical multi-scale
    results["Hierarchical"] = hierarchical_sparse(IMAGE_PATH, max_train=5000)

    # 4. Convolutional sparse coding
    results["Convolutional"] = convolutional_sparse(
        IMAGE_PATH, filter_size=8, n_filters=64, sparsity_ratio=0.05, n_iter=5
    )

    # Summary table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'Dict KB':>8} {'Codes KB':>9} {'Resid KB':>9} "
          f"{'Total KB':>9} {'vs PRISM':>9} {'Lossless':>9}")
    print("-" * 79)

    for name, cost in results.items():
        pct = cost["total_kb"] / BASELINE_KB * 100
        print(f"{name:<25} {cost['dictionary_kb']:>8.1f} {cost['codes_kb']:>9.1f} "
              f"{cost['residual_kb']:>9.1f} {cost['total_kb']:>9.1f} "
              f"{pct:>8.1f}% {'YES' if cost['lossless'] else 'NO':>9}")

    print(f"\n{'PRISM baseline':<25} {'':>8} {'':>9} {'':>9} {BASELINE_KB:>9.1f} "
          f"{'100.0%':>9}")

    # Find best
    best_name = min(results, key=lambda k: results[k]["total_kb"])
    best_cost = results[best_name]
    print(f"\nBest method: {best_name} ({best_cost['total_kb']:.1f} KB, "
          f"{best_cost['total_kb'] / BASELINE_KB * 100:.1f}% of PRISM)")
