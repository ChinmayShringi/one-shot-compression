#!/usr/bin/env python3
"""VECTOR QUANTIZATION CODEC -- Lossless Image Compression Research

Four VQ approaches, all built from scratch (no sklearn, no scipy):

  1. Basic VQ:        K-means++ codebook + indices + residual
  2. Hierarchical VQ: Multi-resolution (16x16 coarse -> 8x8 fine -> 4x4 detail)
  3. Residual VQ:     Multi-stage cascade (encode residual of residual)
  4. Product VQ:      Split vectors into subvectors, quantize independently

All approaches guarantee lossless reconstruction via explicit residual coding.
The question is: how small can we make (codebook + indices + residual)?

Image: img.png (1630x1626 RGB beach photo, ~7.9M pixels)
Color space: YCoCg-R (reversible, integer-exact)
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# -- Project imports --
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prism.transforms import rgb_to_ycocg_r
from experimental.data_structures import KDTree, entropy_bits, extract_patches, zigzag_encode


# ============================================================
# K-MEANS++ FROM SCRATCH
# ============================================================

def kmeans_pp(data, k, max_iter=50, seed=42):
    """K-means++ clustering from scratch.

    Initialization: distance-weighted probability sampling.
    Assignment: nearest centroid by L2 distance.
    Update: mean of assigned points, rounded to int32 for lossless math.

    Args:
        data: (n, d) int32 array of patch vectors
        k: number of clusters
        max_iter: maximum iterations
        seed: random seed for reproducibility

    Returns:
        centroids: (k, d) int32 array
        labels: (n,) int array of cluster assignments
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    data_f = data.astype(np.float64)

    # -- Initialize centroids with k-means++ --
    centroids = [data_f[rng.randint(n)].copy()]
    for _ in range(1, k):
        dists = np.min(
            [np.sum((data_f - c) ** 2, axis=1) for c in centroids],
            axis=0,
        )
        probs = dists / dists.sum()
        centroids.append(data_f[rng.choice(n, p=probs)].copy())
    centroids = np.array(centroids)

    # -- Iterate: assign + update --
    for iteration in range(max_iter):
        # Compute distances from each point to each centroid
        # Shape: (k, n) -- memory-efficient chunked approach for large data
        dists = np.empty((k, n), dtype=np.float64)
        for i, c in enumerate(centroids):
            dists[i] = np.sum((data_f - c) ** 2, axis=1)
        labels = dists.argmin(axis=0)

        # Update centroids
        new_centroids = np.empty_like(centroids)
        for i in range(k):
            mask = labels == i
            if mask.any():
                new_centroids[i] = data_f[mask].mean(axis=0)
            else:
                new_centroids[i] = centroids[i]

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return np.round(centroids).astype(np.int32), labels


# ============================================================
# FAST CODEBOOK LOOKUP VIA KD-TREE
# ============================================================

def build_codebook_tree(centroids):
    """Build a KD-tree over codebook centroids for fast lookup.

    Returns (tree, centroids) pair.
    """
    return KDTree(centroids.astype(np.float64)), centroids


def assign_with_tree(tree, centroids, patches):
    """Assign each patch to nearest centroid using KD-tree.

    Returns:
        labels: (n,) indices into centroids
        reconstructed: (n, d) centroid values for each patch
    """
    n = len(patches)
    labels = np.empty(n, dtype=np.int32)
    for i in range(n):
        _, idx = tree.nearest(patches[i].astype(np.float64), k=1)
        labels[i] = idx[0]
    return labels, centroids[labels]


# ============================================================
# COST ESTIMATION UTILITIES
# ============================================================

def bits_to_kb(bits):
    """Convert bits to KB."""
    return bits / 8.0 / 1024.0


def compute_vq_cost(codebook, labels, residual, border, name="VQ"):
    """Compute and print a detailed cost breakdown.

    All costs estimated via Shannon entropy (lower bound for any
    entropy coder like arithmetic coding or ANS).

    Returns dict with cost details.
    """
    # Codebook: stored raw (could be compressed, but small relative to image)
    codebook_flat = codebook.flatten().astype(np.int32)
    codebook_bits = entropy_bits(zigzag_encode(codebook_flat))

    # Indices: entropy of label sequence
    indices_bits = entropy_bits(labels)

    # Residual: zigzag-encode signed residuals, then entropy
    residual_flat = residual.flatten().astype(np.int32)
    residual_zz = zigzag_encode(residual_flat)
    residual_bits = entropy_bits(residual_zz)

    # Border pixels (not covered by patches)
    border_bits = 0.0
    if len(border) > 0:
        border_zz = zigzag_encode(border.astype(np.int32))
        border_bits = entropy_bits(border_zz)

    total_bits = codebook_bits + indices_bits + residual_bits + border_bits
    total_kb = bits_to_kb(total_bits)

    print(f"\n  [{name}] Cost Breakdown:")
    print(f"    Codebook:   {bits_to_kb(codebook_bits):8.2f} KB  "
          f"({codebook.shape[0]} entries x {codebook.shape[1]}d)")
    print(f"    Indices:    {bits_to_kb(indices_bits):8.2f} KB  "
          f"({len(labels)} labels, {len(np.unique(labels))} unique)")
    print(f"    Residual:   {bits_to_kb(residual_bits):8.2f} KB  "
          f"(range [{residual_flat.min()}, {residual_flat.max()}])")
    if border_bits > 0:
        print(f"    Border:     {bits_to_kb(border_bits):8.2f} KB  "
              f"({len(border)} values)")
    print(f"    ----------------------------------------")
    print(f"    TOTAL:      {total_kb:8.2f} KB  ({total_bits / 8:.0f} bytes)")

    return {
        "name": name,
        "codebook_kb": bits_to_kb(codebook_bits),
        "indices_kb": bits_to_kb(indices_bits),
        "residual_kb": bits_to_kb(residual_bits),
        "border_kb": bits_to_kb(border_bits),
        "total_kb": total_kb,
        "total_bytes": int(total_bits / 8),
    }


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image_ycocg(image_path):
    """Load image, convert RGBA->RGB if needed, apply YCoCg-R transform.

    Returns (H, W, 3) int32 array in YCoCg-R color space.
    """
    img = Image.open(image_path)
    arr = np.array(img)

    # Handle RGBA by dropping alpha channel
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    print(f"  Image: {image_path}")
    print(f"  Shape: {arr.shape} ({arr.shape[0]}x{arr.shape[1]}, {arr.shape[2]}ch)")
    print(f"  Applying YCoCg-R color transform...")

    ycocg = rgb_to_ycocg_r(arr)
    print(f"  YCoCg-R range: Y[{ycocg[:,:,0].min()},{ycocg[:,:,0].max()}] "
          f"Co[{ycocg[:,:,1].min()},{ycocg[:,:,1].max()}] "
          f"Cg[{ycocg[:,:,2].min()},{ycocg[:,:,2].max()}]")

    return ycocg


# ============================================================
# 1. BASIC VQ
# ============================================================

def basic_vq(image_path, patch_size=4, n_clusters=256):
    """Basic Vector Quantization with k-means++ codebook.

    Pipeline:
      1. Load image, convert to YCoCg-R
      2. Extract non-overlapping patches (patch_size x patch_size x 3)
      3. Run k-means++ to build codebook of n_clusters centroids
      4. Assign each patch to nearest centroid (via KD-tree)
      5. Compute residual = original - reconstructed
      6. Verify lossless: original == reconstructed + residual
      7. Report entropy-based cost

    Args:
        image_path: path to input image
        patch_size: side length of square patches
        n_clusters: codebook size

    Returns:
        dict with total_bytes and cost breakdown
    """
    print("\n" + "=" * 60)
    print("  BASIC VQ")
    print(f"  patch_size={patch_size}, n_clusters={n_clusters}")
    print("=" * 60)

    ycocg = load_image_ycocg(image_path)
    patches, (n_h, n_w), border = extract_patches(ycocg, patch_size)
    dim = patches.shape[1]
    print(f"  Patches: {len(patches)} x {dim}d  (grid {n_h}x{n_w})")
    print(f"  Border pixels: {len(border)}")

    # Build codebook
    t0 = time.time()
    print(f"\n  Running k-means++ (k={n_clusters})...")
    centroids, labels_train = kmeans_pp(patches, n_clusters, max_iter=50)
    t_kmeans = time.time() - t0
    print(f"  K-means++ converged in {t_kmeans:.1f}s")

    # Build KD-tree for fast lookup (and verify assignments match)
    tree, _ = build_codebook_tree(centroids)
    labels, reconstructed = assign_with_tree(tree, centroids, patches)

    # Residual
    residual = patches - reconstructed

    # Lossless verification
    recon_check = reconstructed + residual
    assert np.array_equal(patches, recon_check), "LOSSLESS CHECK FAILED"
    print("  Lossless verification: PASSED")

    # Residual statistics
    nonzero_frac = np.count_nonzero(residual) / residual.size
    print(f"  Residual stats: {nonzero_frac * 100:.1f}% nonzero, "
          f"MAE={np.abs(residual).mean():.2f}, "
          f"max|r|={np.abs(residual).max()}")

    result = compute_vq_cost(centroids, labels, residual, border, "Basic VQ")
    return result


# ============================================================
# 2. HIERARCHICAL VQ (Multi-Resolution)
# ============================================================

def hierarchical_vq(image_path):
    """Hierarchical VQ: coarse-to-fine multi-resolution coding.

    Pipeline:
      Level 0 (coarse): 16x16 patches, 64 centroids
      Level 1 (fine):    8x8 patches on the residual, 128 centroids
      Level 2 (detail):  4x4 patches on the remaining residual, 256 centroids

    Each level encodes the RESIDUAL from the previous level's
    reconstruction, progressively capturing finer detail.

    This mirrors how JPEG2000 and modern codecs use multi-scale
    decompositions, but with VQ instead of wavelets.

    Returns:
        dict with total_bytes and cost breakdown
    """
    print("\n" + "=" * 60)
    print("  HIERARCHICAL VQ (Multi-Resolution)")
    print("  Levels: 16x16 (coarse) -> 8x8 (fine) -> 4x4 (detail)")
    print("=" * 60)

    ycocg = load_image_ycocg(image_path)
    h, w, c = ycocg.shape

    levels = [
        {"patch_size": 16, "n_clusters": 64,  "name": "L0-coarse-16x16"},
        {"patch_size": 8,  "n_clusters": 128, "name": "L1-fine-8x8"},
        {"patch_size": 4,  "n_clusters": 256, "name": "L2-detail-4x4"},
    ]

    total_cost = {
        "name": "Hierarchical VQ",
        "codebook_kb": 0.0,
        "indices_kb": 0.0,
        "residual_kb": 0.0,
        "border_kb": 0.0,
        "total_kb": 0.0,
        "total_bytes": 0,
    }

    # Track the full-resolution residual image
    current_residual_image = ycocg.copy()
    all_borders = []

    for level_idx, level in enumerate(levels):
        ps = level["patch_size"]
        nk = level["n_clusters"]
        name = level["name"]

        print(f"\n  --- Level {level_idx}: {name} ---")

        # Extract patches from the current residual image
        patches, (n_h, n_w), border = extract_patches(current_residual_image, ps)
        dim = patches.shape[1]
        print(f"  Patches: {len(patches)} x {dim}d  (grid {n_h}x{n_w})")

        # Build codebook
        t0 = time.time()
        centroids, _ = kmeans_pp(patches, nk, max_iter=30, seed=42 + level_idx)
        t_km = time.time() - t0
        print(f"  K-means++ ({nk} clusters): {t_km:.1f}s")

        # Assign via KD-tree
        tree, _ = build_codebook_tree(centroids)
        labels, reconstructed = assign_with_tree(tree, centroids, patches)

        # Compute residual
        residual = patches - reconstructed

        # Lossless check for this level
        assert np.array_equal(patches, reconstructed + residual)

        # Report cost for this level
        level_cost = compute_vq_cost(centroids, labels, residual, border, name)

        # Accumulate total cost (only last level's residual + border matter)
        total_cost["codebook_kb"] += level_cost["codebook_kb"]
        total_cost["indices_kb"] += level_cost["indices_kb"]

        # Reconstruct the image from patches at this resolution
        covered_h = (n_h - 1) * ps + ps
        covered_w = (n_w - 1) * ps + ps
        recon_image = np.zeros_like(current_residual_image)

        # Place reconstructed patches back into image
        idx = 0
        for i in range(n_h):
            for j in range(n_w):
                y0, x0 = i * ps, j * ps
                patch_2d = reconstructed[idx].reshape(ps, ps, c)
                recon_image[y0:y0 + ps, x0:x0 + ps] = patch_2d
                idx += 1

        # The residual for the next level is what remains
        new_residual_image = current_residual_image.copy()
        new_residual_image[:covered_h, :covered_w] = (
            current_residual_image[:covered_h, :covered_w] -
            recon_image[:covered_h, :covered_w]
        )
        current_residual_image = new_residual_image

        # Save border for this level
        if len(border) > 0:
            all_borders.append(border)

    # Final residual: whatever is left after all levels
    final_residual_flat = current_residual_image.flatten().astype(np.int32)
    final_residual_zz = zigzag_encode(final_residual_flat)
    final_residual_bits = entropy_bits(final_residual_zz)
    final_residual_kb = bits_to_kb(final_residual_bits)

    # Combine all borders
    all_border_vals = np.concatenate(all_borders) if all_borders else np.array([], dtype=np.int32)
    border_bits = 0.0
    if len(all_border_vals) > 0:
        border_bits = entropy_bits(zigzag_encode(all_border_vals.astype(np.int32)))

    total_cost["residual_kb"] = final_residual_kb
    total_cost["border_kb"] = bits_to_kb(border_bits)
    total_cost["total_kb"] = (
        total_cost["codebook_kb"] + total_cost["indices_kb"] +
        total_cost["residual_kb"] + total_cost["border_kb"]
    )
    total_cost["total_bytes"] = int(total_cost["total_kb"] * 1024)

    # Verify lossless reconstruction of the full pipeline
    # Reconstruct from all levels by summing contributions
    print(f"\n  Final residual image: "
          f"nonzero={np.count_nonzero(current_residual_image)}, "
          f"range [{current_residual_image.min()}, {current_residual_image.max()}]")

    print(f"\n  [Hierarchical VQ] TOTAL Cost:")
    print(f"    Codebooks:    {total_cost['codebook_kb']:8.2f} KB")
    print(f"    Indices:      {total_cost['indices_kb']:8.2f} KB")
    print(f"    Residual:     {total_cost['residual_kb']:8.2f} KB")
    print(f"    Border:       {total_cost['border_kb']:8.2f} KB")
    print(f"    ----------------------------------------")
    print(f"    TOTAL:        {total_cost['total_kb']:8.2f} KB")

    return total_cost


# ============================================================
# 3. RESIDUAL VQ (Multi-Stage Cascade)
# ============================================================

def residual_vq(image_path, patch_size=4, n_stages=3):
    """Residual VQ: multi-stage cascaded quantization.

    Stage 0: Quantize original patches -> codebook_0, residual_0
    Stage 1: Quantize residual_0 -> codebook_1, residual_1
    Stage 2: Quantize residual_1 -> codebook_2, residual_2
    ...
    Final representation: codebook_0[i] + codebook_1[j] + ... + residual_N

    Each stage uses progressively fewer clusters since the residual
    has lower variance. The total codebook + indices overhead must
    be justified by the residual entropy reduction.

    This is the core idea behind Residual VQ (RVQ), used in
    neural audio codecs (SoundStream, Encodec).

    Args:
        image_path: path to input image
        patch_size: side length of square patches
        n_stages: number of quantization stages

    Returns:
        dict with total_bytes and cost breakdown
    """
    print("\n" + "=" * 60)
    print("  RESIDUAL VQ (Multi-Stage)")
    print(f"  patch_size={patch_size}, stages={n_stages}")
    print("=" * 60)

    ycocg = load_image_ycocg(image_path)
    patches, (n_h, n_w), border = extract_patches(ycocg, patch_size)
    dim = patches.shape[1]
    print(f"  Patches: {len(patches)} x {dim}d  (grid {n_h}x{n_w})")

    # Stage configuration: decreasing codebook size per stage
    stage_configs = []
    for s in range(n_stages):
        # First stage gets largest codebook, subsequent stages get smaller
        nk = max(32, 256 >> s)
        stage_configs.append({"n_clusters": nk, "name": f"Stage-{s} (k={nk})"})

    total_cost = {
        "name": "Residual VQ",
        "codebook_kb": 0.0,
        "indices_kb": 0.0,
        "residual_kb": 0.0,
        "border_kb": 0.0,
        "total_kb": 0.0,
        "total_bytes": 0,
    }

    current_data = patches.copy()
    all_centroids = []
    all_labels = []
    cumulative_reconstruction = np.zeros_like(patches)

    for stage_idx, cfg in enumerate(stage_configs):
        nk = cfg["n_clusters"]
        name = cfg["name"]

        print(f"\n  --- {name} ---")
        print(f"  Input range: [{current_data.min()}, {current_data.max()}], "
              f"variance: {current_data.var():.1f}")

        # Build codebook for this stage
        t0 = time.time()
        centroids, _ = kmeans_pp(current_data, nk, max_iter=40, seed=42 + stage_idx)
        t_km = time.time() - t0
        print(f"  K-means++ ({nk} clusters): {t_km:.1f}s")

        # Assign via KD-tree
        tree, _ = build_codebook_tree(centroids)
        labels, reconstructed = assign_with_tree(tree, centroids, current_data)

        all_centroids.append(centroids)
        all_labels.append(labels)

        # Compute residual for next stage
        stage_residual = current_data - reconstructed
        cumulative_reconstruction = cumulative_reconstruction + reconstructed

        # Report this stage
        stage_cost = compute_vq_cost(
            centroids, labels, stage_residual,
            np.array([], dtype=np.int32), name,
        )
        total_cost["codebook_kb"] += stage_cost["codebook_kb"]
        total_cost["indices_kb"] += stage_cost["indices_kb"]

        # Residual stats
        print(f"  Residual after stage: "
              f"variance={stage_residual.var():.1f}, "
              f"max|r|={np.abs(stage_residual).max()}")

        # Next stage quantizes the residual
        current_data = stage_residual

    # Final residual (after all stages)
    final_residual = current_data
    final_residual_flat = final_residual.flatten().astype(np.int32)
    final_residual_zz = zigzag_encode(final_residual_flat)
    final_residual_bits = entropy_bits(final_residual_zz)

    # Lossless verification: sum of all stage reconstructions + final residual == original
    full_recon = cumulative_reconstruction + final_residual
    assert np.array_equal(patches, full_recon), "LOSSLESS CHECK FAILED"
    print("\n  Lossless verification: PASSED")

    # Border cost
    border_bits = 0.0
    if len(border) > 0:
        border_bits = entropy_bits(zigzag_encode(border.astype(np.int32)))

    total_cost["residual_kb"] = bits_to_kb(final_residual_bits)
    total_cost["border_kb"] = bits_to_kb(border_bits)
    total_cost["total_kb"] = (
        total_cost["codebook_kb"] + total_cost["indices_kb"] +
        total_cost["residual_kb"] + total_cost["border_kb"]
    )
    total_cost["total_bytes"] = int(total_cost["total_kb"] * 1024)

    print(f"\n  [Residual VQ] TOTAL Cost:")
    print(f"    Codebooks:    {total_cost['codebook_kb']:8.2f} KB")
    print(f"    Indices:      {total_cost['indices_kb']:8.2f} KB")
    print(f"    Residual:     {total_cost['residual_kb']:8.2f} KB")
    print(f"    Border:       {total_cost['border_kb']:8.2f} KB")
    print(f"    ----------------------------------------")
    print(f"    TOTAL:        {total_cost['total_kb']:8.2f} KB")

    return total_cost


# ============================================================
# 4. PRODUCT VQ (Subvector Quantization)
# ============================================================

def product_vq(image_path, patch_size=8, n_sub=4):
    """Product VQ: split patch vector into subvectors, quantize independently.

    A patch of size 8x8x3 = 192 dimensions is split into n_sub subvectors
    of 192/n_sub = 48 dimensions each. Each subvector has its own small
    codebook (e.g., 256 entries in R^48 instead of R^192).

    This dramatically reduces codebook memory:
      Full VQ:    256 * 192 = 49,152 values
      Product VQ: 4 * 256 * 48 = 49,152 values (same!)

    But the key advantage is that k-means in R^48 converges much
    faster and produces better clusters than in R^192.

    The downside: subvector boundaries break correlations between
    the subvector components. The residual captures what's lost.

    Args:
        image_path: path to input image
        patch_size: side length of square patches
        n_sub: number of subvectors per patch

    Returns:
        dict with total_bytes and cost breakdown
    """
    print("\n" + "=" * 60)
    print("  PRODUCT VQ (Subvector Quantization)")
    print(f"  patch_size={patch_size}, n_sub={n_sub}")
    print("=" * 60)

    ycocg = load_image_ycocg(image_path)
    patches, (n_h, n_w), border = extract_patches(ycocg, patch_size)
    n_patches, full_dim = patches.shape
    print(f"  Patches: {n_patches} x {full_dim}d  (grid {n_h}x{n_w})")

    if full_dim % n_sub != 0:
        raise ValueError(
            f"Patch dimension {full_dim} not divisible by n_sub={n_sub}"
        )

    sub_dim = full_dim // n_sub
    n_clusters_per_sub = 256
    print(f"  Subvectors: {n_sub} x {sub_dim}d, "
          f"{n_clusters_per_sub} clusters each")

    total_cost = {
        "name": "Product VQ",
        "codebook_kb": 0.0,
        "indices_kb": 0.0,
        "residual_kb": 0.0,
        "border_kb": 0.0,
        "total_kb": 0.0,
        "total_bytes": 0,
    }

    # Reconstruct by independently quantizing each subvector
    full_reconstructed = np.empty_like(patches)

    for sub_idx in range(n_sub):
        col_start = sub_idx * sub_dim
        col_end = col_start + sub_dim
        sub_data = patches[:, col_start:col_end].copy()

        print(f"\n  --- Subvector {sub_idx} "
              f"(dims [{col_start}:{col_end}]) ---")

        # Build codebook for this subvector space
        t0 = time.time()
        centroids, _ = kmeans_pp(
            sub_data, n_clusters_per_sub, max_iter=30, seed=42 + sub_idx,
        )
        t_km = time.time() - t0
        print(f"  K-means++ ({n_clusters_per_sub} clusters in R^{sub_dim}): "
              f"{t_km:.1f}s")

        # Assign via KD-tree
        tree, _ = build_codebook_tree(centroids)
        labels, reconstructed = assign_with_tree(tree, centroids, sub_data)

        full_reconstructed[:, col_start:col_end] = reconstructed

        # Sub-residual for reporting
        sub_residual = sub_data - reconstructed

        # Cost for this subvector
        sub_cost = compute_vq_cost(
            centroids, labels, sub_residual,
            np.array([], dtype=np.int32),
            f"Sub-{sub_idx}",
        )
        total_cost["codebook_kb"] += sub_cost["codebook_kb"]
        total_cost["indices_kb"] += sub_cost["indices_kb"]

    # Full residual
    full_residual = patches - full_reconstructed

    # Lossless verification
    assert np.array_equal(patches, full_reconstructed + full_residual), \
        "LOSSLESS CHECK FAILED"
    print("\n  Lossless verification: PASSED")

    # Full residual cost
    residual_flat = full_residual.flatten().astype(np.int32)
    residual_zz = zigzag_encode(residual_flat)
    residual_bits = entropy_bits(residual_zz)

    # Border cost
    border_bits = 0.0
    if len(border) > 0:
        border_bits = entropy_bits(zigzag_encode(border.astype(np.int32)))

    total_cost["residual_kb"] = bits_to_kb(residual_bits)
    total_cost["border_kb"] = bits_to_kb(border_bits)
    total_cost["total_kb"] = (
        total_cost["codebook_kb"] + total_cost["indices_kb"] +
        total_cost["residual_kb"] + total_cost["border_kb"]
    )
    total_cost["total_bytes"] = int(total_cost["total_kb"] * 1024)

    # Residual analysis
    nonzero_frac = np.count_nonzero(full_residual) / full_residual.size
    print(f"\n  Full residual: {nonzero_frac * 100:.1f}% nonzero, "
          f"MAE={np.abs(full_residual).mean():.2f}, "
          f"max|r|={np.abs(full_residual).max()}")

    print(f"\n  [Product VQ] TOTAL Cost:")
    print(f"    Codebooks:    {total_cost['codebook_kb']:8.2f} KB")
    print(f"    Indices:      {total_cost['indices_kb']:8.2f} KB")
    print(f"    Residual:     {total_cost['residual_kb']:8.2f} KB")
    print(f"    Border:       {total_cost['border_kb']:8.2f} KB")
    print(f"    ----------------------------------------")
    print(f"    TOTAL:        {total_cost['total_kb']:8.2f} KB")

    return total_cost


# ============================================================
# COMPARISON TABLE
# ============================================================

def print_comparison(results, original_bytes):
    """Print a formatted comparison table of all VQ approaches."""
    print("\n")
    print("=" * 78)
    print("  VQ CODEC COMPARISON TABLE")
    print("=" * 78)
    print(f"  Original image size: {original_bytes:,} bytes "
          f"({original_bytes / 1024:.1f} KB)")
    print()
    print(f"  {'Method':<25} {'Codebook':>10} {'Indices':>10} "
          f"{'Residual':>10} {'Total':>10} {'Ratio':>8}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} "
          f"{'-' * 10} {'-' * 10} {'-' * 8}")

    for r in results:
        ratio = original_bytes / max(1, r["total_bytes"])
        print(f"  {r['name']:<25} "
              f"{r['codebook_kb']:>9.1f}K "
              f"{r['indices_kb']:>9.1f}K "
              f"{r['residual_kb']:>9.1f}K "
              f"{r['total_kb']:>9.1f}K "
              f"{ratio:>7.2f}x")

    print()

    # Analysis: where does the cost live?
    print("  KEY INSIGHTS:")
    best = min(results, key=lambda x: x["total_kb"])
    worst = max(results, key=lambda x: x["total_kb"])
    print(f"  - Best method:  {best['name']} ({best['total_kb']:.1f} KB)")
    print(f"  - Worst method: {worst['name']} ({worst['total_kb']:.1f} KB)")

    for r in results:
        total = r["total_kb"]
        if total > 0:
            res_pct = r["residual_kb"] / total * 100
            idx_pct = r["indices_kb"] / total * 100
            cb_pct = r["codebook_kb"] / total * 100
            print(f"  - {r['name']}: "
                  f"codebook={cb_pct:.0f}% indices={idx_pct:.0f}% "
                  f"residual={res_pct:.0f}%")

    print()
    print("  VQ cannot achieve lossless compression below entropy.")
    print("  The residual dominates cost because k-means centroids")
    print("  are means (real-valued), but pixels are integers.")
    print("  The rounding error must be stored losslessly.")
    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    img_path = str(Path(__file__).resolve().parent.parent / "img.png")

    print("=" * 78)
    print("  VECTOR QUANTIZATION CODEC -- Lossless Compression Research")
    print("  All algorithms from scratch. No sklearn, no scipy.")
    print("=" * 78)

    # Get original size for comparison
    original_bytes = Path(img_path).stat().st_size
    img = Image.open(img_path)
    raw_pixels = np.array(img)
    if raw_pixels.ndim == 3 and raw_pixels.shape[2] == 4:
        raw_pixels = raw_pixels[:, :, :3]
    raw_bytes = raw_pixels.size  # H * W * 3
    print(f"\n  Image:     {img_path}")
    print(f"  File size: {original_bytes:,} bytes ({original_bytes / 1024:.1f} KB)")
    print(f"  Raw RGB:   {raw_bytes:,} bytes ({raw_bytes / 1024:.1f} KB)")

    results = []
    t_total = time.time()

    # 1. Basic VQ
    r1 = basic_vq(img_path, patch_size=4, n_clusters=256)
    results.append(r1)

    # 2. Hierarchical VQ
    r2 = hierarchical_vq(img_path)
    results.append(r2)

    # 3. Residual VQ
    r3 = residual_vq(img_path, patch_size=4, n_stages=3)
    results.append(r3)

    # 4. Product VQ
    r4 = product_vq(img_path, patch_size=8, n_sub=4)
    results.append(r4)

    elapsed = time.time() - t_total
    print(f"\n  Total runtime: {elapsed:.1f}s")

    # Final comparison
    print_comparison(results, raw_bytes)
