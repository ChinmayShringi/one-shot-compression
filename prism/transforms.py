"""Reversible color and spatial transforms - built from scratch.

Key innovation: Image-specific color decorrelation using lifting-based
Givens rotations. Unlike fixed YCbCr, this adapts to the actual color
distribution of each image, concentrating more energy in fewer channels.

Also implements Hilbert curve scanning for 2D-locality-preserving
linearization of the pixel grid.
"""

import numpy as np


# ============================================================
# Reversible YCoCg-R Color Transform
# ============================================================

def rgb_to_ycocg_r(rgb: np.ndarray) -> np.ndarray:
    """Reversible YCoCg-R transform (integer-exact).

    Maps RGB -> (Y, Co, Cg) where:
      Co = R - B           (chroma orange)
      tmp = B + (Co >> 1)  (approximate green)
      Cg = G - tmp         (chroma green)
      Y = tmp + (Cg >> 1)  (luma)

    This decorrelates natural image colors better than
    simple channel differences and is exactly invertible.
    """
    r = rgb[:, :, 0].astype(np.int32)
    g = rgb[:, :, 1].astype(np.int32)
    b = rgb[:, :, 2].astype(np.int32)

    co = r - b
    tmp = b + (co >> 1)
    cg = g - tmp
    y = tmp + (cg >> 1)

    return np.stack([y, co, cg], axis=-1)


def ycocg_r_to_rgb(ycocg: np.ndarray) -> np.ndarray:
    """Inverse YCoCg-R: (Y, Co, Cg) -> RGB (integer-exact)."""
    y = ycocg[:, :, 0].astype(np.int32)
    co = ycocg[:, :, 1].astype(np.int32)
    cg = ycocg[:, :, 2].astype(np.int32)

    tmp = y - (cg >> 1)
    g = tmp + cg
    b = tmp - (co >> 1)
    r = b + co

    return np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)


# ============================================================
# Image-Specific Reversible PCA via Lifting
# ============================================================

def compute_pca_axes(rgb: np.ndarray):
    """Find PCA axes of the image's color distribution.

    Returns (mean, eigenvectors, eigenvalues) where eigenvectors
    are sorted by descending eigenvalue.
    """
    pixels = rgb.reshape(-1, 3).astype(np.float64)
    mean = pixels.mean(axis=0)
    centered = pixels - mean
    # Covariance matrix (3x3)
    cov = (centered.T @ centered) / len(pixels)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = eigenvalues.argsort()[::-1]
    return mean, eigenvectors[:, idx], eigenvalues[idx]


def _givens_angle(v1: int, v2: int, matrix: np.ndarray) -> float:
    """Compute Givens rotation angle for axes v1, v2 to diagonalize."""
    a = matrix[v1, v1]
    b = matrix[v2, v2]
    c = matrix[v1, v2]
    if abs(c) < 1e-10:
        return 0.0
    theta = 0.5 * np.arctan2(2 * c, a - b)
    return theta


def _lift_rotate_forward(ch_a: np.ndarray, ch_b: np.ndarray,
                         theta: float) -> tuple:
    """Apply a Givens rotation via 3-step lifting (integer-exact).

    Decomposes rotation by theta into:
      Step 1: a' = a + floor(alpha * b)
      Step 2: b' = b + floor(beta * a')
      Step 3: a'' = a' + floor(alpha * b')

    where alpha = (cos(theta) - 1) / sin(theta)
          beta  = sin(theta)

    This is EXACTLY reversible with integer arithmetic.
    """
    if abs(theta) < 1e-10:
        return ch_a.copy(), ch_b.copy()

    s = np.sin(theta)
    c = np.cos(theta)
    alpha = (c - 1.0) / s
    beta = s

    a = ch_a.astype(np.int32)
    b = ch_b.astype(np.int32)

    # 3-step lifting
    a = a + np.floor(alpha * b).astype(np.int32)
    b = b + np.floor(beta * a).astype(np.int32)
    a = a + np.floor(alpha * b).astype(np.int32)

    return a, b


def _lift_rotate_inverse(ch_a: np.ndarray, ch_b: np.ndarray,
                          theta: float) -> tuple:
    """Inverse lifting rotation (exact reverse of forward)."""
    if abs(theta) < 1e-10:
        return ch_a.copy(), ch_b.copy()

    s = np.sin(theta)
    c = np.cos(theta)
    alpha = (c - 1.0) / s
    beta = s

    a = ch_a.astype(np.int32)
    b = ch_b.astype(np.int32)

    # Reverse order of lifting steps
    a = a - np.floor(alpha * b).astype(np.int32)
    b = b - np.floor(beta * a).astype(np.int32)
    a = a - np.floor(alpha * b).astype(np.int32)

    return a, b


def pca_forward(rgb: np.ndarray) -> tuple:
    """Apply image-specific PCA via lifting rotations.

    Returns (transformed_channels, params) where params stores
    the rotation angles needed for inversion.

    The 3x3 PCA rotation is decomposed into 3 Givens rotations:
      R = G_12(theta_1) @ G_02(theta_2) @ G_01(theta_3)
    Each Givens rotation is implemented via 3-step lifting.
    """
    mean, eigvecs, eigvals = compute_pca_axes(rgb)

    # Compute Givens angles from eigenvector matrix
    # We decompose the rotation matrix into 3 Givens rotations
    R = eigvecs  # 3x3 rotation matrix

    # Extract angles via QR-like Givens decomposition
    # Rotate in planes (0,1), (0,2), (1,2)
    theta_12 = np.arctan2(R[2, 0], R[2, 2]) if abs(R[2, 0]) > 1e-10 else 0.0
    theta_02 = np.arctan2(-R[1, 0], R[0, 0]) if abs(R[1, 0]) > 1e-10 else 0.0
    theta_01 = np.arctan2(R[2, 1], R[2, 2]) if abs(R[2, 1]) > 1e-10 else 0.0

    h, w, _ = rgb.shape
    ch0 = rgb[:, :, 0].astype(np.int32) - int(round(mean[0]))
    ch1 = rgb[:, :, 1].astype(np.int32) - int(round(mean[1]))
    ch2 = rgb[:, :, 2].astype(np.int32) - int(round(mean[2]))

    # Apply Givens rotations via lifting
    ch0, ch1 = _lift_rotate_forward(ch0, ch1, theta_01)
    ch0, ch2 = _lift_rotate_forward(ch0, ch2, theta_02)
    ch1, ch2 = _lift_rotate_forward(ch1, ch2, theta_12)

    result = np.stack([ch0, ch1, ch2], axis=-1)
    params = {
        'mean': [int(round(m)) for m in mean],
        'angles': [float(theta_01), float(theta_02), float(theta_12)]
    }
    return result, params


def pca_inverse(transformed: np.ndarray, params: dict) -> np.ndarray:
    """Invert PCA lifting transform (integer-exact)."""
    mean = params['mean']
    theta_01, theta_02, theta_12 = params['angles']

    ch0 = transformed[:, :, 0].astype(np.int32)
    ch1 = transformed[:, :, 1].astype(np.int32)
    ch2 = transformed[:, :, 2].astype(np.int32)

    # Reverse order of Givens rotations
    ch1, ch2 = _lift_rotate_inverse(ch1, ch2, theta_12)
    ch0, ch2 = _lift_rotate_inverse(ch0, ch2, theta_02)
    ch0, ch1 = _lift_rotate_inverse(ch0, ch1, theta_01)

    ch0 = ch0 + mean[0]
    ch1 = ch1 + mean[1]
    ch2 = ch2 + mean[2]

    return np.stack([ch0, ch1, ch2], axis=-1).clip(0, 255).astype(np.uint8)


# ============================================================
# Hilbert Curve - 2D locality-preserving scan order
# ============================================================

def _hilbert_d2xy(n: int, d: int) -> tuple:
    """Convert Hilbert curve index d to (x, y) coordinates in an n×n grid."""
    x = y = 0
    s = 1
    while s < n:
        rx = 1 if (d & 2) else 0
        ry = 1 if ((d & 1) ^ rx) else 0
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1
    return x, y


def hilbert_scan_order(height: int, width: int) -> np.ndarray:
    """Generate Hilbert-curve scan order for an h×w image.

    Returns array of shape (h*w, 2) with (y, x) coordinates
    in Hilbert curve order. For non-power-of-2 dimensions,
    uses the next power of 2 and filters valid coordinates.
    """
    # Next power of 2 that covers both dimensions
    n = 1
    max_dim = max(height, width)
    while n < max_dim:
        n <<= 1

    total = n * n
    coords = []
    for d in range(total):
        x, y = _hilbert_d2xy(n, d)
        if y < height and x < width:
            coords.append((y, x))

    return np.array(coords, dtype=np.int32)


# ============================================================
# Zigzag encoding: signed -> unsigned mapping
# ============================================================

def zigzag_encode(x: np.ndarray) -> np.ndarray:
    """Map signed integers to unsigned: 0,-1,1,-2,2 -> 0,1,2,3,4"""
    x = x.astype(np.int32)
    return np.where(x >= 0, 2 * x, -2 * x - 1).astype(np.uint32)


def zigzag_decode(x: np.ndarray) -> np.ndarray:
    """Map unsigned back to signed: 0,1,2,3,4 -> 0,-1,1,-2,2"""
    x = x.astype(np.int32)
    return np.where(x & 1, -(x + 1) // 2, x // 2).astype(np.int32)
