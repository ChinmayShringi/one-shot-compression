"""ORACLE: Per-pixel optimal prediction with learned mode selection.

The oracle proves 727 KB is achievable. The problem is the mode map (920 KB).

Solution: Instead of TRANSMITTING the mode map, LEARN a compact function
that predicts the right mode from context features. The function is stored
in the compressed file (few KB). The decoder runs the same function.

This is fundamentally different from traditional compression:
- The compressed file contains a PROGRAM (decision tree / lookup table)
- The decoder RUNS the program to decide how to predict each pixel
- The program + residuals IS the compressed representation

Approach 1: Decision tree trained to classify optimal mode from context
Approach 2: Lookup table: hash(context) -> best mode
Approach 3: Tiny neural network: f(features) -> mode weights
"""

import numpy as np
from PIL import Image
import math, struct, time
from collections import Counter

from prism.transforms import rgb_to_ycocg_r, ycocg_r_to_rgb, zigzag_encode, zigzag_decode
from prism.predict_fast import compute_all_predictors, _get_neighbors, BORDER_DEFAULT


def entropy_bps(data):
    flat = data.flatten()
    if len(flat) == 0:
        return 0.0
    c = Counter(flat.tolist())
    t = len(flat)
    return -sum((v / t) * math.log2(v / t) for v in c.values())


# ============================================================
# Decision Tree - built from scratch, no sklearn
# ============================================================

class DecisionNode:
    """A single node in a binary decision tree."""
    __slots__ = ('feature', 'threshold', 'left', 'right', 'prediction')

    def __init__(self):
        self.feature = -1
        self.threshold = 0
        self.left = None
        self.right = None
        self.prediction = 0  # Leaf: majority class


def build_tree(X, y, max_depth=12, min_samples=50, n_classes=10):
    """Build a decision tree from scratch. No libraries.

    X: (n_samples, n_features) - context features
    y: (n_samples,) - oracle mode labels
    max_depth: maximum tree depth
    min_samples: minimum samples to split

    Uses Gini impurity for splitting.
    """
    node = DecisionNode()

    # Leaf conditions
    if max_depth <= 0 or len(y) < min_samples:
        counts = np.bincount(y, minlength=n_classes)
        node.prediction = int(counts.argmax())
        return node

    # Check if pure
    unique = np.unique(y)
    if len(unique) == 1:
        node.prediction = int(unique[0])
        return node

    n_samples, n_features = X.shape
    best_gain = -1
    best_feature = 0
    best_threshold = 0

    # Current Gini
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    probs = counts / counts.sum()
    current_gini = 1.0 - np.sum(probs ** 2)

    # Try each feature
    for f in range(n_features):
        col = X[:, f]
        # Try a few thresholds (percentiles for speed)
        thresholds = np.percentile(col, [10, 25, 40, 50, 60, 75, 90])
        thresholds = np.unique(thresholds)

        for t in thresholds:
            left_mask = col <= t
            right_mask = ~left_mask
            n_left = left_mask.sum()
            n_right = right_mask.sum()

            if n_left < min_samples or n_right < min_samples:
                continue

            # Gini for left
            left_counts = np.bincount(y[left_mask], minlength=n_classes).astype(np.float64)
            left_probs = left_counts / left_counts.sum()
            left_gini = 1.0 - np.sum(left_probs ** 2)

            # Gini for right
            right_counts = np.bincount(y[right_mask], minlength=n_classes).astype(np.float64)
            right_probs = right_counts / right_counts.sum()
            right_gini = 1.0 - np.sum(right_probs ** 2)

            # Weighted Gini gain
            w_gini = (n_left * left_gini + n_right * right_gini) / n_samples
            gain = current_gini - w_gini

            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_threshold = t

    if best_gain <= 0:
        counts = np.bincount(y, minlength=n_classes)
        node.prediction = int(counts.argmax())
        return node

    # Split
    node.feature = best_feature
    node.threshold = best_threshold

    left_mask = X[:, best_feature] <= best_threshold
    node.left = build_tree(X[left_mask], y[left_mask], max_depth - 1, min_samples, n_classes)
    node.right = build_tree(X[~left_mask], y[~left_mask], max_depth - 1, min_samples, n_classes)

    return node


def predict_tree(node, X):
    """Predict classes for all samples using the tree."""
    predictions = np.zeros(len(X), dtype=np.int32)
    _predict_batch(node, X, np.arange(len(X)), predictions)
    return predictions


def _predict_batch(node, X, indices, out):
    """Recursively predict for a batch of samples."""
    if node.feature < 0:  # Leaf
        out[indices] = node.prediction
        return

    col = X[indices, node.feature]
    left_mask = col <= node.threshold
    left_idx = indices[left_mask]
    right_idx = indices[~left_mask]

    if len(left_idx) > 0:
        _predict_batch(node.left, X, left_idx, out)
    if len(right_idx) > 0:
        _predict_batch(node.right, X, right_idx, out)


def count_tree_nodes(node):
    """Count total nodes in tree."""
    if node.feature < 0:
        return 1
    return 1 + count_tree_nodes(node.left) + count_tree_nodes(node.right)


def tree_storage_bytes(node):
    """Estimate storage for the tree.
    Each internal node: feature(1) + threshold(2) = 3 bytes
    Each leaf: prediction(1) = 1 byte
    Plus 1 bit per node for leaf/internal flag.
    """
    n = count_tree_nodes(node)
    # Simple estimate: 3 bytes per node average
    return n * 3


# ============================================================
# Context Feature Extraction
# ============================================================

def extract_features(ch):
    """Extract per-pixel context features from a channel.

    Returns (n_pixels, n_features) array.
    All features are causal (available to decoder).
    """
    h, w = ch.shape
    left, above, ul, ur, left2, above2 = _get_neighbors(ch)
    D = BORDER_DEFAULT

    # Features: all computable from causal context
    f1 = left.flatten()                          # left value
    f2 = above.flatten()                         # above value
    f3 = ul.flatten()                            # upper-left value
    f4 = (left + above - ul).flatten()           # gradient estimate
    f5 = np.abs(above - ul).flatten()            # horizontal gradient
    f6 = np.abs(left - ul).flatten()             # vertical gradient
    f7 = (f5 + f6)                               # total gradient magnitude
    f8 = np.abs(left - left2).flatten()          # second-order horizontal
    f9 = np.abs(above - above2).flatten()        # second-order vertical
    f10 = np.abs(left + above - 2 * ul).flatten()  # curvature
    f11 = ((left + above + 1) // 2).flatten()    # average prediction
    f12 = np.minimum(f7, 255)                    # clamped gradient

    features = np.column_stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12])
    return features.astype(np.int32)


# ============================================================
# Full Oracle + Decision Tree Pipeline
# ============================================================

def oracle_tree_analyze(image_path):
    """Build oracle + decision tree codec and measure compression."""
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = img.shape
    tp = h * w
    ycocg = rgb_to_ycocg_r(img)

    print(f"=== ORACLE + DECISION TREE CODEC ===")
    print(f"Image: {w}x{h}")
    print()

    total_tree_bytes = 0
    total_residual_bits = 0

    for ch_idx, name in enumerate(['Y', 'Co', 'Cg']):
        ch = ycocg[:, :, ch_idx].astype(np.int32)
        preds = compute_all_predictors(ch)
        pred_arrays = list(preds.values())
        pred_names = list(preds.keys())
        n_preds = len(pred_arrays)

        print(f"  --- {name} channel ---")

        # Oracle mode map
        t0 = time.time()
        all_abs_res = np.stack([np.abs(ch - p) for p in pred_arrays], axis=-1)
        oracle_modes = np.argmin(all_abs_res, axis=-1).astype(np.int32)  # (h, w)

        # Oracle residual
        oracle_residual = np.zeros_like(ch)
        for i, pred in enumerate(pred_arrays):
            mask = oracle_modes == i
            oracle_residual[mask] = ch[mask] - pred[mask]

        oracle_zz = zigzag_encode(oracle_residual)
        oracle_bps = entropy_bps(oracle_zz)
        oracle_kb = oracle_bps * tp / 8 / 1024
        print(f"    Oracle residual: {oracle_bps:.3f} bps = {oracle_kb:.0f} KB")

        # Extract features
        features = extract_features(ch)
        labels = oracle_modes.flatten()

        # Train decision tree
        for depth in [6, 8, 10, 12, 14]:
            t0 = time.time()
            tree = build_tree(features, labels, max_depth=depth,
                            min_samples=100, n_classes=n_preds)
            dt_train = time.time() - t0

            # Predict with tree
            predicted_modes = predict_tree(tree, features).reshape(h, w)
            accuracy = (predicted_modes == oracle_modes).mean() * 100

            # Compute residual with tree-predicted modes
            tree_residual = np.zeros_like(ch)
            for i, pred in enumerate(pred_arrays):
                mask = predicted_modes == i
                tree_residual[mask] = ch[mask] - pred[mask]

            tree_zz = zigzag_encode(tree_residual)
            tree_bps = entropy_bps(tree_zz)
            tree_kb = tree_bps * tp / 8 / 1024

            # Tree storage
            n_nodes = count_tree_nodes(tree)
            tree_bytes = tree_storage_bytes(tree)

            total_kb = tree_bytes / 1024 + tree_kb
            print(f"    Tree depth={depth:>2d}: {n_nodes:>5d} nodes ({tree_bytes/1024:.1f} KB), "
                  f"accuracy={accuracy:.1f}%, "
                  f"residual={tree_kb:.0f} KB, "
                  f"TOTAL={total_kb:.0f} KB", end='')
            if total_kb < 917 / 3:
                print(' <JXL/3!', end='')
            print()

        # Use depth 12 as default
        tree = build_tree(features, labels, max_depth=12,
                         min_samples=100, n_classes=n_preds)
        predicted_modes = predict_tree(tree, features).reshape(h, w)
        tree_residual = np.zeros_like(ch)
        for i, pred in enumerate(pred_arrays):
            mask = predicted_modes == i
            tree_residual[mask] = ch[mask] - pred[mask]

        tree_zz = zigzag_encode(tree_residual)
        tree_bps = entropy_bps(tree_zz)
        tree_bytes = tree_storage_bytes(tree)

        total_tree_bytes += tree_bytes
        total_residual_bits += tree_bps * tp
        print()

    total_kb = total_tree_bytes / 1024 + total_residual_bits / 8 / 1024
    print(f"  === TOTAL ===")
    print(f"  Trees: {total_tree_bytes / 1024:.0f} KB")
    print(f"  Residuals: {total_residual_bits / 8 / 1024:.0f} KB")
    print(f"  TOTAL: {total_kb:.0f} KB")
    print(f"  vs PRISM v2:  1444 KB ({(1 - total_kb / 1444) * 100:+.1f}%)")
    print(f"  vs JPEG-XL:    917 KB ({(1 - total_kb / 917) * 100:+.1f}%)")

    return total_kb


# ============================================================
# Approach 2: Context Lookup Table
# ============================================================

def lookup_table_analyze(image_path):
    """Build a per-image lookup table predictor.

    For each context pattern, store the most likely pixel value.
    The table IS part of the compressed file.
    """
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = img.shape
    tp = h * w
    ycocg = rgb_to_ycocg_r(img)

    print(f"=== CONTEXT LOOKUP TABLE PREDICTOR ===")
    print(f"Store a table: hash(left, above, ul) -> predicted_value")
    print()

    for n_bits in [6, 8, 10, 12]:
        total_table_bytes = 0
        total_res_bits = 0

        for ch_idx, name in enumerate(['Y', 'Co', 'Cg']):
            ch = ycocg[:, :, ch_idx].astype(np.int32)
            left, above, ul, _, _, _ = _get_neighbors(ch)

            # Quantize context to n_bits total
            bits_per = n_bits // 3
            remaining = n_bits - 3 * bits_per

            q_left = np.right_shift(left.flatten() + 256, 9 - bits_per - (1 if remaining > 0 else 0))
            q_above = np.right_shift(above.flatten() + 256, 9 - bits_per - (1 if remaining > 1 else 0))
            q_ul = np.right_shift(ul.flatten() + 256, 9 - bits_per)

            # Combine into context key
            ctx = (q_left << (2 * bits_per + min(remaining, 2))) | \
                  (q_above << (bits_per + min(remaining - 1, 0) if remaining > 1 else bits_per)) | q_ul

            # Simple hash: just use modulo
            n_entries = 1 << n_bits
            ctx = ctx % n_entries

            # Build table: for each context, store the average value
            table = np.zeros(n_entries, dtype=np.float64)
            counts = np.zeros(n_entries, dtype=np.int64)
            flat = ch.flatten()

            for i in range(len(flat)):
                c = int(ctx[i])
                table[c] += flat[i]
                counts[c] += 1

            # Average
            valid = counts > 0
            table[valid] = table[valid] / counts[valid]
            table = np.round(table).astype(np.int32)

            # Predict using table
            predicted = table[ctx.astype(np.int64)]
            residual = flat - predicted

            zz = zigzag_encode(residual.reshape(h, w))
            bps = entropy_bps(zz)

            table_bytes = n_entries * 2  # 2 bytes per entry
            total_table_bytes += table_bytes
            total_res_bits += bps * tp

        total_kb = total_table_bytes / 1024 + total_res_bits / 8 / 1024
        print(f"  {n_bits}-bit context ({1 << n_bits:>5d} entries): "
              f"table={total_table_bytes / 1024:.0f} KB + "
              f"res={total_res_bits / 8 / 1024:.0f} KB = "
              f"{total_kb:.0f} KB", end='')
        if total_kb < 917:
            print(' <JXL!', end='')
        print()

    print()


# ============================================================
# Approach 3: Tiny Neural Net (from scratch, no frameworks)
# ============================================================

def neural_predictor_analyze(image_path):
    """Fit a tiny neural network f(x,y,context) -> pixel_value.

    The network weights ARE the compressed representation.
    Decoder runs the network to generate pixel predictions.
    Exact residual for lossless.

    Network: 2-layer MLP with ReLU, trained by gradient descent.
    """
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = img.shape
    tp = h * w
    ycocg = rgb_to_ycocg_r(img)

    print(f"=== TINY NEURAL NETWORK PREDICTOR ===")
    print(f"Store network weights. Decoder RUNS the network.")
    print()

    # Coordinate features (available without context)
    ys_norm = np.linspace(-1, 1, h)
    xs_norm = np.linspace(-1, 1, w)
    X, Y = np.meshgrid(xs_norm, ys_norm)

    # Features: (x, y, x^2, y^2, x*y, sin(pi*x), sin(pi*y), ...)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    features = np.column_stack([
        x_flat, y_flat,
        x_flat ** 2, y_flat ** 2,
        x_flat * y_flat,
        np.sin(np.pi * x_flat), np.sin(np.pi * y_flat),
        np.sin(2 * np.pi * x_flat), np.sin(2 * np.pi * y_flat),
        np.cos(np.pi * x_flat), np.cos(np.pi * y_flat),
    ])
    n_features = features.shape[1]

    for hidden_size in [16, 32, 64, 128, 256]:
        total_param_bytes = 0
        total_res_bits = 0

        for ch_idx, name in enumerate(['Y', 'Co', 'Cg']):
            ch_flat = ycocg[:, :, ch_idx].astype(np.float64).flatten()

            # Simple 1-hidden-layer network: features -> hidden -> output
            # Use random projection + least squares (fast, no backprop needed)

            # Random hidden layer (fixed, not stored)
            np.random.seed(42 + ch_idx)
            W1 = np.random.randn(n_features, hidden_size) / np.sqrt(n_features)
            b1 = np.random.randn(hidden_size) * 0.1

            # Hidden activations (ReLU)
            hidden = features @ W1 + b1
            hidden = np.maximum(hidden, 0)  # ReLU

            # Output layer: least squares fit
            # hidden @ W2 + b2 ≈ ch_flat
            hidden_aug = np.column_stack([hidden, np.ones(len(ch_flat))])
            W2, _, _, _ = np.linalg.lstsq(hidden_aug, ch_flat, rcond=None)

            # Predict
            predicted = (hidden_aug @ W2)
            predicted_int = np.round(predicted).astype(np.int32)
            residual = ycocg[:, :, ch_idx].astype(np.int32).flatten() - predicted_int

            zz = zigzag_encode(residual.reshape(ycocg.shape[0], ycocg.shape[1]))
            bps = entropy_bps(zz)

            # Parameters: W2 (hidden_size+1 values) stored as float16
            # W1 is random (seeded, not stored)
            param_bytes = (hidden_size + 1) * 2  # float16
            total_param_bytes += param_bytes
            total_res_bits += bps * tp

        total_kb = total_param_bytes / 1024 + total_res_bits / 8 / 1024
        print(f"  Hidden={hidden_size:>3d}: params={total_param_bytes / 1024:.1f} KB + "
              f"res={total_res_bits / 8 / 1024:.0f} KB = "
              f"{total_kb:.0f} KB", end='')
        if total_kb < 917:
            print(' <JXL!', end='')
        if total_kb < 1444:
            print(' <PRISM', end='')
        print()

    # Try with CAUSAL context features too
    print()
    print("  --- With causal context features ---")
    for hidden_size in [32, 64, 128]:
        total_param_bytes = 0
        total_res_bits = 0

        for ch_idx, name in enumerate(['Y', 'Co', 'Cg']):
            ch = ycocg[:, :, ch_idx].astype(np.int32)
            left, above, ul, _, left2, above2 = _get_neighbors(ch)

            # Context features + coordinate features
            ctx_features = np.column_stack([
                x_flat, y_flat,
                x_flat ** 2, y_flat ** 2,
                left.flatten().astype(np.float64) / 256,
                above.flatten().astype(np.float64) / 256,
                ul.flatten().astype(np.float64) / 256,
                (left + above - ul).flatten().astype(np.float64) / 256,
                np.abs(above - ul).flatten().astype(np.float64) / 256,
                np.abs(left - ul).flatten().astype(np.float64) / 256,
            ])
            n_ctx_feat = ctx_features.shape[1]

            np.random.seed(42 + ch_idx)
            W1 = np.random.randn(n_ctx_feat, hidden_size) / np.sqrt(n_ctx_feat)
            b1 = np.random.randn(hidden_size) * 0.1
            hidden = np.maximum(ctx_features @ W1 + b1, 0)

            hidden_aug = np.column_stack([hidden, np.ones(tp)])
            ch_flat = ch.flatten().astype(np.float64)
            W2, _, _, _ = np.linalg.lstsq(hidden_aug, ch_flat, rcond=None)

            predicted_int = np.round(hidden_aug @ W2).astype(np.int32)
            residual = ch.flatten() - predicted_int

            zz = zigzag_encode(residual.reshape(ch.shape))
            bps = entropy_bps(zz)

            param_bytes = (hidden_size + 1) * 2
            total_param_bytes += param_bytes
            total_res_bits += bps * tp

        total_kb = total_param_bytes / 1024 + total_res_bits / 8 / 1024
        print(f"  Context+Hidden={hidden_size:>3d}: params={total_param_bytes / 1024:.1f} KB + "
              f"res={total_res_bits / 8 / 1024:.0f} KB = "
              f"{total_kb:.0f} KB", end='')
        if total_kb < 917:
            print(' <JXL!', end='')
        if total_kb < 1444:
            print(' <PRISM', end='')
        print()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "img.png"

    oracle_tree_analyze(path)
    print()
    lookup_table_analyze(path)
    print()
    neural_predictor_analyze(path)
