"""Vectorized Block Prediction Engine - the novel core of PRISM.

Unlike traditional codecs that predict pixel-by-pixel with simple filters,
PRISM uses STRUCTURAL prediction:

1. Multiple vectorized predictors (numpy, runs in <1 second)
2. Block polynomial field modeling (captures gradients/curves, NOT just neighbors)
3. Cross-channel structural prediction (G from Y, B from Y+G)
4. Per-block mode selection (pick best strategy per region)

The block polynomial predictor is genuinely novel: it extrapolates from
CONTEXT blocks using a fitted polynomial surface, capturing local
geometry (gradients, curvature) that pixel-level predictors miss.
"""

import numpy as np

# Border default: MUST be identical in encoder and decoder
BORDER_DEFAULT = 128


def _get_neighbors(ch):
    """Get causal neighbor arrays for an entire channel (vectorized).

    CRITICAL: Border defaults must be BORDER_DEFAULT (128) to match
    the decoder's defaults. Using original values as defaults breaks
    lossless round-trip.
    """
    h, w = ch.shape
    i32 = ch.astype(np.int32)
    D = BORDER_DEFAULT

    left = np.full_like(i32, D)
    left[:, 1:] = i32[:, :-1]

    above = np.full_like(i32, D)
    above[1:, :] = i32[:-1, :]

    upper_left = np.full_like(i32, D)
    upper_left[1:, 1:] = i32[:-1, :-1]

    upper_right = np.full_like(i32, D)
    upper_right[1:, :-1] = i32[:-1, 1:]

    left2 = np.full_like(i32, D)
    left2[:, 2:] = i32[:, :-2]

    above2 = np.full_like(i32, D)
    above2[2:, :] = i32[:-2, :]

    return left, above, upper_left, upper_right, left2, above2


def compute_all_predictors(ch):
    """Compute all predictor outputs for entire channel (vectorized).

    Returns dict of predictor_name -> predicted_values (int32 array).
    All predictions are causal (use only left/above neighbors).
    """
    left, above, ul, ur, left2, above2 = _get_neighbors(ch)

    # Standard predictors
    preds = {}

    # P0: Left neighbor
    preds['left'] = left

    # P1: Above neighbor
    preds['above'] = above

    # P2: Linear gradient: left + above - upper_left
    preds['gradient'] = left + above - ul

    # P3: Paeth (PNG-style: pick neighbor closest to gradient estimate)
    p = left + above - ul
    pa = np.abs(p - left)
    pb = np.abs(p - above)
    pc = np.abs(p - ul)
    preds['paeth'] = np.where(
        (pa <= pb) & (pa <= pc), left,
        np.where(pb <= pc, above, ul)
    )

    # P4: Average of left and above
    preds['average'] = (left + above + 1) >> 1

    # P5: Edge-adaptive (novel) - predicts ALONG edges, not across them
    # Only use causal neighbors (left, above, ul) - NOT upper_right,
    # which may cross block boundaries and break encoder/decoder consistency
    dx = np.abs(above - ul)  # horizontal variation
    dy = np.abs(left - ul)   # vertical variation
    preds['edge_adaptive'] = np.where(
        dx > dy + 4, above,   # horizontal edge → use above (along edge)
        np.where(dy > dx + 4, left,  # vertical edge → use left
                 left + above - ul)   # smooth → gradient
    )

    # P6: Median of (left, above, gradient)
    stacked = np.stack([left, above, left + above - ul], axis=-1)
    preds['median3'] = np.median(stacked, axis=-1).astype(np.int32)

    # P7: Second-order horizontal extrapolation: 2*left - left2
    h_extrap = 2 * left - left2
    # P8: Second-order vertical extrapolation: 2*above - above2
    v_extrap = 2 * above - above2
    # Pick the smoother direction
    h_smooth = np.abs(left - left2)
    v_smooth = np.abs(above - above2)
    preds['extrapolate'] = np.where(h_smooth < v_smooth, h_extrap, v_extrap)

    # P9: Weighted gradient: weight left/above by reliability
    # More weight to the direction with less variation
    w_left = 1.0 / (np.abs(left - ul).astype(np.float64) + 1)
    w_above = 1.0 / (np.abs(above - ul).astype(np.float64) + 1)
    w_total = w_left + w_above
    preds['weighted_grad'] = np.round(
        (left * w_left + above * w_above) / w_total
    ).astype(np.int32)

    return preds


def compute_polynomial_predictor(ch, block_size=16):
    """Novel polynomial field predictor.

    For each block, fits a polynomial surface to the CONTEXT
    (blocks above and to the left), then EXTRAPOLATES into the
    current block.

    This captures local geometry (gradients, curvature) that
    simple pixel-level predictors miss. A single polynomial
    describes the entire block's trend with just 6 numbers.

    NO side information needed: the decoder fits the same polynomial
    from already-decoded context.
    """
    h, w = ch.shape
    i32 = ch.astype(np.int32)
    pred = np.zeros_like(i32)

    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            bh = min(block_size, h - by)
            bw = min(block_size, w - bx)

            # Gather context pixels (already decoded)
            ctx_ys = []
            ctx_xs = []
            ctx_vals = []

            # Row above the block
            if by > 0:
                row_above = max(0, by - 2)
                for dy in range(row_above, by):
                    for dx in range(max(0, bx - 2), min(w, bx + bw + 2)):
                        ctx_ys.append(dy)
                        ctx_xs.append(dx)
                        ctx_vals.append(i32[dy, dx])

            # Column left of the block
            if bx > 0:
                col_left = max(0, bx - 2)
                for dy in range(by, min(h, by + bh)):
                    for dx in range(col_left, bx):
                        ctx_ys.append(dy)
                        ctx_xs.append(dx)
                        ctx_vals.append(i32[dy, dx])

            if len(ctx_vals) < 3:
                # Not enough context - use mean of available context or 128
                if ctx_vals:
                    pred[by:by + bh, bx:bx + bw] = int(np.mean(ctx_vals))
                else:
                    pred[by:by + bh, bx:bx + bw] = 128
                continue

            ctx_ys = np.array(ctx_ys, dtype=np.float64)
            ctx_xs = np.array(ctx_xs, dtype=np.float64)
            ctx_vals = np.array(ctx_vals, dtype=np.float64)

            # Normalize coordinates for numerical stability
            y_center = (by + bh / 2)
            x_center = (bx + bw / 2)
            scale = max(bh, bw, 1)
            ny = (ctx_ys - y_center) / scale
            nx = (ctx_xs - x_center) / scale

            # Determine polynomial degree based on context size
            n_ctx = len(ctx_vals)
            if n_ctx >= 6:
                # Quadratic: a + bx + cy + dx² + exy + fy²
                A = np.column_stack([
                    np.ones(n_ctx), nx, ny,
                    nx ** 2, nx * ny, ny ** 2
                ])
            elif n_ctx >= 3:
                # Linear: a + bx + cy
                A = np.column_stack([np.ones(n_ctx), nx, ny])
            else:
                A = np.column_stack([np.ones(n_ctx)])

            try:
                coeffs = np.linalg.lstsq(A, ctx_vals, rcond=None)[0]
            except np.linalg.LinAlgError:
                pred[by:by + bh, bx:bx + bw] = int(np.mean(ctx_vals))
                continue

            # Evaluate polynomial at block positions
            block_ys, block_xs = np.mgrid[by:by + bh, bx:bx + bw]
            bny = (block_ys.astype(np.float64) - y_center) / scale
            bnx = (block_xs.astype(np.float64) - x_center) / scale

            if len(coeffs) >= 6:
                block_pred = (coeffs[0]
                              + coeffs[1] * bnx + coeffs[2] * bny
                              + coeffs[3] * bnx ** 2
                              + coeffs[4] * bnx * bny
                              + coeffs[5] * bny ** 2)
            elif len(coeffs) >= 3:
                block_pred = coeffs[0] + coeffs[1] * bnx + coeffs[2] * bny
            else:
                block_pred = np.full((bh, bw), coeffs[0])

            pred[by:by + bh, bx:bx + bw] = np.round(block_pred).astype(np.int32)

    return pred


def select_best_mode_per_block(ch, preds_dict, block_size=16):
    """Select the best predictor for each block.

    Uses Sum of Absolute Differences (SAD) as a fast proxy for
    entropy. Returns mode grid and optimal residuals.

    Args:
        ch: original channel (int32)
        preds_dict: dict of predictor_name -> prediction array
        block_size: block dimensions

    Returns:
        modes: 2D array of mode indices per block
        residuals: optimal residual array (int16)
        mode_names: list of mode names in index order
    """
    h, w = ch.shape
    i32 = ch.astype(np.int32)

    pred_names = list(preds_dict.keys())
    pred_arrays = [preds_dict[name] for name in pred_names]
    n_modes = len(pred_arrays)

    bh_count = (h + block_size - 1) // block_size
    bw_count = (w + block_size - 1) // block_size

    modes = np.zeros((bh_count, bw_count), dtype=np.uint8)
    residuals = np.zeros((h, w), dtype=np.int16)

    for by_idx in range(bh_count):
        by = by_idx * block_size
        bh = min(block_size, h - by)
        for bx_idx in range(bw_count):
            bx = bx_idx * block_size
            bw = min(block_size, w - bx)

            block = i32[by:by + bh, bx:bx + bw]

            best_sad = float('inf')
            best_mode = 0
            best_res = None

            for mode_idx, pred in enumerate(pred_arrays):
                res = block - pred[by:by + bh, bx:bx + bw]
                sad = np.sum(np.abs(res))
                if sad < best_sad:
                    best_sad = sad
                    best_mode = mode_idx
                    best_res = res

            modes[by_idx, bx_idx] = best_mode
            residuals[by:by + bh, bx:bx + bw] = best_res.astype(np.int16)

    return modes, residuals, pred_names


def compute_cross_channel_prediction(target_ch, ref_ch):
    """Predict target channel from reference channel (vectorized).

    Fits a per-row adaptive linear model: target ≈ a * ref + b.
    This exploits the strong correlation between color channels.

    Novel aspect: the model adapts PER ROW, capturing how the
    color relationship changes across the image (e.g., sky vs sand
    have different R-G relationships).
    """
    h, w = target_ch.shape
    t = target_ch.astype(np.float64)
    r = ref_ch.astype(np.float64)
    pred = np.zeros_like(target_ch, dtype=np.int32)

    # Global model as starting point
    r_mean = r.mean()
    t_mean = t.mean()
    cov = ((r - r_mean) * (t - t_mean)).mean()
    var_r = ((r - r_mean) ** 2).mean() + 1e-10
    a_global = cov / var_r
    b_global = t_mean - a_global * r_mean

    # Per-row adaptive prediction
    # Use exponentially weighted moving model
    a, b = a_global, b_global
    alpha = 0.95  # decay factor

    for y in range(h):
        row_pred = np.round(a * r[y, :] + b).astype(np.int32)
        pred[y, :] = row_pred

        # Update model from this row's actual values
        if y < h - 1:
            row_r = r[y, :]
            row_t = t[y, :]
            row_r_mean = row_r.mean()
            row_t_mean = row_t.mean()
            row_cov = ((row_r - row_r_mean) * (row_t - row_t_mean)).mean()
            row_var = ((row_r - row_r_mean) ** 2).mean() + 1e-10
            a_row = row_cov / row_var
            b_row = row_t_mean - a_row * row_r_mean

            a = alpha * a + (1 - alpha) * a_row
            b = alpha * b + (1 - alpha) * b_row

    return pred


def encode_channel(ch, ref_channels=None, block_size=16):
    """Full prediction pipeline for one channel.

    1. Compute all vectorized predictors
    2. Optionally add cross-channel predictor
    3. Add polynomial field predictor
    4. Select best mode per block
    5. Return modes + residuals

    Returns:
        modes: block mode grid
        residuals: int16 residual array
        mode_names: list of mode names
    """
    # Standard predictors (vectorized, ~instant)
    preds = compute_all_predictors(ch)

    # Novel: polynomial field predictor (captures gradients/curvature)
    poly_pred = compute_polynomial_predictor(ch, block_size)
    preds['polynomial'] = poly_pred

    # Cross-channel prediction (raw reference value)
    # The decoder uses ref[y,x] directly, so encoder must match.
    # The raw reference is a decent predictor for correlated channels.
    if ref_channels:
        for i, ref in enumerate(ref_channels):
            preds[f'cross_ch{i}'] = ref.astype(np.int32)

    # Select best mode per block
    modes, residuals, mode_names = select_best_mode_per_block(
        ch, preds, block_size
    )

    return modes, residuals, mode_names


def decode_channel(modes, residuals, mode_names, ref_channels=None,
                   block_size=16, ch_shape=None):
    """Reconstruct channel from modes + residuals.

    The decoder recomputes the same predictors from already-decoded data,
    then adds residuals. Since it's lossless, reconstructed == original.

    For the initial block row/column, context is limited,
    but predictors handle this gracefully with defaults.
    """
    h, w = ch_shape if ch_shape else residuals.shape
    # We need to reconstruct progressively (block by block)
    # But since all our predictors are causal (left/above),
    # and the modes tell us which predictor to use,
    # we can reconstruct in raster block order.

    # For the VECTORIZED predictors, we need the full channel.
    # Since this is lossless, we can compute:
    # actual = prediction + residual
    # But prediction depends on the actual values of neighbors!

    # Solution: reconstruct block-by-block in raster order.
    # For each block, compute predictors from already-reconstructed data.

    recon = np.zeros((h, w), dtype=np.int32)
    bh_count, bw_count = modes.shape

    for by_idx in range(bh_count):
        by = by_idx * block_size
        bh = min(block_size, h - by)
        for bx_idx in range(bw_count):
            bx = bx_idx * block_size
            bw = min(block_size, w - bx)

            mode = modes[by_idx, bx_idx]
            mode_name = mode_names[mode]

            # Reconstruct PIXEL BY PIXEL within the block.
            # This is critical: each pixel's prediction depends on
            # already-reconstructed neighbors (including earlier
            # pixels in the same block).
            _reconstruct_block_pixelwise(
                recon, residuals, mode_name,
                by, bx, bh, bw, ref_channels, block_size
            )

    return recon


def _reconstruct_block_pixelwise(recon, residuals, mode_name,
                                  by, bx, bh, bw, ref_channels, block_size):
    """Reconstruct a block pixel-by-pixel, updating recon as we go.

    CRITICAL: We must process pixels in raster order within the block
    and update recon immediately, because later pixels' predictions
    depend on earlier pixels' reconstructed values.
    """
    h, w = recon.shape
    D = BORDER_DEFAULT

    if mode_name == 'polynomial':
        # Polynomial: compute whole-block prediction from context
        poly_pred = _polynomial_block_predict(recon, by, bx, bh, bw, block_size)
        recon[by:by + bh, bx:bx + bw] = (
            poly_pred + residuals[by:by + bh, bx:bx + bw].astype(np.int32)
        )
        return

    if mode_name.startswith('cross_ch'):
        ch_idx = int(mode_name[-1])
        if ref_channels and ch_idx < len(ref_channels):
            ref = ref_channels[ch_idx]
            for dy in range(bh):
                for dx in range(bw):
                    y, x = by + dy, bx + dx
                    pred_val = int(ref[y, x])
                    recon[y, x] = pred_val + int(residuals[y, x])
            return

    for dy in range(bh):
        for dx in range(bw):
            y, x = by + dy, bx + dx
            left = int(recon[y, x - 1]) if x > 0 else D
            above = int(recon[y - 1, x]) if y > 0 else D
            ul = int(recon[y - 1, x - 1]) if y > 0 and x > 0 else D
            ur = int(recon[y - 1, x + 1]) if y > 0 and x < w - 1 else D
            left2 = int(recon[y, x - 2]) if x > 1 else D
            above2 = int(recon[y - 2, x]) if y > 1 else D

            if mode_name == 'left':
                pred_val = left
            elif mode_name == 'above':
                pred_val = above
            elif mode_name == 'gradient':
                pred_val = left + above - ul
            elif mode_name == 'paeth':
                p = left + above - ul
                pa, pb, pc = abs(p - left), abs(p - above), abs(p - ul)
                if pa <= pb and pa <= pc:
                    pred_val = left
                elif pb <= pc:
                    pred_val = above
                else:
                    pred_val = ul
            elif mode_name == 'average':
                pred_val = (left + above + 1) >> 1
            elif mode_name == 'edge_adaptive':
                dxx = abs(above - ul)
                dyy = abs(left - ul)
                if dxx > dyy + 4:
                    pred_val = above
                elif dyy > dxx + 4:
                    pred_val = left
                else:
                    pred_val = left + above - ul
            elif mode_name == 'median3':
                vals = sorted([left, above, left + above - ul])
                pred_val = vals[1]
            elif mode_name == 'extrapolate':
                h_ext = 2 * left - left2
                v_ext = 2 * above - above2
                if abs(left - left2) < abs(above - above2):
                    pred_val = h_ext
                else:
                    pred_val = v_ext
            elif mode_name == 'weighted_grad':
                wl = 1.0 / (abs(left - ul) + 1)
                wa = 1.0 / (abs(above - ul) + 1)
                pred_val = int(round((left * wl + above * wa) / (wl + wa)))
            else:
                pred_val = left

            recon[y, x] = pred_val + int(residuals[y, x])


def _compute_block_prediction(recon, mode_name, by, bx, bh, bw,
                               ref_channels, block_size):
    """Compute prediction for a single block from reconstructed context.
    NOTE: This is the OLD version, kept for reference. Use _reconstruct_block_pixelwise instead.
    """
    h, w = recon.shape

    y_start = max(0, by - 2)
    x_start = max(0, bx - 2)
    y_end = min(h, by + bh)
    x_end = min(w, bx + bw + 2)

    # For pixel-level predictors, compute within the block
    pred = np.zeros((bh, bw), dtype=np.int32)

    D = BORDER_DEFAULT
    for dy in range(bh):
        for dx in range(bw):
            y, x = by + dy, bx + dx
            left = int(recon[y, x - 1]) if x > 0 else D
            above = int(recon[y - 1, x]) if y > 0 else D
            ul = int(recon[y - 1, x - 1]) if y > 0 and x > 0 else D
            ur = int(recon[y - 1, x + 1]) if y > 0 and x < w - 1 else D
            left2 = int(recon[y, x - 2]) if x > 1 else D
            above2 = int(recon[y - 2, x]) if y > 1 else D

            if mode_name == 'left':
                pred[dy, dx] = left
            elif mode_name == 'above':
                pred[dy, dx] = above
            elif mode_name == 'gradient':
                pred[dy, dx] = left + above - ul
            elif mode_name == 'paeth':
                p = left + above - ul
                pa, pb, pc = abs(p - left), abs(p - above), abs(p - ul)
                if pa <= pb and pa <= pc:
                    pred[dy, dx] = left
                elif pb <= pc:
                    pred[dy, dx] = above
                else:
                    pred[dy, dx] = ul
            elif mode_name == 'average':
                pred[dy, dx] = (left + above + 1) >> 1
            elif mode_name == 'edge_adaptive':
                dxx = abs(above - ul)
                dyy = abs(left - ul)
                if dxx > dyy + 4:
                    pred[dy, dx] = above
                elif dyy > dxx + 4:
                    pred[dy, dx] = left
                else:
                    pred[dy, dx] = left + above - ul
            elif mode_name == 'median3':
                vals = sorted([left, above, left + above - ul])
                pred[dy, dx] = vals[1]
            elif mode_name == 'extrapolate':
                h_ext = 2 * left - left2
                v_ext = 2 * above - above2
                if abs(left - left2) < abs(above - above2):
                    pred[dy, dx] = h_ext
                else:
                    pred[dy, dx] = v_ext
            elif mode_name == 'weighted_grad':
                wl = 1.0 / (abs(left - ul) + 1)
                wa = 1.0 / (abs(above - ul) + 1)
                pred[dy, dx] = int(round((left * wl + above * wa) / (wl + wa)))
            elif mode_name == 'polynomial':
                # Recompute polynomial prediction from context
                pred[:, :] = _polynomial_block_predict(
                    recon, by, bx, bh, bw, block_size
                )
                return pred
            elif mode_name.startswith('cross_ch'):
                ch_idx = int(mode_name[-1])
                if ref_channels and ch_idx < len(ref_channels):
                    ref = ref_channels[ch_idx]
                    # Simple: use reference value directly
                    # (full linear model would need running stats)
                    pred[dy, dx] = int(ref[y, x])
                else:
                    pred[dy, dx] = left
            else:
                pred[dy, dx] = left

    return pred


def _polynomial_block_predict(recon, by, bx, bh, bw, block_size):
    """Recompute polynomial prediction from reconstructed context."""
    h, w = recon.shape
    ctx_ys, ctx_xs, ctx_vals = [], [], []

    # Row(s) above
    if by > 0:
        for dy in range(max(0, by - 2), by):
            for dx in range(max(0, bx - 2), min(w, bx + bw + 2)):
                ctx_ys.append(dy)
                ctx_xs.append(dx)
                ctx_vals.append(int(recon[dy, dx]))

    # Column(s) left
    if bx > 0:
        for dy in range(by, min(h, by + bh)):
            for dx in range(max(0, bx - 2), bx):
                ctx_ys.append(dy)
                ctx_xs.append(dx)
                ctx_vals.append(int(recon[dy, dx]))

    if len(ctx_vals) < 3:
        if ctx_vals:
            return np.full((bh, bw), int(np.mean(ctx_vals)), dtype=np.int32)
        return np.full((bh, bw), 128, dtype=np.int32)

    ctx_ys = np.array(ctx_ys, dtype=np.float64)
    ctx_xs = np.array(ctx_xs, dtype=np.float64)
    ctx_vals = np.array(ctx_vals, dtype=np.float64)

    y_center = by + bh / 2
    x_center = bx + bw / 2
    scale = max(bh, bw, 1)
    ny = (ctx_ys - y_center) / scale
    nx = (ctx_xs - x_center) / scale

    n_ctx = len(ctx_vals)
    if n_ctx >= 6:
        A = np.column_stack([np.ones(n_ctx), nx, ny, nx ** 2, nx * ny, ny ** 2])
    elif n_ctx >= 3:
        A = np.column_stack([np.ones(n_ctx), nx, ny])
    else:
        A = np.column_stack([np.ones(n_ctx)])

    try:
        coeffs = np.linalg.lstsq(A, ctx_vals, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.full((bh, bw), int(np.mean(ctx_vals)), dtype=np.int32)

    block_ys, block_xs = np.mgrid[by:by + bh, bx:bx + bw]
    bny = (block_ys.astype(np.float64) - y_center) / scale
    bnx = (block_xs.astype(np.float64) - x_center) / scale

    if len(coeffs) >= 6:
        result = (coeffs[0] + coeffs[1] * bnx + coeffs[2] * bny
                  + coeffs[3] * bnx ** 2 + coeffs[4] * bnx * bny
                  + coeffs[5] * bny ** 2)
    elif len(coeffs) >= 3:
        result = coeffs[0] + coeffs[1] * bnx + coeffs[2] * bny
    else:
        result = np.full((bh, bw), coeffs[0])

    return np.round(result).astype(np.int32)
