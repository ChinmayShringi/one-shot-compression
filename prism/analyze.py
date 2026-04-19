"""Quick analysis: measure prediction quality and theoretical compression.

Runs on a small crop first to be fast, then scales up.
Compares our novel prediction against standard approaches.
"""

import numpy as np
from PIL import Image
import time
import math
from collections import Counter


def entropy_bits_per_symbol(data):
    """Shannon entropy in bits per symbol."""
    flat = data.flatten()
    counts = Counter(flat.tolist())
    total = len(flat)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def total_bits(data):
    """Total bits needed at entropy rate."""
    h = entropy_bits_per_symbol(data)
    return h * len(data.flatten())


def zigzag(x):
    x = x.astype(np.int32)
    return np.where(x >= 0, 2 * x, -2 * x - 1).astype(np.uint32)


def analyze_prediction_quality(img_path, crop_size=256):
    """Measure how well different prediction schemes work."""
    from PIL import Image

    img = np.array(Image.open(img_path).convert('RGB'))
    h, w, _ = img.shape

    # Crop for speed
    cy, cx = h // 2 - crop_size // 2, w // 2 - crop_size // 2
    crop = img[cy:cy + crop_size, cx:cx + crop_size]
    ch, cw, _ = crop.shape
    n_pixels = ch * cw
    print(f"Analyzing {cw}x{ch} crop ({n_pixels} pixels per channel)")
    print(f"Full image: {w}x{h}")
    print()

    from prism.transforms import rgb_to_ycocg_r, ycocg_r_to_rgb

    # Test color transforms
    ycocg = rgb_to_ycocg_r(crop)
    rgb_back = ycocg_r_to_rgb(ycocg)
    assert np.array_equal(crop, rgb_back), "YCoCg-R not reversible!"

    for cs_name, data in [("RGB", crop), ("YCoCg-R", ycocg)]:
        print(f"=== {cs_name} ===")
        total_kb = 0

        for ch_idx in range(3):
            ch = data[:, :, ch_idx].astype(np.int32)
            ch_names = {
                "RGB": ["R", "G", "B"],
                "YCoCg-R": ["Y", "Co", "Cg"]
            }
            ch_name = ch_names[cs_name][ch_idx]

            raw_bps = entropy_bits_per_symbol(ch)

            # Standard predictors (vectorized for speed)
            results = []

            # Left
            pred = np.zeros_like(ch)
            pred[:, 1:] = ch[:, :-1]
            pred[:, 0] = 128
            res = zigzag(ch - pred)
            results.append(("Left", entropy_bits_per_symbol(res)))

            # Above
            pred = np.zeros_like(ch)
            pred[1:, :] = ch[:-1, :]
            pred[0, :] = 128
            res = zigzag(ch - pred)
            results.append(("Above", entropy_bits_per_symbol(res)))

            # Gradient (left + above - upper_left)
            left = np.zeros_like(ch); left[:, 1:] = ch[:, :-1]; left[:, 0] = 128
            above = np.zeros_like(ch); above[1:, :] = ch[:-1, :]; above[0, :] = 128
            ul = np.zeros_like(ch); ul[1:, 1:] = ch[:-1, :-1]; ul[0, :] = 128; ul[:, 0] = 128
            pred = np.clip(left + above - ul, 0, 511)
            res = zigzag(ch - pred)
            results.append(("Gradient", entropy_bits_per_symbol(res)))

            # Paeth
            p = left + above - ul
            pa = np.abs(p - left)
            pb = np.abs(p - above)
            pc = np.abs(p - ul)
            pred = np.where((pa <= pb) & (pa <= pc), left,
                           np.where(pb <= pc, above, ul))
            res = zigzag(ch - pred)
            results.append(("Paeth", entropy_bits_per_symbol(res)))

            # Average
            pred = (left + above + 1) // 2
            res = zigzag(ch - pred)
            results.append(("Average", entropy_bits_per_symbol(res)))

            # Edge-adaptive (novel)
            dx = np.abs(above - ul)
            dy = np.abs(left - ul)
            pred = np.where(dx > dy + 4, above,
                           np.where(dy > dx + 4, left,
                           np.clip(left + above - ul, 0, 511)))
            res = zigzag(ch - pred)
            results.append(("EdgeAdaptive", entropy_bits_per_symbol(res)))

            # Median of (left, above, gradient)
            grad = np.clip(left + above - ul, 0, 511)
            stacked = np.stack([left, above, grad], axis=-1)
            pred = np.median(stacked, axis=-1).astype(np.int32)
            res = zigzag(ch - pred)
            results.append(("Median3", entropy_bits_per_symbol(res)))

            # Best per-pixel (oracle): pick the predictor with smallest |residual| per pixel
            all_preds = [
                left, above, np.clip(left + above - ul, 0, 511),
                np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, above, ul)),
                (left + above + 1) // 2,
                np.where(dx > dy + 4, above, np.where(dy > dx + 4, left, np.clip(left + above - ul, 0, 511))),
                pred
            ]
            all_residuals = [np.abs(ch - p) for p in all_preds]
            stacked_res = np.stack(all_residuals, axis=-1)
            best_idx = np.argmin(stacked_res, axis=-1)
            oracle_pred = np.choose(best_idx, all_preds)
            res = zigzag(ch - oracle_pred)
            results.append(("Oracle(best/pixel)", entropy_bits_per_symbol(res)))

            results.sort(key=lambda x: x[1])
            best_name, best_bps = results[0]
            best_kb = best_bps * n_pixels / 8 / 1024

            # Scale to full image
            scale = (w * h) / n_pixels
            full_kb = best_kb * scale

            print(f"  {ch_name}: raw={raw_bps:.2f} bps | best={best_name} @ {best_bps:.3f} bps ({best_kb:.1f} KB crop, ~{full_kb:.0f} KB full)")
            for name, bps in results[:6]:
                print(f"    {name:20s}: {bps:.3f} bps")

            total_kb += full_kb

        print(f"  TOTAL (estimated full image): ~{total_kb:.0f} KB")
        print()

    # Now test our adaptive mixer on the crop
    print("=== PRISM Adaptive Mixer (on crop) ===")
    from prism.predict import compute_residuals_adaptive
    from prism.transforms import rgb_to_ycocg_r

    ycocg_crop = rgb_to_ycocg_r(crop)

    total_prism_kb = 0
    decoded_channels = []

    for ch_idx in range(3):
        ch = ycocg_crop[:, :, ch_idx].astype(np.int32)
        ch_name = ["Y", "Co", "Cg"][ch_idx]

        t0 = time.time()
        ref = decoded_channels if decoded_channels else None
        residuals = compute_residuals_adaptive(ch, ref)
        dt = time.time() - t0

        zz = zigzag(residuals)
        bps = entropy_bits_per_symbol(zz)
        crop_kb = bps * n_pixels / 8 / 1024
        full_kb = crop_kb * (w * h) / n_pixels

        print(f"  {ch_name}: {bps:.3f} bps ({crop_kb:.1f} KB crop, ~{full_kb:.0f} KB full) [{dt:.1f}s]")
        total_prism_kb += full_kb

        decoded_channels.append(ch)  # Use original for now (upper bound)

    print(f"  TOTAL estimated: ~{total_prism_kb:.0f} KB")
    print()

    # Compare against baselines
    import os
    png_size = os.path.getsize(img_path)
    print("=== Comparison ===")
    print(f"  Raw RGB:      {w * h * 3 / 1024:.0f} KB")
    print(f"  PNG:          {png_size / 1024:.0f} KB")
    print(f"  PRISM (est):  ~{total_prism_kb:.0f} KB")
    print(f"  JPEG-XL:      917 KB (benchmark)")
    print(f"  Improvement vs PNG: {(1 - total_prism_kb / (png_size / 1024)) * 100:.1f}%")


if __name__ == "__main__":
    analyze_prediction_quality(
        "/Users/chinmay_shringi/Desktop/time-pass/compress/img.png",
        crop_size=128  # Start small for speed
    )
