"""NEURAL CODEC: Image as a program that runs at decode time.

The compressed file IS a neural network.
The decoder RUNS the network to generate pixels.
Exact residual for lossless guarantee.

Architecture:
  1. SIREN-style MLP: f(x, y) -> (Y, Co, Cg)
  2. Trained on the specific image during encoding
  3. Weights quantized to int8 for compact storage
  4. Residual = original - round(network_output)
  5. Compressed file = quantized_weights + compressed_residual

The key insight for the user's idea about transformers/SLMs:
- A PRETRAINED model (shared by encoder+decoder) provides a prior
- The encoder finds a LATENT CODE that makes the model output ~this image
- Store: latent_code + corrections
- If the model is good, corrections are very sparse

For this implementation, we use the simpler approach:
- Train a small MLP from scratch on THIS specific image
- No pretrained model needed (self-contained)
- The network learns the image's smooth structure
- Corrections handle the detail
"""

import numpy as np
from PIL import Image
import math, time
from collections import Counter
from prism.transforms import rgb_to_ycocg_r, zigzag_encode


def build_siren_features(h, w, n_freqs=8):
    """Build SIREN-style sinusoidal coordinate features.

    These capture multi-scale spatial patterns:
    - Low frequencies: global gradients, sky-to-sand transition
    - High frequencies: texture, edges
    """
    ys = np.linspace(-1, 1, h)
    xs = np.linspace(-1, 1, w)
    X, Y = np.meshgrid(xs, ys)
    x_flat = X.flatten()
    y_flat = Y.flatten()

    features = [x_flat, y_flat]
    for f in range(1, n_freqs + 1):
        omega = f * np.pi
        features.extend([
            np.sin(omega * x_flat),
            np.cos(omega * x_flat),
            np.sin(omega * y_flat),
            np.cos(omega * y_flat),
            np.sin(omega * (x_flat + y_flat) / np.sqrt(2)),
            np.cos(omega * (x_flat - y_flat) / np.sqrt(2)),
        ])

    return np.column_stack(features)


def train_neural_codec(image_path, hidden_sizes=[256], n_freqs=8):
    """Train a neural network to represent the image.

    Returns (weights, residual, stats).
    """
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = img.shape
    tp = h * w
    ycocg = rgb_to_ycocg_r(img)

    features = build_siren_features(h, w, n_freqs)
    n_feat = features.shape[1]

    print(f"Neural codec: {n_feat} input features, hidden={hidden_sizes}")

    total_params = 0
    total_residual_entropy = 0
    all_residuals = []
    all_weights = []

    for ch_idx, name in enumerate(['Y', 'Co', 'Cg']):
        target = ycocg[:, :, ch_idx].flatten().astype(np.float64)

        # Build network layer by layer
        current_input = features
        stored_weights = []

        for layer_idx, hs in enumerate(hidden_sizes):
            n_in = current_input.shape[1]

            # Random projection (seeded, not stored)
            np.random.seed(42 + ch_idx * 1000 + layer_idx * 100)
            W_random = np.random.randn(n_in, hs) * np.sqrt(2.0 / n_in)
            b_random = np.zeros(hs)

            # Hidden activations
            pre_act = current_input @ W_random + b_random
            # Use both ReLU and sin for different representation power
            hidden_relu = np.maximum(pre_act, 0)
            hidden_sin = np.sin(pre_act)
            current_input = np.column_stack([hidden_relu, hidden_sin])

        # Output layer (THIS is what we store)
        out_input = np.column_stack([current_input, np.ones(tp)])
        n_out_params = out_input.shape[1]

        W_out = np.linalg.lstsq(out_input, target, rcond=None)[0]
        stored_weights.append(W_out)

        # Predict
        predicted = out_input @ W_out
        pred_int = np.round(predicted).astype(np.int32).reshape(h, w)
        residual = ycocg[:, :, ch_idx].astype(np.int32) - pred_int

        # Stats
        n_params = n_out_params
        zz = zigzag_encode(residual)
        bps = -sum((c/tp) * math.log2(c/tp)
                   for c in Counter(zz.flatten().tolist()).values())

        total_params += n_params
        total_residual_entropy += bps * tp
        all_residuals.append(residual)
        all_weights.append(stored_weights)

        n_zeros = (residual == 0).sum()
        print(f"  {name}: {n_params} params, {bps:.3f} bps, "
              f"zeros={n_zeros/tp*100:.0f}%")

    param_bytes = total_params * 2 * 3  # float16, 3 channels
    residual_bytes = total_residual_entropy / 8
    total_bytes = param_bytes + residual_bytes

    print(f"\n  Params: {param_bytes/1024:.1f} KB")
    print(f"  Residual: {residual_bytes/1024:.0f} KB")
    print(f"  TOTAL: {total_bytes/1024:.0f} KB")

    return all_weights, all_residuals, total_bytes


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "img.png"

    print("=== NEURAL CODEC: Image as Program ===\n")

    for hidden in [[128], [256], [512], [256, 128]]:
        for nf in [6, 8, 12]:
            print(f"\n--- hidden={hidden}, freqs={nf} ---")
            train_neural_codec(path, hidden_sizes=hidden, n_freqs=nf)
