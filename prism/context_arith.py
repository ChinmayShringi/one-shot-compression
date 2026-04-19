"""Contextual Adaptive Arithmetic Coder.

The key insight: different parts of the image have different statistics.
A pixel in smooth sky has residual ≈ 0 (nearly certain).
A pixel at an edge has larger residuals (uncertain).

Traditional: one probability model for all pixels.
Contextual: different model for each context → MUCH better probabilities.

Context = f(local gradient magnitude, local direction, previous residual)

This is the #1 technique that separates JPEG-XL (917KB) from PNG (2699KB).
"""

import numpy as np
from prism.arith import AdaptiveModel, ArithEncoder, ArithDecoder


class ContextualArithEncoder:
    """Arithmetic encoder with context-dependent probability models.

    Each context bucket gets its own adaptive frequency model.
    Symbols in smooth regions get a model heavily biased toward zero.
    Symbols in edge regions get a wider distribution.
    """

    def __init__(self, alphabet_size, n_contexts=64):
        self.alphabet_size = alphabet_size
        self.n_contexts = n_contexts
        self.models = [AdaptiveModel(alphabet_size) for _ in range(n_contexts)]
        self.enc = ArithEncoder()

    def encode(self, symbol, context):
        """Encode symbol using context-dependent model."""
        ctx = context % self.n_contexts
        model = self.models[ctx]
        cum_low, cum_high, total = model.get_range(symbol)
        self.enc.encode(cum_low, cum_high, total)
        model.update(symbol)

    def finish(self):
        self.enc.finish()
        return self.enc.finish()


class ContextualArithDecoder:
    """Arithmetic decoder with context-dependent probability models."""

    def __init__(self, data, alphabet_size, n_contexts=64):
        self.alphabet_size = alphabet_size
        self.n_contexts = n_contexts
        self.models = [AdaptiveModel(alphabet_size) for _ in range(n_contexts)]
        self.dec = ArithDecoder(data)

    def decode(self, context):
        """Decode one symbol using context-dependent model."""
        ctx = context % self.n_contexts
        model = self.models[ctx]
        total = model.total
        value = self.dec.get_value(total)
        symbol, cum_low, cum_high = model.find_symbol(value)
        self.dec.decode(cum_low, cum_high, total)
        model.update(symbol)
        return symbol


def compute_context_2d(residuals_so_far, y, x, h, w):
    """Compute 2D context from already-coded residuals.

    Context encodes:
    - Magnitude of neighboring residuals (are we in smooth or complex area?)
    - Direction of change
    """
    # Get already-decoded residual neighbors
    left_res = abs(int(residuals_so_far[y, x - 1])) if x > 0 else 0
    above_res = abs(int(residuals_so_far[y - 1, x])) if y > 0 else 0

    # Quantize magnitude: 0=zero, 1=tiny, 2=small, 3=medium, 4+=large
    def mag_q(v):
        if v == 0:
            return 0
        elif v <= 1:
            return 1
        elif v <= 3:
            return 2
        elif v <= 7:
            return 3
        elif v <= 15:
            return 4
        elif v <= 31:
            return 5
        elif v <= 63:
            return 6
        else:
            return 7

    ctx_left = mag_q(left_res)
    ctx_above = mag_q(above_res)

    return ctx_left * 8 + ctx_above  # 64 contexts
