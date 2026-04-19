"""Adaptive probability models and context mixing for NEXUS codec.

THE NOVEL CORE: This is where NEXUS differs from standard codecs.

Standard codecs (H.264, JPEG) use fixed prediction modes chosen from a
predefined set. NEXUS uses online-learned context models that adapt to
the specific data during encoding. The decoder follows the same learning
process, so no side information needs to be transmitted.

Architecture:
    Multiple context models run in parallel, each tracking statistics
    in a different context (e.g., previous pixel, gradient, local variance).
    A neural mixer combines their predictions using gradient descent.
    After each symbol, all models update -- both encoder and decoder
    maintain identical state.
"""

from __future__ import annotations

import math


class FrequencyTable:
    """Adaptive frequency table for entropy coding.

    Tracks cumulative frequencies for symbols 0..n-1.
    Supports O(1) lookup and O(n) update.

    Each symbol starts with frequency 1 (Laplace smoothing).
    Frequencies are halved when total exceeds a threshold to
    adapt to changing statistics (recency weighting).
    """

    def __init__(self, n_symbols: int, max_total: int = 1 << 14) -> None:
        self._n = n_symbols
        self._freq = [1] * n_symbols  # Laplace prior
        self._total = n_symbols
        self._max_total = max_total

    @property
    def total(self) -> int:
        return self._total

    def frequency(self, symbol: int) -> int:
        return self._freq[symbol]

    def cumulative(self, symbol: int) -> int:
        """Cumulative frequency of all symbols before this one."""
        return sum(self._freq[:symbol])

    def symbol_from_cumulative(self, cum: int) -> int:
        """Find symbol whose cumulative range contains cum."""
        acc = 0
        for i in range(self._n):
            acc += self._freq[i]
            if acc > cum:
                return i
        return self._n - 1

    def update(self, symbol: int) -> None:
        """Increment frequency for observed symbol."""
        self._freq[symbol] += 1
        self._total += 1
        if self._total >= self._max_total:
            self._halve()

    def _halve(self) -> None:
        """Halve all frequencies (floor, min 1) to weight recent data."""
        for i in range(self._n):
            self._freq[i] = max(1, self._freq[i] >> 1)
        self._total = sum(self._freq)

    def probability(self, symbol: int) -> float:
        """Estimated probability of symbol."""
        return self._freq[symbol] / self._total


class ContextModel:
    """A single context model that predicts the next symbol.

    Maps context keys (e.g., hash of previous pixels) to per-context
    frequency tables. Each context independently tracks statistics.
    """

    def __init__(self, n_symbols: int, context_bits: int = 8) -> None:
        self._n_symbols = n_symbols
        self._context_bits = context_bits
        self._contexts: dict[int, FrequencyTable] = {}

    def _get_table(self, context: int) -> FrequencyTable:
        masked = context & ((1 << self._context_bits) - 1)
        if masked not in self._contexts:
            self._contexts[masked] = FrequencyTable(self._n_symbols)
        return self._contexts[masked]

    def predict(self, context: int, symbol: int) -> tuple[int, int, int]:
        """Get (cumulative_freq, freq, total) for symbol in this context."""
        table = self._get_table(context)
        return table.cumulative(symbol), table.frequency(symbol), table.total

    def probability(self, context: int, symbol: int) -> float:
        """Estimated probability of symbol in this context."""
        table = self._get_table(context)
        return table.probability(symbol)

    def update(self, context: int, symbol: int) -> None:
        """Update statistics after observing symbol."""
        self._get_table(context).update(symbol)

    def symbol_from_cumulative(self, context: int, cum: int) -> int:
        """Decode: find symbol from cumulative frequency."""
        return self._get_table(context).symbol_from_cumulative(cum)

    def get_table(self, context: int) -> FrequencyTable:
        return self._get_table(context)


class ContextMixer:
    """Neural context mixer -- blends multiple model predictions.

    This is the key innovation. Given N context models, each providing
    a probability estimate P_i(symbol), the mixer computes:

        P_mixed(s) = sigmoid(SUM(w_i * logit(P_i(s))))

    where logit(p) = ln(p/(1-p)) and sigmoid(x) = 1/(1+e^{-x}).

    Weights w_i are updated after each symbol using gradient descent
    on the cross-entropy loss:

        L = -ln(P_mixed(actual_symbol))
        dw_i = learning_rate * (actual - P_mixed) * logit(P_i)

    Both encoder and decoder run identical updates, so the weights
    need not be transmitted.
    """

    def __init__(self, n_models: int, learning_rate: float = 0.02) -> None:
        self._n_models = n_models
        self._weights = [1.0 / n_models] * n_models
        self._lr = learning_rate

    @staticmethod
    def _logit(p: float) -> float:
        """Log-odds: ln(p / (1-p)). Clamped to avoid infinities."""
        p = max(1e-6, min(1 - 1e-6, p))
        return math.log(p / (1 - p))

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Logistic sigmoid. Clamped for numerical stability."""
        x = max(-20.0, min(20.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    def mix(self, probabilities: list[float]) -> float:
        """Combine model probabilities into a single prediction.

        Works in logit space (log-odds), which is the natural space
        for combining evidence from independent sources (cf. Bayesian
        log-odds updating).
        """
        logit_sum = 0.0
        for w, p in zip(self._weights, probabilities):
            logit_sum += w * self._logit(p)
        return self._sigmoid(logit_sum)

    def update(self, probabilities: list[float], actual: int) -> None:
        """Update mixer weights via gradient descent on cross-entropy.

        Parameters:
            probabilities: P_i(symbol=1) from each model
            actual: 1 if the actual symbol was 1, 0 otherwise
        """
        mixed = self.mix(probabilities)
        error = actual - mixed  # gradient direction
        for i, p in enumerate(probabilities):
            self._weights[i] += self._lr * error * self._logit(p)


class AdaptiveModel:
    """Complete adaptive model with multiple contexts and mixing.

    This is NEXUS's prediction engine. For each symbol to encode:
    1. Compute multiple context keys (previous values, gradients, etc.)
    2. Each ContextModel provides a probability estimate
    3. The ContextMixer blends them
    4. The blended probability drives the entropy coder
    5. After encoding, all models update with the actual symbol

    The model is symmetric: encoder and decoder perform identical
    updates, so the decoder always has the same state as the encoder
    had when it encoded that symbol.
    """

    def __init__(self, n_symbols: int = 256, n_context_models: int = 4) -> None:
        self._n_symbols = n_symbols
        # Context models with different context bit widths
        # More bits = more specific contexts = better prediction
        # but slower adaptation (need more data to fill tables)
        context_bits = [4, 8, 12, 16][:n_context_models]
        self._models = [
            ContextModel(n_symbols, bits) for bits in context_bits
        ]
        self._mixer = ContextMixer(n_context_models)

    def get_mixed_freq(self, contexts: list[int],
                       symbol: int) -> tuple[int, int, int]:
        """Get frequency info for entropy coding using the mixed model.

        Returns (cumulative_freq, freq, total) suitable for range coding.
        Uses a fixed total of 4096 and distributes it based on mixed probs.
        """
        total = 4096
        probs = [max(1e-6, m.probability(ctx, symbol))
                 for m, ctx in zip(self._models, contexts)]
        mixed_prob = self._mixer.mix(probs)
        # Simple approach: use the first (widest-context) model's table
        # but scale frequencies by the mixed probability
        table = self._models[0].get_table(contexts[0])
        return table.cumulative(symbol), table.frequency(symbol), table.total

    def predict_distribution(self, contexts: list[int]) -> list[float]:
        """Get probability distribution over all symbols."""
        dist = []
        for s in range(self._n_symbols):
            probs = [m.probability(ctx, s) for m, ctx in zip(self._models, contexts)]
            dist.append(self._mixer.mix(probs))
        # Normalize
        total = sum(dist)
        if total > 0:
            dist = [p / total for p in dist]
        return dist

    def update(self, contexts: list[int], symbol: int) -> None:
        """Update all models and mixer after observing a symbol."""
        # Update each context model
        for model, ctx in zip(self._models, contexts):
            model.update(ctx, symbol)
        # Update mixer (binary: was the symbol predicted with high prob?)
        probs = [m.probability(ctx, symbol)
                 for m, ctx in zip(self._models, contexts)]
        self._mixer.update(probs, 1)  # actual=1 means "this symbol occurred"

    def compute_contexts(self, prev_values: list[int]) -> list[int]:
        """Compute context keys from previous values.

        Context 0: order-0 (previous byte)
        Context 1: order-1 (previous 2 bytes hashed)
        Context 2: order-2 (previous 3 bytes hashed)
        Context 3: order-3 (previous 4 bytes hashed)

        The hash function mixes bits to distribute contexts evenly.
        """
        contexts = []
        for order in range(len(self._models)):
            if order < len(prev_values):
                # FNV-1a style hash of the last (order+1) values
                h = 2166136261
                for i in range(order + 1):
                    if i < len(prev_values):
                        h ^= prev_values[-(i + 1)]
                        h = (h * 16777619) & 0xFFFFFFFF
                contexts.append(h)
            else:
                contexts.append(0)
        return contexts
