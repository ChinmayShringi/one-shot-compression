# JPEG-XL Lossless Compression — Deep Technical Study

## Source: libjxl (https://github.com/libjxl/libjxl), BSD-3

---

## 1. Architecture Overview

JPEG-XL lossless uses the **Modular** coding path (not VarDCT which is for lossy).
The pipeline is:

```
Input RGB → Color Transform (RCT) → Squeeze Transform (multi-scale)
         → Per-pixel Prediction (weighted blend) → Residual
         → MA Decision Tree (context selection) → ANS Entropy Coding
         → Bitstream
```

---

## 2. The Weighted Predictor (THE key innovation)

**File**: `lib/jxl/modular/encoding/context_predict.h`

### 2.1 Four Sub-Predictors

JPEG-XL uses 4 parameterized predictors that are NOT independent — they incorporate
ERROR FEEDBACK from previous pixels:

```
P0 = W + NE - N                          (standard gradient)
P1 = N - ((teW + teN + teNE) * p1C) >> 5  (N corrected by error sum)
P2 = W - ((teW + teN + teNW) * p2C) >> 5  (W corrected by error sum)
P3 = N - (a*teNW + b*teN + c*teNE + d*(NN-N) + e*(NW-W)) >> 5
     (N corrected by 5-term error model)
```

Where `teW`, `teN`, `teNW`, `teNE` are the PREDICTION ERRORS at the corresponding
neighbor positions (the difference between what was predicted and what was actual).

**Why this matters**: P1-P3 are SELF-CORRECTING. If the predictor consistently
over-estimates in a region, the error terms grow, and the correction term pulls
the prediction back. This is like an adaptive filter that learns from its mistakes
IN REAL-TIME as it scans across the image.

### 2.2 Online Weight Update

The 4 predictions are combined via a WEIGHTED AVERAGE where weights depend on
recent prediction accuracy:

```cpp
// For each predictor, accumulate error at 3 neighbor positions
weights[i] = pred_errors[i][pos_N] + pred_errors[i][pos_NE] + pred_errors[i][pos_NW];
// Convert to weight (inverse of error)
weights[i] = ErrorWeight(weights[i], header.w[i]);
// ErrorWeight ≈ 4 + (maxweight << 24) / (error + 1)
```

The predictor with the lowest recent error gets the highest weight.
After each pixel, errors are updated:

```cpp
pred_errors[i][cur_row + x] = |prediction[i] - actual|;
// Also propagate error forward:
pred_errors[i][prev_row + x + 1] += |prediction[i] - actual|;
```

The forward propagation (`prev_row + x + 1`) means the error at position (x, y)
also affects the weight calculation for position (x+1, y+1) — creating a
diagonal error feedback path.

### 2.3 Clamping

When neighboring errors all have the same sign (all positive or all negative),
the prediction is CLAMPED to the range [min(W, NE, N), max(W, NE, N)]:

```cpp
if (((teN ^ teW) | (teN ^ teNW)) > 0) {
    // Different signs — just round
    return (pred + kPredictionRound) >> kPredExtraBits;
}
// Same sign — clamp to prevent overshooting
pixel_type_w mx = std::max(W, std::max(NE, N));
pixel_type_w mn = std::min(W, std::min(NE, N));
pred = std::max(mn, std::min(mx, pred));
```

This prevents the predictor from extrapolating beyond the observed range of neighbors.

### 2.4 Why this beats our approach

| PRISM / Hash Codec | JPEG-XL Weighted Predictor |
|---|---|
| Fixed predictors (MED, gradient, etc.) | Self-correcting predictors with error feedback |
| Select one predictor per pixel/block | Blend ALL predictors with adaptive weights |
| Weights don't adapt | Weights update every pixel based on recent errors |
| No error feedback in prediction | Prediction directly uses recent errors |
| Static context for entropy coding | MA tree creates dynamic contexts from properties |

---

## 3. The MA Decision Tree (Context Selection)

**File**: `lib/jxl/modular/encoding/enc_ma.cc`

### 3.1 Properties (Features)

For each pixel, 13+ properties are computed:

```
p[0] = channel index (static)
p[1] = group_id (static)
p[2] = y coordinate
p[3] = x coordinate
p[4] = |N| (abs value of north neighbor)
p[5] = |W| (abs value of west neighbor)
p[6] = N (signed north value)
p[7] = W (signed west value)
p[8] = W - prev_local_gradient  (gradient change)
p[9] = W + N - NW  (local gradient / Laplacian)
p[10] = W - NW  (horizontal gradient at top)
p[11] = NW - N  (diagonal gradient)
p[12] = N - NE  (horizontal gradient)
p[13] = N - NN  (vertical 2nd order)
p[14] = W - WW  (horizontal 2nd order)
p[kWPProp] = weighted predictor error signal

Plus for each reference channel (already decoded):
  |value|, value, |residual|, residual
```

### 3.2 Tree Building (Encoder-Side Optimization)

The encoder builds an OPTIMAL decision tree by:
1. Starting with all pixels in one leaf
2. For each leaf, trying ALL (property, threshold) splits
3. Computing the entropy reduction for each split
4. Choosing the split that gives the best bits saved minus tree overhead
5. Each leaf gets its OWN: predictor selection, context ID, offset, multiplier
6. Process repeats until no split improves compression beyond the threshold

This is a **rate-distortion optimized** context selection — it finds the exact
context boundaries that minimize total compressed size.

### 3.3 What the tree learns

The tree effectively learns rules like:
- "In smooth areas (|W-NW| < 3 and |N-NE| < 2), use the Weighted predictor
   with context #17"
- "Near strong horizontal edges (|N-NE| > 20), use the Left predictor
   with context #42"
- "In the sky channel, when gradient is close to zero, use context #5
   with a tight model"

Each context gets its own entropy model, so the ANS coder has maximally
informative probability estimates.

### 3.4 ANS vs Arithmetic Coding

JPEG-XL uses rANS (range Asymmetric Numeral Systems), not arithmetic coding.
ANS has similar compression efficiency but:
- Decoding is a single table lookup (faster)
- Encoding is equally fast
- No carries, no renormalization delays
- Better for SIMD/parallel implementation

---

## 4. The Squeeze Transform (Multi-Scale Decorrelation)

**File**: `lib/jxl/modular/transform/squeeze.h`, `squeeze.cc`

### 4.1 Basic Operation

Squeeze is a Haar-like integer wavelet:
```
Given adjacent pixels A, B:
  avg = (A + B) >> 1           (stored in one channel)
  diff = A - B - tendency      (stored in a new channel)
```

Where `tendency` is a SMOOTH TREND ESTIMATOR that biases the difference
toward zero in smooth regions:

```cpp
// SmoothTendency(B, a, n) where B=prev, a=current_avg, n=next_avg
// Estimates the expected difference based on smooth interpolation
// Only applies when B, a, n are monotonic (smooth gradient)
if (B >= a && a >= n) {
    diff = (4*B - 3*n - a + 6) / 12;  // cubic interpolation estimate
    // Clamp to prevent overshooting
}
```

### 4.2 Why tendency matters

Without tendency: `diff = A - B` (could be large in gradient areas)
With tendency: `diff = A - B - estimated_trend` (near-zero in smooth gradients)

This is the key insight: in a smooth gradient, consecutive pixels increase
linearly. The raw difference captures the gradient, but the tendency-corrected
difference captures only the deviation FROM the gradient — which is much smaller.

### 4.3 Multi-scale application

Squeeze is applied repeatedly (alternating horizontal and vertical) to create
a multi-resolution decomposition:

```
Original (WxH)
  → Squeeze horizontal: avg (W/2 x H) + diff (W/2 x H)
    → Squeeze vertical on avg: avg (W/2 x H/2) + diff (W/2 x H/2)
      → Squeeze horizontal: avg (W/4 x H/2) + diff (W/4 x H/2)
        ... (typically 3-4 levels)
```

The final structure is:
- A small low-frequency thumbnail (encodes cheaply)
- Multiple detail channels at increasing resolutions (mostly zeros)

### 4.4 Integration with prediction

Each squeeze channel is independently coded with the MA tree + prediction.
The detail channels are mostly zero → very peaked distribution → entropy
coder handles efficiently with context-adaptive models.

---

## 5. Color Transform (RCT)

**File**: `lib/jxl/modular/transform/rct.cc`

Reversible Color Transform — same principle as our YCoCg-R:
```
Multiple options tried by encoder, picks best:
  YCoCg:   Y = (R + 2G + B) / 4, Co = R - B, Cg = G - (R + B) / 2
  Various permutations and lifting steps
```

The encoder tries multiple RCT variants and picks the one that gives the
best compression for this specific image.

---

## 6. Patches (Self-Referential Copy)

**File**: `lib/jxl/enc_patch_dictionary.cc`

Patches allow copying previously-decoded regions to the current position.
For natural photos, this gives marginal benefit (the image doesn't have
exact self-similarity). But for screenshots, UI, text, and synthetic
images it can be very effective.

---

## 7. What We Need to Implement to Match JPEG-XL

### Priority 1: Self-Correcting Weighted Predictor
- 4 parameterized sub-predictors with error feedback
- Online weight update based on accumulated prediction errors
- Clamping when errors are consistently biased
- **Expected improvement: 15-25%** (this is the biggest gap)

### Priority 2: MA Decision Tree Context Selection
- Compute 13+ properties per pixel
- Build optimal binary decision tree via entropy-based splitting
- Each leaf selects: predictor, context, offset
- **Expected improvement: 10-15%** (replaces our fixed 75 contexts)

### Priority 3: Squeeze Transform
- Multi-scale integer wavelet decomposition
- Smooth tendency correction in smooth areas
- Code detail channels (mostly zeros) separately
- **Expected improvement: 5-10%** (decorrelation at multiple scales)

### Priority 4: rANS Entropy Coding
- Replace arithmetic coding with ANS for speed
- Similar compression efficiency, much faster decode
- **Expected improvement: 0-2%** in size (mainly speed gain)

### Combined expected improvement: 30-50% over current hash codec
Target: 1225 KB × 0.55 ≈ 670-860 KB (potentially matching JPEG-XL's 917 KB)

---

## 8. Key Insight Summary

JPEG-XL's advantage is NOT any single technique but the tight integration:
1. Weighted predictor adapts to local statistics in real-time
2. MA tree selects the right predictor AND context for each region
3. Squeeze decorrelates at multiple scales
4. ANS codes efficiently with hundreds of fine-grained contexts

Each component feeds into the next. The predictor's error signal becomes
a property for the MA tree. The tree selects the best predictor. The squeeze
transform makes detail channels easier to predict. It's a closed loop that
continuously self-optimizes as it scans across the image.
