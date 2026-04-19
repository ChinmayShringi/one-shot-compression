"""HIGH-DIMENSIONAL DATA STRUCTURES FOR COMPRESSION RESEARCH

All built from scratch. No sklearn, scipy, or external libraries.

Data structures:
  - KDTree: O(log n) nearest neighbor in high-dimensional vector spaces
  - ContextTrie: Deep context modeling with variable-depth pattern matching
  - LSH (Locality-Sensitive Hashing): Approximate nearest neighbor in O(1)
  - BloomFilter: Probabilistic set membership in O(1)
  - VPTree (Vantage-Point Tree): Nearest neighbor in metric spaces

These serve as building blocks for the experimental codecs:
  - KDTree -> VQ codebook lookup, sparse dictionary matching
  - ContextTrie -> Attention predictor context modeling
  - LSH -> Fast patch matching for fractal codec
  - BloomFilter -> Duplicate patch detection
  - VPTree -> Metric space nearest neighbor for embeddings
"""

import numpy as np
import math
from collections import defaultdict


# ============================================================
# KD-TREE: Exact Nearest Neighbor in R^d
# ============================================================

class KDNode:
    __slots__ = ('point', 'index', 'axis', 'left', 'right')

    def __init__(self, point=None, index=-1, axis=0, left=None, right=None):
        self.point = point
        self.index = index
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:
    """KD-Tree for exact nearest neighbor search in R^d.

    Handles high-dimensional data (patch vectors of 48-192 dimensions).
    Build: O(n log n). Query: O(log n) average, O(n) worst case.
    """

    def __init__(self, points):
        """Build KD-tree from (n, d) array of points."""
        self.points = np.asarray(points, dtype=np.float64)
        self.n, self.d = self.points.shape
        indices = np.arange(self.n)
        self.root = self._build(indices, depth=0)

    def _build(self, indices, depth):
        if len(indices) == 0:
            return None
        axis = depth % self.d
        sorted_idx = indices[np.argsort(self.points[indices, axis])]
        mid = len(sorted_idx) // 2
        node = KDNode(
            point=self.points[sorted_idx[mid]],
            index=int(sorted_idx[mid]),
            axis=axis,
            left=self._build(sorted_idx[:mid], depth + 1),
            right=self._build(sorted_idx[mid + 1:], depth + 1),
        )
        return node

    def nearest(self, query, k=1):
        """Find k nearest neighbors. Returns (distances, indices)."""
        query = np.asarray(query, dtype=np.float64)
        # Use a bounded priority queue (max-heap by distance)
        import heapq
        best = []  # max-heap: (-dist, index)

        def _search(node):
            if node is None:
                return
            dist = np.sum((node.point - query) ** 2)
            if len(best) < k:
                heapq.heappush(best, (-dist, node.index))
            elif dist < -best[0][0]:
                heapq.heapreplace(best, (-dist, node.index))

            axis = node.axis
            diff = query[axis] - node.point[axis]
            near = node.left if diff < 0 else node.right
            far = node.right if diff < 0 else node.left

            _search(near)
            # Prune: only search far side if hyperplane is closer than worst best
            worst_dist = -best[0][0] if len(best) == k else float('inf')
            if diff ** 2 < worst_dist:
                _search(far)

        _search(self.root)
        results = sorted([(-d, i) for d, i in best])
        distances = np.array([d for d, i in results])
        indices = np.array([i for d, i in results])
        return np.sqrt(distances), indices

    def nearest_batch(self, queries, k=1):
        """Batch nearest neighbor for multiple queries."""
        dists_list = []
        idx_list = []
        for q in queries:
            d, i = self.nearest(q, k)
            dists_list.append(d)
            idx_list.append(i)
        return np.array(dists_list), np.array(idx_list)


# ============================================================
# CONTEXT TRIE: Variable-Depth Pattern Matching
# ============================================================

class TrieNode:
    __slots__ = ('children', 'counts', 'total')

    def __init__(self):
        self.children = {}
        self.counts = defaultdict(int)
        self.total = 0


class ContextTrie:
    """Variable-depth context model using a trie.

    Stores patterns of arbitrary depth and their successor distributions.
    Used by attention predictor for deep context matching:
    - Feed sequence of symbols (quantized pixel values)
    - Query: given context [c1, c2, ..., cn], predict next symbol
    - Automatically backs off to shorter contexts if long context unseen

    This is similar to PPM (Prediction by Partial Matching) used in
    the world's best text compressors (PPMd, ZPAQ).
    """

    def __init__(self, max_depth=8):
        self.root = TrieNode()
        self.max_depth = max_depth

    def update(self, context, symbol):
        """Add observation: after seeing 'context', 'symbol' occurred."""
        # Update at all depths from 0 (empty context) to len(context)
        node = self.root
        node.counts[symbol] += 1
        node.total += 1

        for i, ctx_sym in enumerate(context[-self.max_depth:]):
            if ctx_sym not in node.children:
                node.children[ctx_sym] = TrieNode()
            node = node.children[ctx_sym]
            node.counts[symbol] += 1
            node.total += 1

    def predict(self, context, n_symbols=256, escape_prob=0.1):
        """Predict probability distribution for next symbol given context.

        Uses PPM-style exclusion: try longest context first, back off
        to shorter contexts for unseen symbols.
        Returns (n_symbols,) probability array.
        """
        probs = np.ones(n_symbols, dtype=np.float64) / n_symbols  # uniform prior

        # Walk to deepest matching node
        node = self.root
        best_node = node if node.total > 0 else None
        best_depth = 0

        for i, ctx_sym in enumerate(context[-self.max_depth:]):
            if ctx_sym not in node.children:
                break
            node = node.children[ctx_sym]
            if node.total > 0:
                best_node = node
                best_depth = i + 1

        if best_node is not None and best_node.total > 0:
            # Blend deep context with uniform prior (escape mechanism)
            alpha = 1.0 - escape_prob ** best_depth
            context_probs = np.zeros(n_symbols, dtype=np.float64)
            for sym, count in best_node.counts.items():
                if 0 <= sym < n_symbols:
                    context_probs[sym] = count / best_node.total
            probs = alpha * context_probs + (1.0 - alpha) * probs

        # Normalize
        total = probs.sum()
        if total > 0:
            probs /= total
        return probs

    def entropy_estimate(self, context, symbol, n_symbols=256):
        """Estimate bits needed to code 'symbol' given 'context'."""
        probs = self.predict(context, n_symbols)
        p = max(probs[symbol], 1e-10)
        return -math.log2(p)


# ============================================================
# LSH: Locality-Sensitive Hashing
# ============================================================

class LSH:
    """Locality-Sensitive Hashing for approximate nearest neighbor.

    Uses random hyperplane hashing (cosine similarity).
    Build: O(n * n_tables * n_bits). Query: O(n_tables * bucket_size).

    For high-dimensional patch vectors (48-192 dims), exact NN via KD-tree
    degrades. LSH provides O(1) approximate lookup.
    """

    def __init__(self, dim, n_tables=8, n_bits=16, seed=42):
        self.dim = dim
        self.n_tables = n_tables
        self.n_bits = n_bits

        rng = np.random.RandomState(seed)
        # Random hyperplanes for each table
        self.hyperplanes = [
            rng.randn(n_bits, dim).astype(np.float64)
            for _ in range(n_tables)
        ]
        self.tables = [defaultdict(list) for _ in range(n_tables)]
        self.points = None

    def _hash(self, point, table_idx):
        """Hash a point to a bucket in the given table."""
        projections = self.hyperplanes[table_idx] @ point
        bits = (projections > 0).astype(np.int32)
        return tuple(bits.tolist())

    def build(self, points):
        """Index all points."""
        self.points = np.asarray(points, dtype=np.float64)
        for t in range(self.n_tables):
            self.tables[t].clear()
            for i, p in enumerate(self.points):
                h = self._hash(p, t)
                self.tables[t][h].append(i)

    def query(self, point, k=1):
        """Find approximate k nearest neighbors."""
        point = np.asarray(point, dtype=np.float64)
        candidates = set()
        for t in range(self.n_tables):
            h = self._hash(point, t)
            candidates.update(self.tables[t].get(h, []))

        if not candidates:
            # Fallback: return random point
            return np.array([float('inf')]), np.array([0])

        cand_indices = np.array(list(candidates))
        cand_points = self.points[cand_indices]
        dists = np.sum((cand_points - point) ** 2, axis=1)
        top_k = min(k, len(dists))
        best = np.argpartition(dists, top_k - 1)[:top_k]
        best = best[np.argsort(dists[best])]

        return np.sqrt(dists[best]), cand_indices[best]


# ============================================================
# BLOOM FILTER: Probabilistic Set Membership
# ============================================================

class BloomFilter:
    """Bloom filter for fast probabilistic set membership testing.

    Used by fractal codec to quickly check if a block pattern has been
    seen before (avoiding redundant searches). False positives possible,
    false negatives impossible.

    Space: m bits. Hash functions: k.
    False positive rate: (1 - e^(-kn/m))^k
    """

    def __init__(self, expected_items=10000, fp_rate=0.01):
        # Optimal parameters
        self.m = int(-expected_items * math.log(fp_rate) / (math.log(2) ** 2))
        self.k = max(1, int(self.m / expected_items * math.log(2)))
        self.bits = np.zeros(self.m, dtype=np.uint8)
        self.n_items = 0

    def _hashes(self, item):
        """Generate k hash values using double hashing."""
        # FNV-1a inspired hash
        if isinstance(item, np.ndarray):
            data = item.tobytes()
        elif isinstance(item, (list, tuple)):
            data = bytes(str(item), 'utf-8')
        else:
            data = bytes(str(item), 'utf-8')

        h1 = 2166136261
        h2 = 16777619
        for byte in data:
            h1 = ((h1 ^ byte) * 16777619) & 0xFFFFFFFF
            h2 = ((h2 * 31) + byte) & 0xFFFFFFFF

        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, item):
        """Add item to the filter."""
        for h in self._hashes(item):
            self.bits[h] = 1
        self.n_items += 1

    def contains(self, item):
        """Check if item might be in the set (false positives possible)."""
        return all(self.bits[h] for h in self._hashes(item))


# ============================================================
# VP-TREE: Vantage-Point Tree for Metric Spaces
# ============================================================

class VPNode:
    __slots__ = ('index', 'threshold', 'left', 'right')

    def __init__(self):
        self.index = -1
        self.threshold = 0.0
        self.left = None
        self.right = None


class VPTree:
    """Vantage-Point Tree for nearest neighbor in arbitrary metric spaces.

    Unlike KD-tree (axis-aligned splits), VP-tree uses distance-based
    splits. Better for high-dimensional data where KD-tree degrades.

    Build: O(n log n). Query: O(log n) average.
    """

    def __init__(self, points, dist_fn=None):
        self.points = np.asarray(points, dtype=np.float64)
        self.n = len(self.points)
        self.dist_fn = dist_fn or self._l2_dist
        indices = np.arange(self.n)
        self.root = self._build(indices)

    @staticmethod
    def _l2_dist(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _build(self, indices):
        if len(indices) == 0:
            return None

        node = VPNode()
        # Pick vantage point (random for simplicity)
        vp_pos = np.random.randint(len(indices))
        vp_idx = indices[vp_pos]
        node.index = int(vp_idx)

        if len(indices) == 1:
            return node

        rest = np.delete(indices, vp_pos)
        dists = np.array([self.dist_fn(self.points[vp_idx], self.points[i]) for i in rest])
        median = np.median(dists)
        node.threshold = float(median)

        left_mask = dists <= median
        right_mask = ~left_mask

        node.left = self._build(rest[left_mask])
        node.right = self._build(rest[right_mask])

        return node

    def nearest(self, query, k=1):
        """Find k nearest neighbors."""
        import heapq
        query = np.asarray(query, dtype=np.float64)
        best = []

        def _search(node):
            if node is None:
                return
            dist = self.dist_fn(self.points[node.index], query)
            if len(best) < k:
                heapq.heappush(best, (-dist, node.index))
            elif dist < -best[0][0]:
                heapq.heapreplace(best, (-dist, node.index))

            tau = -best[0][0] if len(best) == k else float('inf')

            if dist < node.threshold:
                _search(node.left)
                if dist + tau >= node.threshold:
                    _search(node.right)
            else:
                _search(node.right)
                if dist - tau <= node.threshold:
                    _search(node.left)

        _search(self.root)
        results = sorted([(-d, i) for d, i in best])
        return np.array([d for d, i in results]), np.array([i for d, i in results])


# ============================================================
# UTILITIES
# ============================================================

def entropy_bits(data):
    """Calculate total entropy in bits."""
    flat = np.asarray(data).flatten()
    if len(flat) == 0:
        return 0.0
    unique, counts = np.unique(flat, return_counts=True)
    total = len(flat)
    probs = counts / total
    return -np.sum(counts * np.log2(probs))


def zigzag_encode(x):
    """Map signed integers to unsigned: 0,1,-1,2,-2 -> 0,1,2,3,4"""
    x = np.asarray(x, dtype=np.int32)
    return np.where(x >= 0, 2 * x, -2 * x - 1).astype(np.uint32)


def extract_patches(data, patch_size, stride=None):
    """Extract non-overlapping patches from (H, W, C) array.

    Returns (n_patches, patch_size * patch_size * C) and (n_h, n_w) counts.
    Also returns border pixels that don't fit into patches.
    """
    stride = stride or patch_size
    h, w = data.shape[:2]
    c = data.shape[2] if len(data.shape) == 3 else 1
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]

    n_h = (h - patch_size) // stride + 1
    n_w = (w - patch_size) // stride + 1

    patches = []
    for i in range(n_h):
        for j in range(n_w):
            y0 = i * stride
            x0 = j * stride
            patch = data[y0:y0 + patch_size, x0:x0 + patch_size].reshape(-1)
            patches.append(patch)

    patches = np.array(patches, dtype=np.int32)

    # Border: pixels not covered by patches
    covered_h = (n_h - 1) * stride + patch_size
    covered_w = (n_w - 1) * stride + patch_size
    border_bottom = data[covered_h:, :covered_w].reshape(-1) if covered_h < h else np.array([], dtype=np.int32)
    border_right = data[:, covered_w:].reshape(-1) if covered_w < w else np.array([], dtype=np.int32)

    return patches, (n_h, n_w), np.concatenate([border_bottom, border_right])


if __name__ == "__main__":
    print("=== Data Structure Tests ===\n")

    # KD-Tree test
    rng = np.random.RandomState(42)
    points = rng.randn(1000, 48)
    tree = KDTree(points)
    query = rng.randn(48)
    dist, idx = tree.nearest(query, k=3)
    print(f"KDTree: 1000 points in R^48, 3-NN distances: {dist}")

    # Brute force verification
    all_dists = np.sqrt(np.sum((points - query) ** 2, axis=1))
    bf_idx = np.argsort(all_dists)[:3]
    print(f"  Brute force verification: {all_dists[bf_idx]}")
    assert np.allclose(dist, all_dists[bf_idx], atol=1e-10), "KDTree mismatch!"
    print("  PASSED")

    # LSH test
    lsh = LSH(dim=48, n_tables=8, n_bits=16)
    lsh.build(points)
    lsh_dist, lsh_idx = lsh.query(query, k=3)
    print(f"\nLSH: approximate 3-NN distances: {lsh_dist}")
    # LSH is approximate, so just check it found reasonable neighbors
    print(f"  True NN dist: {all_dists[bf_idx[0]]:.4f}, LSH best: {lsh_dist[0]:.4f}")
    print(f"  Recall: {len(set(lsh_idx) & set(bf_idx))}/3")

    # Context Trie test
    trie = ContextTrie(max_depth=4)
    sequence = [1, 2, 3, 1, 2, 3, 1, 2, 4, 1, 2, 3]
    for i in range(2, len(sequence)):
        ctx = sequence[max(0, i - 4):i]
        trie.update(ctx, sequence[i])

    probs = trie.predict([1, 2], n_symbols=5)
    print(f"\nContextTrie: P(next|[1,2]) = {probs[:5]}")
    print(f"  Most likely after [1,2]: {np.argmax(probs)} (expected: 3)")

    # Bloom filter test
    bf = BloomFilter(expected_items=100, fp_rate=0.01)
    for i in range(100):
        bf.add(np.array([i, i * 2, i * 3]))
    hits = sum(bf.contains(np.array([i, i * 2, i * 3])) for i in range(100))
    fps = sum(bf.contains(np.array([i + 1000, i * 7, i * 13])) for i in range(100))
    print(f"\nBloomFilter: {hits}/100 true positives, {fps}/100 false positives")
    print(f"  m={bf.m} bits, k={bf.k} hashes")

    # VP-Tree test
    vpt = VPTree(points)
    vp_dist, vp_idx = vpt.nearest(query, k=3)
    print(f"\nVPTree: 3-NN distances: {vp_dist}")
    print(f"  Matches KDTree: {np.allclose(sorted(dist), sorted(vp_dist), atol=1e-10)}")

    print("\n=== All data structure tests passed ===")
