"""PRISM Codec - Full encode/decode pipeline.

Ties together color transform, prediction, and entropy coding
into a working lossless image compressor.

Pipeline:
  Encode: RGB -> YCoCg-R -> block prediction -> residuals -> entropy code -> .prism file
  Decode: .prism file -> entropy decode -> block reconstruct -> YCoCg-R -> RGB

Container format:
  b'PRSM' (4B magic)
  uint8   version (1)
  uint32  width, height
  uint8   block_size
  uint8   n_channels (3)
  -- per channel --
  uint8   n_modes
  uint16  mode_grid_h, mode_grid_w
  bytes   mode_grid (raw, 1 byte per block)
  uint32  residual_data_len
  bytes   residual_data (entropy coded)
  -- end per channel --
  b'PRSE' (4B footer)
"""

import struct
import numpy as np
from PIL import Image
import time
import math
from collections import Counter

from prism.transforms import (
    rgb_to_ycocg_r, ycocg_r_to_rgb,
    zigzag_encode, zigzag_decode
)
from prism.predict_fast import (
    encode_channel, decode_channel
)
from prism.arith import (
    encode_symbols_adaptive, decode_symbols_adaptive
)


# ============================================================
# Entropy measurement (for theoretical size)
# ============================================================

def entropy_bps(data):
    """Shannon entropy in bits per symbol."""
    flat = data.flatten()
    if len(flat) == 0:
        return 0.0
    counts = Counter(flat.tolist())
    total = len(flat)
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h


# ============================================================
# Bucket-based entropy coding
# ============================================================

def value_to_bucket(val):
    """Map unsigned value to (bucket_id, extra_bits, extra_value).

    Bucket 0: val=0 (0 extra bits)
    Bucket 1: val=1 (0 extra bits)
    Bucket k (k>=2): val in [2^(k-1), 2^k - 1] (k-1 extra bits)
    """
    if val == 0:
        return 0, 0, 0
    if val == 1:
        return 1, 0, 0
    # Find bucket: k = floor(log2(val)) + 1
    k = val.bit_length()  # number of bits needed
    extra_bits = k - 1
    extra_value = val - (1 << (k - 1))  # value within bucket
    return k, extra_bits, extra_value


def bucket_to_value(bucket_id, extra_value):
    """Reconstruct value from bucket_id and extra_value."""
    if bucket_id == 0:
        return 0
    if bucket_id == 1:
        return 1
    return (1 << (bucket_id - 1)) + extra_value


# Vectorized versions for encoding
def values_to_buckets(vals):
    """Vectorized: map array of unsigned values to bucket arrays."""
    vals = vals.astype(np.int64)
    buckets = np.zeros_like(vals)
    extra_bits = np.zeros_like(vals)
    extra_values = np.zeros_like(vals)

    # val == 0 -> bucket 0
    # val == 1 -> bucket 1
    mask_0 = vals == 0
    mask_1 = vals == 1
    mask_rest = ~mask_0 & ~mask_1

    buckets[mask_1] = 1

    if mask_rest.any():
        rest_vals = vals[mask_rest]
        # bit_length via log2
        k = np.floor(np.log2(rest_vals.astype(np.float64))).astype(np.int64) + 1
        buckets[mask_rest] = k
        extra_bits[mask_rest] = k - 1
        extra_values[mask_rest] = rest_vals - (1 << (k - 1))

    return buckets, extra_bits, extra_values


# ============================================================
# Entropy coding: Huffman (bucket symbols) + raw bits (extra)
# ============================================================

import heapq


def _build_huffman_tree(freq):
    """Build Huffman tree from frequency list. Returns {symbol: bitstring}."""
    # Filter to non-zero frequencies
    nodes = [(f, i) for i, f in enumerate(freq) if f > 0]
    if len(nodes) == 0:
        return {}
    if len(nodes) == 1:
        return {nodes[0][1]: '0'}

    heapq.heapify(nodes)
    node_id = len(freq)
    children = {}

    while len(nodes) > 1:
        f1, n1 = heapq.heappop(nodes)
        f2, n2 = heapq.heappop(nodes)
        children[node_id] = (n1, n2)
        heapq.heappush(nodes, (f1 + f2, node_id))
        node_id += 1

    # Traverse tree to get codes
    codes = {}
    def traverse(node, code):
        if node in children:
            left, right = children[node]
            traverse(left, code + '0')
            traverse(right, code + '1')
        else:
            codes[node] = code if code else '0'

    traverse(nodes[0][1], '')
    return codes


class BitWriter:
    """Write individual bits to a byte buffer."""
    __slots__ = ('buffer', 'byte', 'bit_pos')

    def __init__(self):
        self.buffer = bytearray()
        self.byte = 0
        self.bit_pos = 0

    def write_bit(self, bit):
        self.byte = (self.byte << 1) | (bit & 1)
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.buffer.append(self.byte)
            self.byte = 0
            self.bit_pos = 0

    def write_bits(self, value, n_bits):
        for i in range(n_bits - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    def write_bitstring(self, s):
        for ch in s:
            self.write_bit(int(ch))

    def flush(self):
        if self.bit_pos > 0:
            self.byte <<= (8 - self.bit_pos)
            self.buffer.append(self.byte)
            self.byte = 0
            self.bit_pos = 0
        return bytes(self.buffer)


class BitReader:
    """Read individual bits from a byte buffer."""
    __slots__ = ('data', 'byte_pos', 'bit_pos')

    def __init__(self, data):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0

    def read_bit(self):
        if self.byte_pos >= len(self.data):
            return 0
        bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
        return bit

    def read_bits(self, n):
        value = 0
        for _ in range(n):
            value = (value << 1) | self.read_bit()
        return value


def encode_residuals(residuals_flat, max_bucket=20):
    """Encode residuals using Huffman-coded buckets + raw extra bits.

    1. Zigzag encode (signed -> unsigned)
    2. Split into bucket_id + extra_bits
    3. Huffman-code the bucket sequence
    4. Append extra bits directly
    """
    zz = zigzag_encode(residuals_flat.astype(np.int32)).flatten()

    # Split into buckets
    buckets, extra_bits_arr, extra_values_arr = values_to_buckets(zz)
    buckets = np.minimum(buckets.astype(np.int32), max_bucket)

    # Count bucket frequencies
    freq = [0] * (max_bucket + 1)
    for b in buckets.flat:
        freq[int(b)] += 1

    # Build Huffman codes
    codes = _build_huffman_tree(freq)

    # Write bitstream
    writer = BitWriter()

    for i in range(len(zz)):
        b = int(buckets[i])
        # Write Huffman code for bucket
        if b in codes:
            writer.write_bitstring(codes[b])
        else:
            writer.write_bitstring(codes.get(0, '0'))
        # Write extra bits for the value within the bucket
        n_extra = int(extra_bits_arr[i])
        if n_extra > 0:
            writer.write_bits(int(extra_values_arr[i]), n_extra)

    bitstream = writer.flush()

    # Pack: code_table + bitstream
    result = bytearray()

    # Store code table: for each symbol 0..max_bucket, store code length (uint8)
    # and the code bits. If length==0, symbol not present.
    code_lengths = [0] * (max_bucket + 1)
    for sym, code in codes.items():
        if sym <= max_bucket:
            code_lengths[sym] = len(code)

    # Store max code length and canonical code info
    for cl in code_lengths:
        result.append(cl & 0xFF)

    # Store actual code bits (for reconstruction)
    for sym in range(max_bucket + 1):
        if sym in codes:
            code = codes[sym]
            n_bytes = (len(code) + 7) // 8
            val = int(code, 2) if code else 0
            result.extend(val.to_bytes(max(1, n_bytes), 'big'))

    # Bitstream length in bytes + data
    result.extend(struct.pack('<I', len(bitstream)))
    result.extend(bitstream)

    return bytes(result), len(zz)


def decode_residuals(data, count, max_bucket=20):
    """Decode residuals from Huffman-coded bucket stream."""
    pos = 0

    # Read code lengths
    code_lengths = []
    for _ in range(max_bucket + 1):
        code_lengths.append(data[pos])
        pos += 1

    # Read actual code bits and build decode table
    decode_table = {}  # bitstring -> symbol
    for sym in range(max_bucket + 1):
        cl = code_lengths[sym]
        if cl > 0:
            n_bytes = (cl + 7) // 8
            val = int.from_bytes(data[pos:pos + n_bytes], 'big')
            pos += n_bytes
            # Convert to bitstring of exact length
            code = bin(val)[2:].zfill(cl)[-cl:]
            decode_table[code] = sym

    # Read bitstream
    bs_len = struct.unpack_from('<I', data, pos)[0]
    pos += 4
    bitstream = data[pos:pos + bs_len]

    reader = BitReader(bitstream)

    # Decode symbols
    values = []
    for _ in range(count):
        # Read Huffman code bit by bit
        current = ''
        symbol = None
        for _ in range(32):  # Max code length safety
            current += str(reader.read_bit())
            if current in decode_table:
                symbol = decode_table[current]
                break
        if symbol is None:
            symbol = 0  # Fallback

        # Read extra bits
        if symbol == 0:
            values.append(0)
        elif symbol == 1:
            values.append(1)
        else:
            n_extra = symbol - 1
            extra_val = reader.read_bits(n_extra)
            values.append((1 << (symbol - 1)) + extra_val)

    zz = np.array(values, dtype=np.int32)
    return zigzag_decode(zz)


# ============================================================
# Adaptive Arithmetic Coding (near-optimal, replaces Huffman)
# ============================================================

def encode_residuals_arith(residuals_flat, max_bucket=16):
    """Encode residuals using adaptive arithmetic coding on buckets.

    Near-optimal: within 0.5% of Shannon entropy.
    """
    zz = zigzag_encode(residuals_flat.astype(np.int32)).flatten()
    buckets, extra_bits_arr, extra_values_arr = values_to_buckets(zz)
    buckets = np.minimum(buckets.astype(np.int64), max_bucket)

    # Arithmetic-code the bucket sequence (adaptive)
    bucket_list = buckets.tolist()
    bucket_data = encode_symbols_adaptive(bucket_list, max_bucket + 1)

    # Pack extra bits
    writer = BitWriter()
    for i in range(len(zz)):
        n_extra = int(extra_bits_arr[i])
        if n_extra > 0:
            writer.write_bits(int(extra_values_arr[i]), n_extra)
    extra_data = writer.flush()

    # Pack: bucket_data_len + bucket_data + extra_data
    result = bytearray()
    result.extend(struct.pack('<I', len(bucket_data)))
    result.extend(bucket_data)
    result.extend(extra_data)

    return bytes(result), len(zz)


def decode_residuals_arith(data, count, max_bucket=16):
    """Decode residuals from adaptive arithmetic-coded buckets."""
    pos = 0

    bucket_data_len = struct.unpack_from('<I', data, pos)[0]
    pos += 4
    bucket_data = data[pos:pos + bucket_data_len]
    pos += bucket_data_len

    # Decode buckets
    buckets = decode_symbols_adaptive(bucket_data, count, max_bucket + 1)

    # Read extra bits
    extra_data = data[pos:]
    reader = BitReader(extra_data)

    values = []
    for bucket_id in buckets:
        if bucket_id == 0:
            values.append(0)
        elif bucket_id == 1:
            values.append(1)
        else:
            n_extra = bucket_id - 1
            extra_val = reader.read_bits(n_extra)
            values.append((1 << (bucket_id - 1)) + extra_val)

    zz = np.array(values, dtype=np.int32)
    return zigzag_decode(zz)


# ============================================================
# Full encode/decode pipeline
# ============================================================

def encode_image(image_path, output_path, block_size=8):
    """Encode image to .prism file."""
    print(f"Encoding: {image_path}")
    t_start = time.time()

    # Load
    img = Image.open(image_path)
    if img.mode == 'RGBA':
        rgb = np.array(img)[:, :, :3]  # Drop alpha
    else:
        rgb = np.array(img.convert('RGB'))
    h, w, _ = rgb.shape
    raw_size = h * w * 3
    print(f"  Image: {w}x{h}, raw: {raw_size:,} bytes ({raw_size / 1024:.0f} KB)")

    # Color transform
    t0 = time.time()
    ycocg = rgb_to_ycocg_r(rgb)
    print(f"  Color transform: {time.time() - t0:.2f}s")

    # Predict + encode each channel
    all_channel_data = []
    decoded_chs = []
    total_compressed = 0
    total_theoretical = 0

    ch_names = ['Y', 'Co', 'Cg']

    for ch_idx in range(3):
        ch = ycocg[:, :, ch_idx].astype(np.int32)
        ch_name = ch_names[ch_idx]

        t0 = time.time()
        ref = decoded_chs if decoded_chs else None
        modes, residuals, mode_names = encode_channel(ch, ref, block_size)
        t_pred = time.time() - t0

        # Theoretical size (entropy)
        zz = zigzag_encode(residuals)
        bps = entropy_bps(zz)
        theoretical_kb = bps * h * w / 8 / 1024
        total_theoretical += theoretical_kb

        # Actual entropy coding
        t0 = time.time()
        res_data, res_count = encode_residuals_arith(residuals, max_bucket=16)
        t_code = time.time() - t0

        # Mode grid (simple: store raw for now, tiny overhead)
        mode_data = modes.tobytes()

        actual_kb = (len(res_data) + len(mode_data)) / 1024
        total_compressed += len(res_data) + len(mode_data)

        all_channel_data.append({
            'modes': modes,
            'mode_names': mode_names,
            'residual_data': res_data,
            'residual_count': res_count,
            'mode_data': mode_data,
        })

        decoded_chs.append(ch)

        print(f"  {ch_name}: theoretical={theoretical_kb:.0f} KB, "
              f"actual={actual_kb:.0f} KB, "
              f"pred={t_pred:.1f}s, code={t_code:.1f}s")

    # Write .prism file
    t0 = time.time()
    with open(output_path, 'wb') as f:
        # Header
        f.write(b'PRSM')
        f.write(struct.pack('<BIIBB', 1, w, h, block_size, 3))

        for ch_data in all_channel_data:
            modes = ch_data['modes']
            mode_names = ch_data['mode_names']
            res_data = ch_data['residual_data']
            mode_data = ch_data['mode_data']

            # Number of modes
            f.write(struct.pack('<B', len(mode_names)))
            # Mode names (length-prefixed strings)
            for name in mode_names:
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('<B', len(name_bytes)))
                f.write(name_bytes)
            # Mode grid dimensions
            mh, mw = modes.shape
            f.write(struct.pack('<HH', mh, mw))
            # Mode grid data
            f.write(struct.pack('<I', len(mode_data)))
            f.write(mode_data)
            # Residual count and data
            f.write(struct.pack('<I', ch_data['residual_count']))
            f.write(struct.pack('<I', len(res_data)))
            f.write(res_data)

        f.write(b'PRSE')

    file_size = len(open(output_path, 'rb').read())
    t_total = time.time() - t_start

    print(f"\n  === Results ===")
    print(f"  Raw:         {raw_size:,} bytes ({raw_size / 1024:.0f} KB)")
    print(f"  Theoretical: {total_theoretical:.0f} KB (entropy limit)")
    print(f"  Actual:      {file_size:,} bytes ({file_size / 1024:.0f} KB)")
    print(f"  Ratio:       {raw_size / file_size:.2f}x")
    print(f"  Total time:  {t_total:.1f}s")

    return file_size


def decode_image(input_path, output_path=None):
    """Decode .prism file to image."""
    print(f"Decoding: {input_path}")
    t_start = time.time()

    with open(input_path, 'rb') as f:
        # Header
        magic = f.read(4)
        assert magic == b'PRSM', f"Invalid magic: {magic}"
        version, w, h, block_size, n_channels = struct.unpack('<BIIBB', f.read(11))
        assert version == 1

        channels = []
        decoded_chs = []
        ch_names = ['Y', 'Co', 'Cg']

        for ch_idx in range(n_channels):
            ch_name = ch_names[ch_idx]

            # Read mode names
            n_modes = struct.unpack('<B', f.read(1))[0]
            mode_names = []
            for _ in range(n_modes):
                name_len = struct.unpack('<B', f.read(1))[0]
                name = f.read(name_len).decode('utf-8')
                mode_names.append(name)

            # Read mode grid
            mh, mw = struct.unpack('<HH', f.read(4))
            mode_data_len = struct.unpack('<I', f.read(4))[0]
            mode_data = f.read(mode_data_len)
            modes = np.frombuffer(mode_data, dtype=np.uint8).reshape(mh, mw)

            # Read residual data
            res_count = struct.unpack('<I', f.read(4))[0]
            res_data_len = struct.unpack('<I', f.read(4))[0]
            res_data = f.read(res_data_len)

            # Decode residuals
            t0 = time.time()
            residuals_flat = decode_residuals_arith(res_data, res_count, max_bucket=16)
            residuals = residuals_flat.reshape(h, w)
            t_decode = time.time() - t0

            # Reconstruct channel
            t0 = time.time()
            ref = decoded_chs if decoded_chs else None
            recon = decode_channel(
                modes, residuals, mode_names,
                ref_channels=ref,
                block_size=block_size,
                ch_shape=(h, w)
            )
            t_recon = time.time() - t0

            channels.append(recon)
            decoded_chs.append(recon)
            print(f"  {ch_name}: decode={t_decode:.1f}s, recon={t_recon:.1f}s")

        # Verify footer
        footer = f.read(4)
        assert footer == b'PRSE', f"Invalid footer: {footer}"

    # Reconstruct image
    ycocg = np.stack(channels, axis=-1)
    rgb = ycocg_r_to_rgb(ycocg)

    if output_path:
        Image.fromarray(rgb).save(output_path)
        print(f"  Saved to: {output_path}")

    print(f"  Total time: {time.time() - t_start:.1f}s")
    return rgb


def verify_lossless(image_path, compressed_path):
    """Verify that encode -> decode produces identical image."""
    original = Image.open(image_path)
    if original.mode == 'RGBA':
        original_rgb = np.array(original)[:, :, :3]
    else:
        original_rgb = np.array(original.convert('RGB'))

    decoded_rgb = decode_image(compressed_path)

    if np.array_equal(original_rgb, decoded_rgb):
        print("\n  LOSSLESS VERIFICATION: PASS (bit-exact match)")
        return True
    else:
        diff = np.abs(original_rgb.astype(int) - decoded_rgb.astype(int))
        print(f"\n  LOSSLESS VERIFICATION: FAIL")
        print(f"  Max diff: {diff.max()}")
        print(f"  Pixels different: {(diff > 0).any(axis=-1).sum()}")
        return False
