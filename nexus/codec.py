"""NEXUS codec - main encoder/decoder pipeline.

Compresses video/image/audio using from-scratch algorithms:
- Adaptive spatial prediction (novel weighted multi-mode)
- Temporal frame differencing
- Huffman entropy coding (built from scratch)
- Custom NEXUS container format (.nxv)

File format:
    Magic: "NXV1" (4 bytes)
    Header: width(4) + height(4) + n_frames(4) + fps_num(2) + fps_den(2)
            + channels(1) + has_audio(1) + audio_rate(4) + audio_channels(1)
            + reserved(5) = 32 bytes
    Frame packets: [type(1) + size(4) + data(size)]
        type 0x01 = intra frame (spatial prediction only)
        type 0x02 = inter frame (temporal prediction)
        type 0x03 = audio packet
    Footer: "NXVE" (4 bytes)
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np

from .bitstream import BitWriter, BitReader
from .entropy import HuffmanCodec, compute_frequencies
from .predictor import predict_frame_spatial, predict_frame_temporal


# ---------------------------------------------------------------------------
# Container format constants
# ---------------------------------------------------------------------------
MAGIC = b"NXV1"
FOOTER = b"NXVE"
PKT_INTRA = 0x01
PKT_INTER = 0x02
PKT_AUDIO = 0x03

# GOP (Group of Pictures) size -- every N frames, force an intra frame
GOP_SIZE = 30


def _encode_residuals_huffman(residuals_flat: np.ndarray) -> bytes:
    """Encode residual values using from-scratch Huffman coding.

    Residuals are int16 in [-255, 255]. We map to unsigned [0, 511]
    for Huffman coding, then the table handles the rest.
    """
    # Map signed residuals to unsigned: add 255 -> range [0, 510]
    mapped = (residuals_flat.astype(np.int16) + 255).clip(0, 510).astype(np.uint16)
    symbols = mapped.tolist()

    freqs = compute_frequencies(symbols)
    codec = HuffmanCodec(freqs)

    writer = BitWriter()
    # Write the Huffman table
    codec.serialize_table(writer)
    # Write symbol count
    writer.write_uint32(len(symbols))
    # Encode all symbols
    for s in symbols:
        codec.encode_symbol(writer, s)
    return writer.flush()


def _decode_residuals_huffman(data: bytes, expected_size: int) -> np.ndarray:
    """Decode Huffman-coded residuals back to int16 array."""
    reader = BitReader(data)
    codec = HuffmanCodec.deserialize_table(reader)
    n_symbols = reader.read_uint32()
    symbols = []
    for _ in range(n_symbols):
        symbols.append(codec.decode_symbol(reader))
    # Map back: subtract 255 to get signed residuals
    residuals = np.array(symbols, dtype=np.int16) - 255
    return residuals


def _causal_predict(frame: np.ndarray) -> np.ndarray:
    """Causal adaptive spatial prediction (used by BOTH encoder and decoder).

    Processes in raster order. Each pixel's prediction uses ONLY
    already-processed neighbors. The adaptive mode selection tracks
    per-mode error from these same neighbors, so encoder and decoder
    converge to identical mode choices without transmitting mode info.

    This is NEXUS's key innovation: zero-cost adaptive prediction.
    """
    h, w = frame.shape
    residuals = np.zeros((h, w), dtype=np.int16)
    # Track cumulative error per mode for adaptation
    mode_errors = np.ones(6, dtype=np.float64)  # 6 prediction modes
    decay = 0.998

    for y in range(h):
        for x in range(w):
            left = int(frame[y, x - 1]) if x > 0 else 128
            above = int(frame[y - 1, x]) if y > 0 else 128
            diag = int(frame[y - 1, x - 1]) if (y > 0 and x > 0) else 0
            actual = int(frame[y, x])

            # 6 prediction modes
            preds = [
                left,                                              # 0: left
                above,                                             # 1: above
                diag if (y > 0 and x > 0) else left,              # 2: diagonal
                0,                                                 # 3: paeth (computed below)
                0,                                                 # 4: gradient
                (left + above + 1) >> 1,                           # 5: average
            ]
            # Paeth
            est = left + above - diag
            dl, da, dd = abs(est - left), abs(est - above), abs(est - diag)
            if dl <= da and dl <= dd:
                preds[3] = left
            elif da <= dd:
                preds[3] = above
            else:
                preds[3] = diag
            # Gradient
            if y >= 1 and x >= 1:
                gh = left - diag
                gv = above - diag
                if abs(gh) > abs(gv):
                    preds[4] = max(0, min(255, above + gh))
                else:
                    preds[4] = max(0, min(255, left + gv))
            else:
                preds[4] = preds[3]

            # Pick best mode (lowest cumulative error)
            best = int(np.argmin(mode_errors))
            pred = preds[best]
            residuals[y, x] = actual - pred

            # Update all mode errors (both encoder and decoder do this)
            for m in range(6):
                err = (actual - preds[m]) ** 2
                mode_errors[m] = mode_errors[m] * decay + err

    return residuals


def _causal_reconstruct(residuals: np.ndarray, h: int, w: int) -> np.ndarray:
    """Reconstruct frame from residuals using same causal prediction.

    MUST match _causal_predict exactly (same modes, same adaptation).
    """
    frame = np.zeros((h, w), dtype=np.uint8)
    mode_errors = np.ones(6, dtype=np.float64)
    decay = 0.998

    for y in range(h):
        for x in range(w):
            left = int(frame[y, x - 1]) if x > 0 else 128
            above = int(frame[y - 1, x]) if y > 0 else 128
            diag = int(frame[y - 1, x - 1]) if (y > 0 and x > 0) else 0

            preds = [
                left, above,
                diag if (y > 0 and x > 0) else left,
                0, 0,
                (left + above + 1) >> 1,
            ]
            est = left + above - diag
            dl, da, dd = abs(est - left), abs(est - above), abs(est - diag)
            if dl <= da and dl <= dd:
                preds[3] = left
            elif da <= dd:
                preds[3] = above
            else:
                preds[3] = diag
            if y >= 1 and x >= 1:
                gh = left - diag
                gv = above - diag
                if abs(gh) > abs(gv):
                    preds[4] = max(0, min(255, above + gh))
                else:
                    preds[4] = max(0, min(255, left + gv))
            else:
                preds[4] = preds[3]

            best = int(np.argmin(mode_errors))
            pred = preds[best]
            actual = max(0, min(255, pred + int(residuals[y, x])))
            frame[y, x] = actual

            for m in range(6):
                err = (actual - preds[m]) ** 2
                mode_errors[m] = mode_errors[m] * decay + err

    return frame


def encode_frame_intra(frame_y: np.ndarray) -> bytes:
    """Encode a single frame using causal adaptive prediction + Huffman."""
    residuals = _causal_predict(frame_y)
    return _encode_residuals_huffman(residuals.ravel())


def encode_frame_inter(frame_y: np.ndarray, ref_y: np.ndarray) -> bytes:
    """Encode a frame using temporal prediction + Huffman.

    Steps:
    1. Compute temporal residuals (current - reference)
    2. Huffman-encode the residuals
    """
    residuals = predict_frame_temporal(frame_y, ref_y)
    return _encode_residuals_huffman(residuals.ravel())


def decode_frame_intra(data: bytes, width: int, height: int) -> np.ndarray:
    """Decode an intra-coded frame using same causal prediction as encoder."""
    residuals = _decode_residuals_huffman(data, width * height)
    residuals_2d = residuals.reshape(height, width)
    return _causal_reconstruct(residuals_2d, height, width)


def decode_frame_inter(data: bytes, ref: np.ndarray,
                       width: int, height: int) -> np.ndarray:
    """Decode an inter-coded frame."""
    residuals = _decode_residuals_huffman(data, width * height)
    residuals_2d = residuals.reshape(height, width)
    return np.clip(
        ref.astype(np.int16) + residuals_2d,
        0, 255,
    ).astype(np.uint8)


def encode_audio_pcm(audio_samples: np.ndarray) -> bytes:
    """Encode audio using delta prediction + Huffman.

    LPC-style: predict each sample from previous sample,
    encode the difference.
    """
    if len(audio_samples) == 0:
        return b""
    samples = audio_samples.astype(np.int16)
    # Delta encoding: residual = sample[i] - sample[i-1]
    deltas = np.zeros_like(samples)
    deltas[0] = samples[0]
    deltas[1:] = samples[1:] - samples[:-1]
    # Map to unsigned for Huffman (shift by 32768 for int16 range)
    mapped = (deltas.astype(np.int32) + 32768).clip(0, 65535).astype(np.uint16)
    symbols = mapped.tolist()
    freqs = compute_frequencies(symbols)
    codec = HuffmanCodec(freqs)
    writer = BitWriter()
    codec.serialize_table(writer)
    writer.write_uint32(len(symbols))
    for s in symbols:
        codec.encode_symbol(writer, s)
    return writer.flush()


def decode_audio_pcm(data: bytes) -> np.ndarray:
    """Decode delta-coded audio."""
    reader = BitReader(data)
    codec = HuffmanCodec.deserialize_table(reader)
    n_symbols = reader.read_uint32()
    symbols = []
    for _ in range(n_symbols):
        symbols.append(codec.decode_symbol(reader))
    # Unmap and undo delta
    deltas = np.array(symbols, dtype=np.int32) - 32768
    samples = np.cumsum(deltas).astype(np.int16)
    return samples


# ---------------------------------------------------------------------------
# Full video encode/decode
# ---------------------------------------------------------------------------

def encode_video(frames: list[np.ndarray], width: int, height: int,
                 fps: tuple[int, int] = (25, 1),
                 audio_samples: np.ndarray | None = None,
                 audio_rate: int = 0,
                 audio_channels: int = 0,
                 progress_fn=None) -> bytes:
    """Encode a sequence of grayscale frames to NEXUS format.

    Parameters:
        frames: list of uint8 arrays, each (height, width)
        width, height: frame dimensions
        fps: (numerator, denominator)
        audio_samples: optional int16 audio array
        audio_rate: audio sample rate
        audio_channels: number of audio channels
        progress_fn: optional callback(frame_idx, total_frames)

    Returns: complete NEXUS bitstream as bytes
    """
    n_frames = len(frames)
    has_audio = audio_samples is not None and len(audio_samples) > 0

    # Build output buffer
    output = bytearray()

    # --- Header ---
    output += MAGIC
    output += struct.pack("<IIIHHBBIBBB",
                          width, height, n_frames,
                          fps[0], fps[1],
                          1,  # channels (Y only for now)
                          1 if has_audio else 0,
                          audio_rate,
                          audio_channels,
                          0, 0)  # reserved

    # --- Audio packet (if present) ---
    if has_audio:
        audio_data = encode_audio_pcm(audio_samples)
        output += struct.pack("<BI", PKT_AUDIO, len(audio_data))
        output += audio_data

    # --- Frame packets ---
    ref_frame = None
    for i, frame in enumerate(frames):
        is_intra = (i % GOP_SIZE == 0) or ref_frame is None

        if is_intra:
            frame_data = encode_frame_intra(frame)
            pkt_type = PKT_INTRA
        else:
            frame_data = encode_frame_inter(frame, ref_frame)
            pkt_type = PKT_INTER

        output += struct.pack("<BI", pkt_type, len(frame_data))
        output += frame_data
        ref_frame = frame

        if progress_fn:
            progress_fn(i + 1, n_frames)

    # --- Footer ---
    output += FOOTER
    return bytes(output)


def decode_video(data: bytes,
                 progress_fn=None) -> tuple[list[np.ndarray], dict]:
    """Decode NEXUS bitstream back to frames.

    Returns: (frames, metadata_dict)
    """
    offset = 0

    # --- Magic ---
    assert data[offset:offset + 4] == MAGIC, "Not a NEXUS file"
    offset += 4

    # --- Header ---
    (width, height, n_frames, fps_num, fps_den,
     channels, has_audio, audio_rate, audio_channels,
     _, _) = struct.unpack_from("<IIIHHBBIBBB", data, offset)
    offset += struct.calcsize("<IIIHHBBIBBB")  # 25 bytes

    metadata = {
        "width": width, "height": height, "n_frames": n_frames,
        "fps": (fps_num, fps_den), "channels": channels,
        "audio_rate": audio_rate, "audio_channels": audio_channels,
    }

    # --- Audio packet ---
    audio_samples = None
    if has_audio:
        pkt_type = data[offset]
        offset += 1
        pkt_size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        audio_data = data[offset:offset + pkt_size]
        offset += pkt_size
        audio_samples = decode_audio_pcm(audio_data)
        metadata["audio_samples"] = audio_samples

    # --- Frame packets ---
    frames = []
    ref_frame = None
    for i in range(n_frames):
        pkt_type = data[offset]
        offset += 1
        pkt_size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        pkt_data = data[offset:offset + pkt_size]
        offset += pkt_size

        if pkt_type == PKT_INTRA:
            frame = decode_frame_intra(pkt_data, width, height)
        elif pkt_type == PKT_INTER:
            frame = decode_frame_inter(pkt_data, ref_frame, width, height)
        else:
            raise ValueError(f"Unknown packet type: {pkt_type}")

        frames.append(frame)
        ref_frame = frame

        if progress_fn:
            progress_fn(i + 1, n_frames)

    # --- Footer ---
    assert data[offset:offset + 4] == FOOTER, "Missing NEXUS footer"

    return frames, metadata
