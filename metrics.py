"""Quality metrics for image compression comparison."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def load_and_prepare(compressed_path: Path, original_size: tuple,
                     is_custom: bool = False,
                     decompress_cmd: str | None = None) -> np.ndarray:
    """Load a compressed image and resize to original dimensions for comparison.

    For custom formats, runs decompress_cmd to produce a temp PNG first.
    """
    ext = compressed_path.suffix.lower()

    if is_custom and decompress_cmd:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(
            decompress_cmd.split() + [str(compressed_path), tmp_path,
                                       "--target-size", f"{original_size[1]}x{original_size[0]}"],
            check=True, capture_output=True,
        )
        img = Image.open(tmp_path).convert("RGB")
        Path(tmp_path).unlink(missing_ok=True)
    elif ext == ".jxl":
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(["djxl", str(compressed_path), tmp_path],
                       check=True, capture_output=True)
        img = Image.open(tmp_path).convert("RGB")
        Path(tmp_path).unlink(missing_ok=True)
    else:
        img = Image.open(compressed_path).convert("RGB")

    arr = np.array(img)
    h, w = original_size[:2]
    if arr.shape[0] != h or arr.shape[1] != w:
        img_resized = img.resize((w, h), Image.LANCZOS)
        arr = np.array(img_resized)
    return arr


def compute_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
    """Compute SSIM between two RGB arrays. Returns float in [0, 1]."""
    return structural_similarity(
        original, compressed, channel_axis=2, data_range=255,
    )


def compute_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """Compute PSNR between two RGB arrays. Returns dB (inf for identical)."""
    if np.array_equal(original, compressed):
        return float("inf")
    return peak_signal_noise_ratio(original, compressed, data_range=255)


def compute_metrics(original: np.ndarray, compressed_path: Path,
                    is_custom: bool = False,
                    decompress_cmd: str | None = None) -> tuple[float, float]:
    """Compute (SSIM, PSNR) for a compressed file against the original array."""
    compressed = load_and_prepare(
        compressed_path, original.shape, is_custom, decompress_cmd,
    )
    ssim = compute_ssim(original, compressed)
    psnr = compute_psnr(original, compressed)
    return ssim, psnr
