"""All compression algorithms across 5 tiers."""

import io
import math
import struct
import subprocess
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
from sklearn.cluster import MiniBatchKMeans

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


class Tier(Enum):
    LOSSLESS = 1
    NEAR_LOSSLESS = 2
    HIGH_QUALITY_LOSSY = 3
    EXTREME = 4
    ONE_KB = 5


@dataclass
class CompressorResult:
    name: str
    tier: Tier
    output_path: Path
    original_size: int
    compressed_size: int
    compression_ratio: float
    ssim: Optional[float] = None
    psnr: Optional[float] = None
    elapsed_seconds: float = 0.0
    format: str = ""
    quality_param: Optional[str] = None
    notes: str = ""
    decompressor: Optional[str] = None


def _make_result(name: str, tier: Tier, output_path: Path,
                 original_size: int, start: float, fmt: str,
                 quality_param: str = "", notes: str = "",
                 decompressor: str | None = None) -> CompressorResult:
    size = output_path.stat().st_size
    return CompressorResult(
        name=name, tier=tier, output_path=output_path,
        original_size=original_size, compressed_size=size,
        compression_ratio=original_size / size if size > 0 else float("inf"),
        elapsed_seconds=time.time() - start, format=fmt,
        quality_param=quality_param, notes=notes, decompressor=decompressor,
    )


def _save_rgb_png(img: Image.Image, path: Path) -> Path:
    """Save RGB image as temp PNG for subprocess tools."""
    rgb = img.convert("RGB")
    rgb.save(path, "PNG")
    return path


# ---------------------------------------------------------------------------
# TIER 1: Lossless Optimization
# ---------------------------------------------------------------------------

def compress_png_max(img: Image.Image, output_dir: Path,
                     source_size: int) -> list[CompressorResult]:
    results = []
    t = time.time()
    out = output_dir / "png_max.png"
    img.save(out, "PNG", optimize=True, compress_level=9)
    results.append(_make_result("png_max", Tier.LOSSLESS, out, source_size, t, "png",
                                "compress_level=9"))
    return results


def compress_webp_lossless(img: Image.Image, output_dir: Path,
                           source_size: int) -> list[CompressorResult]:
    t = time.time()
    out = output_dir / "webp_lossless.webp"
    img.save(out, "WebP", lossless=True, quality=100, method=6)
    return [_make_result("webp_lossless", Tier.LOSSLESS, out, source_size, t, "webp",
                         "lossless, method=6")]


def compress_jxl_lossless(img: Image.Image, output_dir: Path,
                          source_size: int) -> list[CompressorResult]:
    t = time.time()
    tmp_png = output_dir / "_temp_jxl_input.png"
    _save_rgb_png(img, tmp_png)
    out = output_dir / "jxl_lossless.jxl"
    subprocess.run(["cjxl", str(tmp_png), str(out), "-d", "0", "-e", "9"],
                   check=True, capture_output=True)
    tmp_png.unlink(missing_ok=True)
    return [_make_result("jxl_lossless", Tier.LOSSLESS, out, source_size, t, "jxl",
                         "d=0, e=9")]


def compress_secondary(primary_results: list[CompressorResult],
                       output_dir: Path,
                       source_size: int) -> list[CompressorResult]:
    """Apply brotli/zstd/xz on top of lossless outputs."""
    results = []
    compressors = [
        ("brotli", ["brotli", "--best", "-o"], ".br"),
        ("zstd", ["zstd", "-19", "--ultra", "-o"], ".zst"),
        ("xz", ["xz", "-9e", "--keep", "--stdout"], ".xz"),
    ]
    for pr in primary_results:
        for cname, cmd, ext in compressors:
            t = time.time()
            out = output_dir / f"{pr.name}_{cname}{ext}"
            try:
                if cname == "xz":
                    with open(out, "wb") as f:
                        subprocess.run(cmd + [str(pr.output_path)],
                                       check=True, stdout=f, stderr=subprocess.PIPE)
                else:
                    subprocess.run(cmd + [str(out), str(pr.output_path)],
                                   check=True, capture_output=True)
                if out.exists():
                    results.append(_make_result(
                        f"{pr.name}+{cname}", Tier.LOSSLESS, out, source_size, t,
                        f"{pr.format}+{cname}", notes="secondary compression"))
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
    return results


# ---------------------------------------------------------------------------
# TIER 2: Near-Lossless (SSIM >= 0.99)
# ---------------------------------------------------------------------------

def compress_jpeg_near_lossless(img: Image.Image, output_dir: Path,
                                source_size: int) -> list[CompressorResult]:
    results = []
    for q in [95, 97, 99, 100]:
        t = time.time()
        out = output_dir / f"jpeg_q{q}.jpg"
        img.save(out, "JPEG", quality=q, optimize=True, subsampling=0)
        results.append(_make_result(f"jpeg_q{q}", Tier.NEAR_LOSSLESS, out,
                                    source_size, t, "jpeg", f"q={q}"))
    return results


def compress_webp_near_lossless(img: Image.Image, output_dir: Path,
                                source_size: int) -> list[CompressorResult]:
    results = []
    for q in [95, 97, 99, 100]:
        t = time.time()
        out = output_dir / f"webp_q{q}.webp"
        img.save(out, "WebP", quality=q, method=6)
        results.append(_make_result(f"webp_q{q}", Tier.NEAR_LOSSLESS, out,
                                    source_size, t, "webp", f"q={q}"))
    return results


def compress_jxl_near_lossless(img: Image.Image, output_dir: Path,
                               source_size: int) -> list[CompressorResult]:
    results = []
    tmp_png = output_dir / "_temp_jxl_nl.png"
    _save_rgb_png(img, tmp_png)
    for d in [0.1, 0.2, 0.3, 0.5]:
        t = time.time()
        out = output_dir / f"jxl_d{d}.jxl"
        subprocess.run(["cjxl", str(tmp_png), str(out), "-d", str(d), "-e", "9"],
                       check=True, capture_output=True)
        results.append(_make_result(f"jxl_d{d}", Tier.NEAR_LOSSLESS, out,
                                    source_size, t, "jxl", f"d={d}"))
    tmp_png.unlink(missing_ok=True)
    return results


def compress_avif_near_lossless(img: Image.Image, output_dir: Path,
                                source_size: int) -> list[CompressorResult]:
    results = []
    for q in [90, 95, 97, 100]:
        t = time.time()
        out = output_dir / f"avif_q{q}.avif"
        try:
            img.save(out, "AVIF", quality=q)
            results.append(_make_result(f"avif_q{q}", Tier.NEAR_LOSSLESS, out,
                                        source_size, t, "avif", f"q={q}"))
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# TIER 3: High-Quality Lossy (Full Quality Sweep)
# ---------------------------------------------------------------------------

def compress_quality_sweep(img: Image.Image, output_dir: Path,
                           source_size: int) -> list[CompressorResult]:
    results = []
    qualities = [90, 80, 70, 60, 50, 40, 30, 20, 10]

    # JPEG sweep
    for q in qualities:
        t = time.time()
        out = output_dir / f"jpeg_q{q}.jpg"
        img.save(out, "JPEG", quality=q, optimize=True, subsampling=0)
        results.append(_make_result(f"jpeg_q{q}", Tier.HIGH_QUALITY_LOSSY, out,
                                    source_size, t, "jpeg", f"q={q}"))

    # WebP sweep
    for q in qualities:
        t = time.time()
        out = output_dir / f"webp_q{q}.webp"
        img.save(out, "WebP", quality=q, method=6)
        results.append(_make_result(f"webp_q{q}", Tier.HIGH_QUALITY_LOSSY, out,
                                    source_size, t, "webp", f"q={q}"))

    # AVIF sweep
    for q in qualities:
        t = time.time()
        out = output_dir / f"avif_q{q}.avif"
        try:
            img.save(out, "AVIF", quality=q)
            results.append(_make_result(f"avif_q{q}", Tier.HIGH_QUALITY_LOSSY, out,
                                        source_size, t, "avif", f"q={q}"))
        except Exception:
            pass

    # JXL sweep
    tmp_png = output_dir / "_temp_jxl_sweep.png"
    _save_rgb_png(img, tmp_png)
    for d in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]:
        t = time.time()
        out = output_dir / f"jxl_d{d}.jxl"
        try:
            subprocess.run(["cjxl", str(tmp_png), str(out), "-d", str(d), "-e", "7"],
                           check=True, capture_output=True)
            results.append(_make_result(f"jxl_d{d}", Tier.HIGH_QUALITY_LOSSY, out,
                                        source_size, t, "jxl", f"d={d}"))
        except subprocess.CalledProcessError:
            pass
    tmp_png.unlink(missing_ok=True)
    return results


# ---------------------------------------------------------------------------
# TIER 4: Extreme Techniques
# ---------------------------------------------------------------------------

def compress_svd(arr: np.ndarray, output_dir: Path,
                 source_size: int) -> list[CompressorResult]:
    """SVD rank reduction per channel."""
    results = []
    h, w, c = arr.shape

    for rank in [5, 10, 20, 50, 100]:
        t = time.time()
        reconstructed = np.zeros_like(arr, dtype=np.float64)
        for ch in range(min(c, 3)):
            channel = arr[:, :, ch].astype(np.float64)
            U, S, Vt = np.linalg.svd(channel, full_matrices=False)
            U_k = U[:, :rank]
            S_k = S[:rank]
            Vt_k = Vt[:rank, :]
            reconstructed[:, :, ch] = U_k @ np.diag(S_k) @ Vt_k

        recon_clipped = np.clip(reconstructed, 0, 255).astype(np.uint8)
        out_png = output_dir / f"svd_rank{rank}.png"
        Image.fromarray(recon_clipped).save(out_png, "PNG", optimize=True)
        results.append(_make_result(f"svd_rank{rank}", Tier.EXTREME, out_png,
                                    source_size, t, "png", f"rank={rank}",
                                    notes="SVD rank reduction"))

        # Also save as custom binary .svd format with zlib
        t2 = time.time()
        out_svd = output_dir / f"svd_rank{rank}.svd"
        _save_svd_binary(arr, rank, out_svd)
        results.append(_make_result(f"svd_rank{rank}_binary", Tier.EXTREME, out_svd,
                                    source_size, t2, "svd", f"rank={rank}",
                                    notes="Custom SVD binary",
                                    decompressor="python3 decompress.py"))
    return results


def _save_svd_binary(arr: np.ndarray, rank: int, path: Path) -> None:
    """Save SVD decomposition as custom binary format with zlib compression."""
    h, w, c = arr.shape
    channels = min(c, 3)
    # Header: magic(4) + width(2) + height(2) + rank(2) + channels(1) + pad(1) = 12 bytes
    header = struct.pack("<4sHHHBB", b"SVD1", w, h, rank, channels, 0)
    body = bytearray()
    for ch in range(channels):
        channel = arr[:, :, ch].astype(np.float64)
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)
        U_k = U[:, :rank]
        S_k = S[:rank].astype(np.float32)
        Vt_k = Vt[:rank, :]
        # Quantize U and Vt to uint8 with min/max scaling
        u_min, u_max = float(U_k.min()), float(U_k.max())
        vt_min, vt_max = float(Vt_k.min()), float(Vt_k.max())
        u_range = u_max - u_min if u_max != u_min else 1.0
        vt_range = vt_max - vt_min if vt_max != vt_min else 1.0
        U_q = ((U_k - u_min) / u_range * 255).astype(np.uint8)
        Vt_q = ((Vt_k - vt_min) / vt_range * 255).astype(np.uint8)
        body += struct.pack("<ff", u_min, u_max)
        body += U_q.tobytes()
        body += S_k.tobytes()
        body += struct.pack("<ff", vt_min, vt_max)
        body += Vt_q.tobytes()
    compressed_body = zlib.compress(bytes(body), 9)
    with open(path, "wb") as f:
        f.write(header)
        f.write(compressed_body)


def compress_color_quantize(img: Image.Image, arr: np.ndarray, output_dir: Path,
                            source_size: int) -> list[CompressorResult]:
    """K-means color quantization."""
    results = []
    pixels = arr[:, :, :3].reshape(-1, 3).astype(np.float32)
    h, w = arr.shape[:2]

    for k in [8, 16, 32, 64, 128, 256]:
        t = time.time()
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init=3, random_state=42)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_.astype(np.uint8)
        quantized = centers[labels].reshape(h, w, 3)
        out = output_dir / f"kmeans_k{k}.png"
        Image.fromarray(quantized).save(out, "PNG", optimize=True)
        results.append(_make_result(f"kmeans_k{k}", Tier.EXTREME, out,
                                    source_size, t, "png", f"k={k}",
                                    notes="K-means color quantization"))
    return results


def compress_downscale(img: Image.Image, output_dir: Path,
                       source_size: int) -> list[CompressorResult]:
    """Downscale + high-quality encode."""
    results = []
    w, h = img.size
    for scale_name, factor in [("2x", 2), ("4x", 4), ("8x", 8)]:
        new_w, new_h = w // factor, h // factor
        small = img.resize((new_w, new_h), Image.LANCZOS)
        # WebP
        t = time.time()
        out = output_dir / f"downscale_{scale_name}_webp90.webp"
        small.save(out, "WebP", quality=90, method=6)
        results.append(_make_result(f"downscale_{scale_name}_webp90", Tier.EXTREME, out,
                                    source_size, t, "webp", f"scale=1/{factor}, q=90",
                                    notes=f"Downscaled to {new_w}x{new_h}"))
        # JXL
        t = time.time()
        tmp = output_dir / f"_temp_ds_{scale_name}.png"
        small.save(tmp, "PNG")
        out_jxl = output_dir / f"downscale_{scale_name}_jxl2.jxl"
        try:
            subprocess.run(["cjxl", str(tmp), str(out_jxl), "-d", "2", "-e", "7"],
                           check=True, capture_output=True)
            results.append(_make_result(f"downscale_{scale_name}_jxl2", Tier.EXTREME,
                                        out_jxl, source_size, t, "jxl",
                                        f"scale=1/{factor}, d=2",
                                        notes=f"Downscaled to {new_w}x{new_h}"))
        except subprocess.CalledProcessError:
            pass
        tmp.unlink(missing_ok=True)
    return results


def compress_dct_threshold(arr: np.ndarray, output_dir: Path,
                           source_size: int) -> list[CompressorResult]:
    """DCT coefficient thresholding (JPEG-style but custom thresholds)."""
    results = []
    h, w, c = arr.shape
    # Pad to multiple of 8
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    padded = np.pad(arr[:, :, :3].astype(np.float64),
                    ((0, ph), (0, pw), (0, 0)), mode="edge")

    for pct in [90, 95, 99]:
        t = time.time()
        reconstructed = np.zeros_like(padded)
        for ch in range(3):
            channel = padded[:, :, ch]
            dct_coeffs = dctn(channel, type=2, norm="ortho")
            threshold = np.percentile(np.abs(dct_coeffs), pct)
            dct_coeffs[np.abs(dct_coeffs) < threshold] = 0
            reconstructed[:, :, ch] = idctn(dct_coeffs, type=2, norm="ortho")
        recon = np.clip(reconstructed[:h, :w, :], 0, 255).astype(np.uint8)
        out = output_dir / f"dct_pct{pct}.png"
        Image.fromarray(recon).save(out, "PNG", optimize=True)
        results.append(_make_result(f"dct_pct{pct}", Tier.EXTREME, out,
                                    source_size, t, "png", f"percentile={pct}",
                                    notes="DCT coefficient thresholding"))
    return results


def compress_wavelet(arr: np.ndarray, output_dir: Path,
                     source_size: int) -> list[CompressorResult]:
    """Wavelet compression using pywavelets."""
    if not HAS_PYWT:
        return []
    results = []
    h, w, c = arr.shape

    for wavelet_name in ["haar", "db2"]:
        for thresh_pct in [90, 95, 99]:
            t = time.time()
            reconstructed = np.zeros((h, w, min(c, 3)), dtype=np.float64)
            for ch in range(min(c, 3)):
                channel = arr[:, :, ch].astype(np.float64)
                coeffs = pywt.wavedec2(channel, wavelet_name, level=3)
                # Threshold detail coefficients
                all_detail = np.concatenate([d.ravel() for level in coeffs[1:]
                                             for d in level])
                threshold = np.percentile(np.abs(all_detail), thresh_pct)
                new_coeffs = [coeffs[0]]
                for level_details in coeffs[1:]:
                    new_level = tuple(
                        np.where(np.abs(d) < threshold, 0, d)
                        for d in level_details
                    )
                    new_coeffs.append(new_level)
                reconstructed[:, :, ch] = pywt.waverec2(new_coeffs, wavelet_name)[:h, :w]
            recon = np.clip(reconstructed, 0, 255).astype(np.uint8)
            out = output_dir / f"wavelet_{wavelet_name}_pct{thresh_pct}.png"
            Image.fromarray(recon).save(out, "PNG", optimize=True)
            results.append(_make_result(
                f"wavelet_{wavelet_name}_pct{thresh_pct}", Tier.EXTREME, out,
                source_size, t, "png", f"wavelet={wavelet_name}, pct={thresh_pct}",
                notes="Wavelet coefficient thresholding"))
    return results


# ---------------------------------------------------------------------------
# TIER 5: The 1KB Challenge
# ---------------------------------------------------------------------------

def compress_1kb_webp(img: Image.Image, output_dir: Path,
                      source_size: int) -> list[CompressorResult]:
    """Binary search for largest WebP that fits in 1024 bytes."""
    results = []
    target = 1024
    w_orig, h_orig = img.size
    best_size = 0
    best_dim = 0
    best_q = 0

    # Coarse grid search
    for dim in range(120, 16, -4):
        for q in range(80, 1, -5):
            small = img.resize((dim, dim), Image.LANCZOS)
            buf = io.BytesIO()
            small.save(buf, "WebP", quality=q, method=6)
            size = buf.tell()
            if size <= target and size > best_size:
                best_size = size
                best_dim = dim
                best_q = q

    # Fine-tune around best
    if best_dim > 0:
        for dim in range(best_dim + 4, best_dim - 5, -1):
            if dim < 8:
                continue
            for q in range(min(best_q + 10, 100), max(best_q - 10, 1), -1):
                small = img.resize((dim, dim), Image.LANCZOS)
                buf = io.BytesIO()
                small.save(buf, "WebP", quality=q, method=6)
                size = buf.tell()
                if size <= target and size > best_size:
                    best_size = size
                    best_dim = dim
                    best_q = q

    if best_dim > 0:
        t = time.time()
        small = img.resize((best_dim, best_dim), Image.LANCZOS)
        out = output_dir / f"webp_{best_dim}x{best_dim}_q{best_q}.webp"
        small.save(out, "WebP", quality=best_q, method=6)
        results.append(_make_result(
            f"webp_{best_dim}x{best_dim}_q{best_q}", Tier.ONE_KB, out,
            source_size, t, "webp", f"dim={best_dim}, q={best_q}",
            notes=f"Best WebP fitting 1KB: {best_size} bytes"))

    # Also a few fixed presets for comparison
    for dim, q in [(64, 50), (48, 75), (32, 95)]:
        t = time.time()
        small = img.resize((dim, dim), Image.LANCZOS)
        out = output_dir / f"webp_{dim}x{dim}_q{q}.webp"
        small.save(out, "WebP", quality=q, method=6)
        actual = out.stat().st_size
        if actual <= target:
            results.append(_make_result(
                f"webp_{dim}x{dim}_q{q}", Tier.ONE_KB, out,
                source_size, t, "webp", f"dim={dim}, q={q}",
                notes=f"Fixed preset: {actual} bytes"))
    return results


def compress_1kb_jxl(img: Image.Image, output_dir: Path,
                     source_size: int) -> list[CompressorResult]:
    """JXL tiny thumbnails targeting 1KB."""
    results = []
    target = 1024
    best_size = 0
    best_dim = 0
    best_d = 0.0

    for dim in range(80, 16, -4):
        for d in [1.0, 2.0, 3.0, 5.0, 8.0, 15.0]:
            small = img.resize((dim, dim), Image.LANCZOS)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                small.save(tmp.name, "PNG")
                tmp_png = tmp.name
            out_tmp = tempfile.NamedTemporaryFile(suffix=".jxl", delete=False)
            out_path = out_tmp.name
            out_tmp.close()
            try:
                subprocess.run(["cjxl", tmp_png, out_path, "-d", str(d), "-e", "9"],
                               check=True, capture_output=True)
                size = Path(out_path).stat().st_size
                if size <= target and size > best_size:
                    best_size = size
                    best_dim = dim
                    best_d = d
            except subprocess.CalledProcessError:
                pass
            Path(tmp_png).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)

    if best_dim > 0:
        t = time.time()
        small = img.resize((best_dim, best_dim), Image.LANCZOS)
        tmp_png = output_dir / "_temp_1kb_jxl.png"
        small.save(tmp_png, "PNG")
        out = output_dir / f"jxl_{best_dim}x{best_dim}_d{best_d}.jxl"
        subprocess.run(["cjxl", str(tmp_png), str(out), "-d", str(best_d), "-e", "9"],
                       check=True, capture_output=True)
        tmp_png.unlink(missing_ok=True)
        results.append(_make_result(
            f"jxl_{best_dim}x{best_dim}_d{best_d}", Tier.ONE_KB, out,
            source_size, t, "jxl", f"dim={best_dim}, d={best_d}",
            notes=f"Best JXL fitting 1KB: {best_size} bytes"))
    return results


def compress_1kb_avif(img: Image.Image, output_dir: Path,
                      source_size: int) -> list[CompressorResult]:
    """AVIF tiny thumbnails targeting 1KB."""
    results = []
    target = 1024
    best_size = 0
    best_dim = 0
    best_q = 0

    for dim in range(120, 16, -8):
        for q in range(80, 1, -10):
            small = img.resize((dim, dim), Image.LANCZOS)
            buf = io.BytesIO()
            try:
                small.save(buf, "AVIF", quality=q)
                size = buf.tell()
                if size <= target and size > best_size:
                    best_size = size
                    best_dim = dim
                    best_q = q
            except Exception:
                pass

    # Fine-tune
    if best_dim > 0:
        for dim in range(best_dim + 8, best_dim - 9, -1):
            if dim < 8:
                continue
            for q in range(min(best_q + 15, 100), max(best_q - 15, 1), -1):
                small = img.resize((dim, dim), Image.LANCZOS)
                buf = io.BytesIO()
                try:
                    small.save(buf, "AVIF", quality=q)
                    size = buf.tell()
                    if size <= target and size > best_size:
                        best_size = size
                        best_dim = dim
                        best_q = q
                except Exception:
                    pass

    if best_dim > 0:
        t = time.time()
        small = img.resize((best_dim, best_dim), Image.LANCZOS)
        out = output_dir / f"avif_{best_dim}x{best_dim}_q{best_q}.avif"
        try:
            small.save(out, "AVIF", quality=best_q)
            results.append(_make_result(
                f"avif_{best_dim}x{best_dim}_q{best_q}", Tier.ONE_KB, out,
                source_size, t, "avif", f"dim={best_dim}, q={best_q}",
                notes=f"Best AVIF fitting 1KB: {best_size} bytes"))
        except Exception:
            pass
    return results


def compress_1kb_jpeg(img: Image.Image, output_dir: Path,
                      source_size: int) -> list[CompressorResult]:
    """JPEG tiny thumbnails targeting 1KB."""
    results = []
    target = 1024
    best_size = 0
    best_dim = 0
    best_q = 0

    for dim in range(80, 8, -4):
        for q in range(80, 1, -5):
            small = img.resize((dim, dim), Image.LANCZOS)
            buf = io.BytesIO()
            small.save(buf, "JPEG", quality=q, optimize=True)
            size = buf.tell()
            if size <= target and size > best_size:
                best_size = size
                best_dim = dim
                best_q = q

    if best_dim > 0:
        # Fine-tune
        for dim in range(best_dim + 4, best_dim - 5, -1):
            if dim < 8:
                continue
            for q in range(min(best_q + 10, 100), max(best_q - 10, 1), -1):
                small = img.resize((dim, dim), Image.LANCZOS)
                buf = io.BytesIO()
                small.save(buf, "JPEG", quality=q, optimize=True)
                size = buf.tell()
                if size <= target and size > best_size:
                    best_size = size
                    best_dim = dim
                    best_q = q

        t = time.time()
        small = img.resize((best_dim, best_dim), Image.LANCZOS)
        out = output_dir / f"jpeg_{best_dim}x{best_dim}_q{best_q}.jpg"
        small.save(out, "JPEG", quality=best_q, optimize=True)
        results.append(_make_result(
            f"jpeg_{best_dim}x{best_dim}_q{best_q}", Tier.ONE_KB, out,
            source_size, t, "jpeg", f"dim={best_dim}, q={best_q}",
            notes=f"Best JPEG fitting 1KB: {best_size} bytes"))
    return results


def compress_1kb_custom_palette(img: Image.Image, arr: np.ndarray, output_dir: Path,
                                source_size: int) -> list[CompressorResult]:
    """Custom palette format: header + palette + zlib-compressed bit-packed indices."""
    results = []
    target = 1024

    best_size = 0
    best_dim = 0
    best_k = 0
    best_data = b""

    for dim in range(96, 16, -4):
        small = img.resize((dim, dim), Image.LANCZOS)
        small_arr = np.array(small)[:, :, :3].reshape(-1, 3).astype(np.float32)
        for k in [4, 8, 16, 32]:
            bits_per_pixel = math.ceil(math.log2(k))
            kmeans = MiniBatchKMeans(n_clusters=k, batch_size=512, n_init=1,
                                     random_state=42)
            labels = kmeans.fit_predict(small_arr)
            centers = kmeans.cluster_centers_.clip(0, 255).astype(np.uint8)
            # Pack indices
            packed = _bitpack_indices(labels, bits_per_pixel)
            compressed = zlib.compress(packed, 9)
            # Header: magic(4) + width(2) + height(2) + n_colors(1) + bpp(1) = 10
            header = struct.pack("<4sHHBB", b"CPAL", dim, dim, k, bits_per_pixel)
            palette_bytes = centers.tobytes()
            data = header + palette_bytes + compressed
            if len(data) <= target and len(data) > best_size:
                best_size = len(data)
                best_dim = dim
                best_k = k
                best_data = data

    if best_data:
        t = time.time()
        out = output_dir / f"cpal_{best_dim}x{best_dim}_k{best_k}.cpal"
        with open(out, "wb") as f:
            f.write(best_data)
        results.append(_make_result(
            f"cpal_{best_dim}x{best_dim}_k{best_k}", Tier.ONE_KB, out,
            source_size, t, "cpal", f"dim={best_dim}, k={best_k}",
            notes=f"Custom palette: {best_size} bytes",
            decompressor="python3 decompress.py"))
    return results


def _bitpack_indices(indices: np.ndarray, bits_per_pixel: int) -> bytes:
    """Pack indices into a byte array using bits_per_pixel bits each."""
    total_bits = len(indices) * bits_per_pixel
    total_bytes = (total_bits + 7) // 8
    result = bytearray(total_bytes)
    bit_pos = 0
    for idx in indices:
        byte_offset = bit_pos // 8
        bit_offset = bit_pos % 8
        # Write the value across potentially 2 bytes
        val = int(idx) & ((1 << bits_per_pixel) - 1)
        result[byte_offset] |= (val << bit_offset) & 0xFF
        if bit_offset + bits_per_pixel > 8 and byte_offset + 1 < total_bytes:
            result[byte_offset + 1] |= (val >> (8 - bit_offset)) & 0xFF
        bit_pos += bits_per_pixel
    return bytes(result)


def compress_1kb_hybrid(img: Image.Image, output_dir: Path,
                        source_size: int) -> list[CompressorResult]:
    """Hybrid: tiny WebP thumbnail + edge structure in 1KB."""
    results = []
    target = 1024
    import cv2

    # Try different splits between thumbnail and edge data
    for thumb_dim in [40, 48, 56]:
        thumb = img.resize((thumb_dim, thumb_dim), Image.LANCZOS)
        # Find WebP quality that leaves room for edge data
        for thumb_q in range(60, 5, -5):
            thumb_buf = io.BytesIO()
            thumb.save(thumb_buf, "WebP", quality=thumb_q, method=6)
            thumb_bytes = thumb_buf.getvalue()
            thumb_size = len(thumb_bytes)
            remaining = target - 12 - thumb_size  # 12 bytes header
            if remaining < 50:
                continue
            # Edge map at smaller resolution
            edge_dim = thumb_dim // 2
            gray = np.array(img.convert("L").resize((edge_dim, edge_dim), Image.LANCZOS))
            edges = cv2.Canny(gray, 50, 150)
            # Pack edges as bitfield + zlib
            edge_packed = np.packbits(edges > 0).tobytes()
            edge_compressed = zlib.compress(edge_packed, 9)
            if len(edge_compressed) <= remaining:
                # Header: magic(4) + thumb_dim(1) + thumb_q(1) + edge_dim(1) +
                #          thumb_size(2) + edge_size(2) + pad(1) = 12
                header = struct.pack("<4sBBBHHB", b"CHYB", thumb_dim, thumb_q,
                                     edge_dim, thumb_size, len(edge_compressed), 0)
                data = header + thumb_bytes + edge_compressed
                if len(data) <= target:
                    t = time.time()
                    out = output_dir / f"hybrid_{thumb_dim}_{edge_dim}.chyb"
                    with open(out, "wb") as f:
                        f.write(data)
                    results.append(_make_result(
                        f"hybrid_{thumb_dim}_{edge_dim}", Tier.ONE_KB, out,
                        source_size, t, "chyb",
                        f"thumb={thumb_dim}, edge={edge_dim}, q={thumb_q}",
                        notes=f"Hybrid: {len(data)} bytes",
                        decompressor="python3 decompress.py"))
                    break
        if results:
            break
    return results


def compress_1kb_parametric(img: Image.Image, arr: np.ndarray, output_dir: Path,
                            source_size: int) -> list[CompressorResult]:
    """Parametric: store dominant colors + gradient parameters."""
    t = time.time()
    target = 1024

    # Analyze image: extract dominant colors and their regions
    h, w = arr.shape[:2]
    small = img.resize((64, 64), Image.LANCZOS)
    small_arr = np.array(small)[:, :, :3]

    # K-means for dominant colors
    pixels = small_arr.reshape(-1, 3).astype(np.float32)
    kmeans = MiniBatchKMeans(n_clusters=8, batch_size=512, n_init=3, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.clip(0, 255).astype(np.uint8)

    # Compute per-row average color for gradient approximation
    row_avgs = np.mean(small_arr.astype(np.float32), axis=1).astype(np.uint8)  # 64x3

    # Store: color map (8x3=24) + row gradient (64x3=192) + label map compressed
    label_map = labels.reshape(64, 64).astype(np.uint8)
    label_compressed = zlib.compress(label_map.tobytes(), 9)

    # Header: magic(4) + n_colors(1) + grad_rows(1) + label_size(2) = 8
    header = struct.pack("<4sBBH", b"CPAR", 8, 64, len(label_compressed))
    data = header + centers.tobytes() + row_avgs.tobytes() + label_compressed

    if len(data) <= target:
        out = output_dir / f"parametric_8c_64g.cpar"
        with open(out, "wb") as f:
            f.write(data)
        return [_make_result(
            "parametric_8c_64g", Tier.ONE_KB, out,
            source_size, t, "cpar", "8 colors, 64 gradient rows",
            notes=f"Parametric: {len(data)} bytes",
            decompressor="python3 decompress.py")]

    # Fallback: smaller
    small32 = img.resize((32, 32), Image.LANCZOS)
    small32_arr = np.array(small32)[:, :, :3]
    pixels32 = small32_arr.reshape(-1, 3).astype(np.float32)
    kmeans4 = MiniBatchKMeans(n_clusters=4, batch_size=256, n_init=3, random_state=42)
    labels4 = kmeans4.fit_predict(pixels32)
    centers4 = kmeans4.cluster_centers_.clip(0, 255).astype(np.uint8)
    row_avgs32 = np.mean(small32_arr.astype(np.float32), axis=1).astype(np.uint8)
    label_map4 = labels4.reshape(32, 32).astype(np.uint8)
    label_compressed4 = zlib.compress(label_map4.tobytes(), 9)
    header4 = struct.pack("<4sBBH", b"CPAR", 4, 32, len(label_compressed4))
    data4 = header4 + centers4.tobytes() + row_avgs32.tobytes() + label_compressed4
    out = output_dir / f"parametric_4c_32g.cpar"
    with open(out, "wb") as f:
        f.write(data4)
    return [_make_result(
        "parametric_4c_32g", Tier.ONE_KB, out,
        source_size, t, "cpar", "4 colors, 32 gradient rows",
        notes=f"Parametric: {len(data4)} bytes",
        decompressor="python3 decompress.py")]
