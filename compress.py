#!/usr/bin/env python3
"""Main compression pipeline runner.

Usage:
    python compress.py [--tiers 1,2,3,4,5] [--source img.png]
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from compressors import (
    Tier, CompressorResult,
    compress_png_max, compress_webp_lossless, compress_jxl_lossless,
    compress_secondary,
    compress_jpeg_near_lossless, compress_webp_near_lossless,
    compress_jxl_near_lossless, compress_avif_near_lossless,
    compress_quality_sweep,
    compress_svd, compress_color_quantize, compress_downscale,
    compress_dct_threshold, compress_wavelet,
    compress_1kb_webp, compress_1kb_jxl, compress_1kb_avif, compress_1kb_jpeg,
    compress_1kb_custom_palette, compress_1kb_hybrid, compress_1kb_parametric,
)
from metrics import compute_metrics
from report import generate_text_report, generate_html_report


BASE_DIR = Path(__file__).parent
DEFAULT_SOURCE = BASE_DIR / "img.png"
OUTPUT_DIR = BASE_DIR / "output"

TIER_DIRS = {
    Tier.LOSSLESS: "tier1_lossless",
    Tier.NEAR_LOSSLESS: "tier2_near_lossless",
    Tier.HIGH_QUALITY_LOSSY: "tier3_lossy",
    Tier.EXTREME: "tier4_extreme",
    Tier.ONE_KB: "tier5_1kb",
}


def load_source(path: Path) -> tuple[Image.Image, np.ndarray, int]:
    """Load image, strip alpha, return (PIL RGB, numpy RGB array, file size)."""
    source_size = path.stat().st_size
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    print(f"  Loaded: {arr.shape[1]}x{arr.shape[0]} RGB, {source_size:,} bytes")
    return img, arr, source_size


def ensure_dirs(tiers: list[int]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for tier in Tier:
        if tier.value in tiers:
            (OUTPUT_DIR / TIER_DIRS[tier]).mkdir(exist_ok=True)


def compute_all_metrics(results: list[CompressorResult],
                        original: np.ndarray) -> None:
    """Compute SSIM/PSNR for all results."""
    for r in results:
        try:
            is_custom = r.format in ("svd", "cpal", "chyb", "cpar")
            r.ssim, r.psnr = compute_metrics(
                original, r.output_path,
                is_custom=is_custom,
                decompress_cmd=r.decompressor,
            )
        except Exception as e:
            r.notes += f" [metric error: {e}]"


def run_tier1(img: Image.Image, arr: np.ndarray, source_size: int,
              output_dir: Path) -> list[CompressorResult]:
    print("\n  [Tier 1] Lossless optimization...")
    results = []
    # Run primary compressors in parallel (img.copy() for thread safety)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(compress_png_max, img.copy(), output_dir, source_size),
            pool.submit(compress_webp_lossless, img.copy(), output_dir, source_size),
            pool.submit(compress_jxl_lossless, img.copy(), output_dir, source_size),
        ]
        for f in futures:
            try:
                results.extend(f.result())
            except Exception as e:
                print(f"    Warning: {e}")

    # Secondary compression on primary results
    secondary = compress_secondary(results, output_dir, source_size)
    results.extend(secondary)

    for r in results:
        print(f"    {r.name}: {r.compressed_size:,} bytes ({r.compression_ratio:.1f}x)")
    return results


def run_tier2(img: Image.Image, arr: np.ndarray, source_size: int,
              output_dir: Path) -> list[CompressorResult]:
    print("\n  [Tier 2] Near-lossless...")
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(compress_jpeg_near_lossless, img.copy(), output_dir, source_size),
            pool.submit(compress_webp_near_lossless, img.copy(), output_dir, source_size),
            pool.submit(compress_jxl_near_lossless, img.copy(), output_dir, source_size),
            pool.submit(compress_avif_near_lossless, img.copy(), output_dir, source_size),
        ]
        for f in futures:
            try:
                results.extend(f.result())
            except Exception as e:
                print(f"    Warning: {e}")

    for r in sorted(results, key=lambda x: x.compressed_size):
        print(f"    {r.name}: {r.compressed_size:,} bytes")
    return results


def run_tier3(img: Image.Image, arr: np.ndarray, source_size: int,
              output_dir: Path) -> list[CompressorResult]:
    print("\n  [Tier 3] Quality sweep across all formats...")
    results = compress_quality_sweep(img, output_dir, source_size)
    print(f"    Generated {len(results)} variants")
    smallest = min(results, key=lambda r: r.compressed_size)
    print(f"    Smallest: {smallest.name} at {smallest.compressed_size:,} bytes")
    return results


def run_tier4(img: Image.Image, arr: np.ndarray, source_size: int,
              output_dir: Path) -> list[CompressorResult]:
    print("\n  [Tier 4] Extreme techniques...")
    results = []

    # Run heavy computations
    fns = [
        ("SVD", lambda: compress_svd(arr, output_dir, source_size)),
        ("Color Quantize", lambda: compress_color_quantize(img, arr, output_dir, source_size)),
        ("Downscale", lambda: compress_downscale(img, output_dir, source_size)),
        ("DCT", lambda: compress_dct_threshold(arr, output_dir, source_size)),
        ("Wavelet", lambda: compress_wavelet(arr, output_dir, source_size)),
    ]
    for name, fn in fns:
        try:
            r = fn()
            results.extend(r)
            print(f"    {name}: {len(r)} variants")
        except Exception as e:
            print(f"    {name} failed: {e}")

    if results:
        smallest = min(results, key=lambda r: r.compressed_size)
        print(f"    Smallest: {smallest.name} at {smallest.compressed_size:,} bytes")
    return results


def run_tier5(img: Image.Image, arr: np.ndarray, source_size: int,
              output_dir: Path) -> list[CompressorResult]:
    print("\n  [Tier 5] The 1KB Challenge...")
    results = []

    fns = [
        ("WebP 1KB", lambda: compress_1kb_webp(img, output_dir, source_size)),
        ("JXL 1KB", lambda: compress_1kb_jxl(img, output_dir, source_size)),
        ("AVIF 1KB", lambda: compress_1kb_avif(img, output_dir, source_size)),
        ("JPEG 1KB", lambda: compress_1kb_jpeg(img, output_dir, source_size)),
        ("Custom Palette", lambda: compress_1kb_custom_palette(img, arr, output_dir, source_size)),
        ("Hybrid", lambda: compress_1kb_hybrid(img, output_dir, source_size)),
        ("Parametric", lambda: compress_1kb_parametric(img, arr, output_dir, source_size)),
    ]
    for name, fn in fns:
        try:
            r = fn()
            results.extend(r)
            for ri in r:
                marker = " <=1KB" if ri.compressed_size <= 1024 else ""
                print(f"    {ri.name}: {ri.compressed_size} bytes{marker}")
        except Exception as e:
            print(f"    {name} failed: {e}")

    fits = [r for r in results if r.compressed_size <= 1024]
    print(f"\n    {len(fits)} methods fit in 1KB!")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Compression Pipeline")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--tiers", default="1,2,3,4,5",
                        help="Comma-separated tier numbers to run")
    args = parser.parse_args()

    tiers = [int(t) for t in args.tiers.split(",")]
    print(f"Image Compression Pipeline")
    print(f"  Source: {args.source}")
    print(f"  Tiers: {tiers}")

    img, arr, source_size = load_source(args.source)
    ensure_dirs(tiers)

    all_results: list[CompressorResult] = []
    start = time.time()

    tier_runners = {
        1: run_tier1,
        2: run_tier2,
        3: run_tier3,
        4: run_tier4,
        5: run_tier5,
    }

    for tier_num in tiers:
        runner = tier_runners.get(tier_num)
        if runner:
            output_dir = OUTPUT_DIR / TIER_DIRS[Tier(tier_num)]
            results = runner(img, arr, source_size, output_dir)
            print(f"  Computing quality metrics for {len(results)} results...")
            compute_all_metrics(results, arr)
            all_results.extend(results)

    elapsed = time.time() - start
    print(f"\nTotal: {len(all_results)} methods in {elapsed:.1f}s")

    # Generate reports
    print("Generating reports...")
    generate_text_report(all_results, args.source, OUTPUT_DIR / "results.txt")
    generate_html_report(all_results, args.source, OUTPUT_DIR)

    # Print final summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for tier in Tier:
        tier_results = [r for r in all_results if r.tier == tier]
        if tier_results:
            best = min(tier_results, key=lambda r: r.compressed_size)
            ssim_str = f", SSIM={best.ssim:.4f}" if best.ssim is not None else ""
            print(f"  Tier {tier.value}: {best.name} = "
                  f"{best.compressed_size:,} bytes "
                  f"({best.compression_ratio:.1f}x{ssim_str})")

    fits_1kb = [r for r in all_results if r.tier == Tier.ONE_KB
                and r.compressed_size <= 1024]
    if fits_1kb:
        best_1kb = max(fits_1kb, key=lambda r: r.ssim or 0)
        print(f"\n  Best 1KB result: {best_1kb.name}")
        print(f"    Size: {best_1kb.compressed_size} bytes")
        print(f"    SSIM: {best_1kb.ssim:.4f}" if best_1kb.ssim else "")
        print(f"    Compression: {best_1kb.compression_ratio:.0f}x from original")

    print(f"\nReports: {OUTPUT_DIR / 'results.txt'}")
    print(f"         {OUTPUT_DIR / 'comparison.html'}")


if __name__ == "__main__":
    main()
