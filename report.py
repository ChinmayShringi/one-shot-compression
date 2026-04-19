"""Generate text and HTML reports from compression results."""

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from compressors import CompressorResult, Tier


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.2f} MB"


def _format_ratio(ratio: float) -> str:
    if ratio == float("inf"):
        return "inf"
    return f"{ratio:.1f}x"


def _format_metric(val: float | None) -> str:
    if val is None:
        return "N/A"
    if val == float("inf"):
        return "inf"
    return f"{val:.4f}"


def generate_text_report(results: list[CompressorResult],
                         source_path: Path, output_path: Path) -> None:
    """Write results.txt with per-tier tables and winners."""
    lines = [
        "=" * 80,
        "IMAGE COMPRESSION PIPELINE - RESULTS REPORT",
        "=" * 80,
        f"Source: {source_path}",
        f"Source size: {_format_size(results[0].original_size) if results else 'N/A'}",
        "",
    ]

    tier_names = {
        Tier.LOSSLESS: "TIER 1: Lossless Optimization",
        Tier.NEAR_LOSSLESS: "TIER 2: Near-Lossless (SSIM >= 0.99)",
        Tier.HIGH_QUALITY_LOSSY: "TIER 3: High-Quality Lossy Sweep",
        Tier.EXTREME: "TIER 4: Extreme Techniques",
        Tier.ONE_KB: "TIER 5: The 1KB Challenge",
    }

    for tier in Tier:
        tier_results = [r for r in results if r.tier == tier]
        if not tier_results:
            continue
        tier_results.sort(key=lambda r: r.compressed_size)

        lines.append("-" * 80)
        lines.append(tier_names.get(tier, str(tier)))
        lines.append("-" * 80)
        lines.append(f"{'Name':<35} {'Size':>10} {'Ratio':>8} {'SSIM':>8} "
                     f"{'PSNR':>8} {'Time':>6} {'Params'}")
        lines.append("-" * 80)

        for r in tier_results:
            lines.append(
                f"{r.name:<35} {_format_size(r.compressed_size):>10} "
                f"{_format_ratio(r.compression_ratio):>8} "
                f"{_format_metric(r.ssim):>8} "
                f"{_format_metric(r.psnr):>8} "
                f"{r.elapsed_seconds:>5.1f}s "
                f"{r.quality_param or ''}"
            )

        # Winner
        winner = tier_results[0]
        lines.append("")
        lines.append(f"  WINNER: {winner.name} at {_format_size(winner.compressed_size)} "
                     f"({_format_ratio(winner.compression_ratio)} compression)")
        if winner.ssim is not None:
            lines.append(f"  Quality: SSIM={_format_metric(winner.ssim)}, "
                         f"PSNR={_format_metric(winner.psnr)}")
        lines.append("")

    # Overall summary
    lines.append("=" * 80)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 80)
    for tier in Tier:
        tier_results = [r for r in results if r.tier == tier]
        if tier_results:
            best = min(tier_results, key=lambda r: r.compressed_size)
            lines.append(f"  {tier_names.get(tier, str(tier))[:40]:<42} "
                         f"Best: {_format_size(best.compressed_size):>10} "
                         f"({best.name})")

    # 1KB results highlight
    kb_results = [r for r in results if r.tier == Tier.ONE_KB and r.compressed_size <= 1024]
    if kb_results:
        lines.append("")
        lines.append("  FILES FITTING IN 1KB:")
        for r in sorted(kb_results, key=lambda r: -(r.ssim or 0)):
            lines.append(f"    {r.name}: {r.compressed_size} bytes, "
                         f"SSIM={_format_metric(r.ssim)}")

    lines.append("")
    lines.append(f"Total methods tested: {len(results)}")
    lines.append("=" * 80)

    output_path.write_text("\n".join(lines))


def _img_to_base64(path: Path, max_dim: int = 200) -> str:
    """Load image, resize to thumbnail, return base64 JPEG string."""
    try:
        ext = path.suffix.lower()
        if ext == ".jxl":
            import subprocess
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                subprocess.run(["djxl", str(path), tmp.name],
                               check=True, capture_output=True)
                img = Image.open(tmp.name).convert("RGB")
                Path(tmp.name).unlink(missing_ok=True)
        elif ext in (".svd", ".cpal", ".chyb", ".cpar"):
            from decompress import decompress as _decompress
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                _decompress(str(path), tmp.name)
                img = Image.open(tmp.name).convert("RGB")
                Path(tmp.name).unlink(missing_ok=True)
        else:
            img = Image.open(path).convert("RGB")
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def generate_html_report(results: list[CompressorResult],
                         source_path: Path, output_dir: Path) -> None:
    """Write comparison.html with embedded thumbnails and sortable tables."""
    # Get source thumbnail
    source_b64 = ""
    try:
        src_img = Image.open(source_path).convert("RGB")
        src_img.thumbnail((300, 300), Image.LANCZOS)
        buf = io.BytesIO()
        src_img.save(buf, "JPEG", quality=85)
        source_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        pass

    tier_names = {
        Tier.LOSSLESS: "Tier 1: Lossless",
        Tier.NEAR_LOSSLESS: "Tier 2: Near-Lossless",
        Tier.HIGH_QUALITY_LOSSY: "Tier 3: Lossy Quality Sweep",
        Tier.EXTREME: "Tier 4: Extreme Techniques",
        Tier.ONE_KB: "Tier 5: 1KB Challenge",
    }

    html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Image Compression Results</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0a0a0a; color: #e0e0e0; padding: 24px; }}
h1 {{ font-size: 28px; margin-bottom: 8px; color: #fff; }}
h2 {{ font-size: 20px; margin: 32px 0 16px; color: #90caf9;
      border-bottom: 1px solid #333; padding-bottom: 8px; }}
.source {{ display: flex; align-items: center; gap: 24px; margin: 16px 0 32px;
           background: #1a1a1a; padding: 16px; border-radius: 8px; }}
.source img {{ border-radius: 4px; }}
.source-info {{ font-size: 14px; color: #999; }}
.source-info b {{ color: #fff; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0 24px; font-size: 13px; }}
th {{ background: #1a1a2e; padding: 10px 8px; text-align: left; cursor: pointer;
      user-select: none; position: sticky; top: 0; }}
th:hover {{ background: #252540; }}
td {{ padding: 8px; border-bottom: 1px solid #222; vertical-align: middle; }}
tr:hover td {{ background: #1a1a1a; }}
.thumb {{ width: 80px; height: 80px; object-fit: cover; border-radius: 4px;
          border: 1px solid #333; }}
.winner {{ background: #0d2818 !important; }}
.winner td {{ border-bottom-color: #1a5e2a; }}
.badge {{ display: inline-block; background: #2e7d32; color: #fff; font-size: 10px;
          padding: 2px 6px; border-radius: 3px; margin-left: 6px; }}
.fits-1kb {{ color: #66bb6a; font-weight: bold; }}
.over-1kb {{ color: #ef5350; }}
.gallery {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }}
.gallery-item {{ background: #1a1a1a; border-radius: 8px; padding: 8px;
                 text-align: center; width: 160px; }}
.gallery-item img {{ width: 140px; height: 140px; object-fit: cover;
                     border-radius: 4px; margin-bottom: 6px; }}
.gallery-item .label {{ font-size: 11px; color: #aaa; }}
.gallery-item .size {{ font-size: 13px; font-weight: bold; color: #fff; }}
.stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
.stat-card {{ background: #1a1a1a; padding: 16px; border-radius: 8px; min-width: 150px; }}
.stat-card .val {{ font-size: 24px; font-weight: bold; color: #90caf9; }}
.stat-card .lbl {{ font-size: 12px; color: #888; margin-top: 4px; }}
</style>
</head>
<body>
<h1>Image Compression Pipeline Results</h1>
<div class="source">
"""]

    if source_b64:
        html_parts.append(f'<img src="data:image/jpeg;base64,{source_b64}" '
                          f'style="max-height:200px">')
    source_size = results[0].original_size if results else 0
    html_parts.append(f"""<div class="source-info">
<b>Source:</b> {source_path.name}<br>
<b>Size:</b> {_format_size(source_size)}<br>
<b>Methods tested:</b> {len(results)}
</div></div>""")

    # Stats cards
    all_1kb = [r for r in results if r.tier == Tier.ONE_KB and r.compressed_size <= 1024]
    best_lossless = min((r for r in results if r.tier == Tier.LOSSLESS),
                        key=lambda r: r.compressed_size, default=None)
    best_overall = min(results, key=lambda r: r.compressed_size, default=None)

    html_parts.append('<div class="stats">')
    if best_lossless:
        html_parts.append(f'<div class="stat-card"><div class="val">'
                          f'{_format_size(best_lossless.compressed_size)}</div>'
                          f'<div class="lbl">Best Lossless</div></div>')
    if all_1kb:
        best_1kb = max(all_1kb, key=lambda r: r.ssim or 0)
        html_parts.append(f'<div class="stat-card"><div class="val">'
                          f'{best_1kb.compressed_size} B</div>'
                          f'<div class="lbl">Best 1KB (SSIM {_format_metric(best_1kb.ssim)})'
                          f'</div></div>')
    html_parts.append(f'<div class="stat-card"><div class="val">{len(results)}</div>'
                      f'<div class="lbl">Total Methods</div></div>')
    html_parts.append(f'<div class="stat-card"><div class="val">{len(all_1kb)}</div>'
                      f'<div class="lbl">Fit in 1KB</div></div>')
    html_parts.append('</div>')

    # Per-tier tables
    for tier in Tier:
        tier_results = [r for r in results if r.tier == tier]
        if not tier_results:
            continue
        tier_results.sort(key=lambda r: r.compressed_size)
        winner = tier_results[0]

        html_parts.append(f'<h2>{tier_names.get(tier, str(tier))}</h2>')

        if tier == Tier.ONE_KB:
            # Gallery view for 1KB results
            html_parts.append('<div class="gallery">')
            for r in tier_results:
                b64 = _img_to_base64(r.output_path)
                size_class = "fits-1kb" if r.compressed_size <= 1024 else "over-1kb"
                html_parts.append(f"""<div class="gallery-item">
                    {"<img src='data:image/jpeg;base64," + b64 + "'>" if b64 else ""}
                    <div class="size {size_class}">{r.compressed_size} B</div>
                    <div class="label">{r.name}</div>
                    <div class="label">SSIM: {_format_metric(r.ssim)}</div>
                </div>""")
            html_parts.append('</div>')

        html_parts.append("""<table>
<thead><tr>
<th>Preview</th><th onclick="sortTable(this,1)">Name</th>
<th onclick="sortTable(this,2)">Size</th><th onclick="sortTable(this,3)">Ratio</th>
<th onclick="sortTable(this,4)">SSIM</th><th onclick="sortTable(this,5)">PSNR</th>
<th onclick="sortTable(this,6)">Time</th><th>Params</th>
</tr></thead><tbody>""")

        for r in tier_results:
            b64 = _img_to_base64(r.output_path)
            row_class = ' class="winner"' if r is winner else ""
            badge = '<span class="badge">WINNER</span>' if r is winner else ""
            img_tag = (f'<img class="thumb" src="data:image/jpeg;base64,{b64}">'
                       if b64 else "<em>N/A</em>")
            html_parts.append(f"""<tr{row_class}>
<td>{img_tag}</td>
<td>{r.name}{badge}</td>
<td data-v="{r.compressed_size}">{_format_size(r.compressed_size)}</td>
<td data-v="{r.compression_ratio:.2f}">{_format_ratio(r.compression_ratio)}</td>
<td data-v="{r.ssim or 0:.6f}">{_format_metric(r.ssim)}</td>
<td data-v="{r.psnr if r.psnr and r.psnr != float('inf') else 999:.2f}">{_format_metric(r.psnr)}</td>
<td data-v="{r.elapsed_seconds:.3f}">{r.elapsed_seconds:.1f}s</td>
<td>{r.quality_param or ''}</td>
</tr>""")

        html_parts.append("</tbody></table>")

    # JavaScript for sorting
    html_parts.append("""
<script>
function sortTable(th, colIdx) {
  const table = th.closest('table');
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const asc = th.dataset.dir !== 'asc';
  th.dataset.dir = asc ? 'asc' : 'desc';
  rows.sort((a, b) => {
    const av = parseFloat(a.cells[colIdx].dataset.v || a.cells[colIdx].textContent);
    const bv = parseFloat(b.cells[colIdx].dataset.v || b.cells[colIdx].textContent);
    if (isNaN(av) && isNaN(bv)) return a.cells[colIdx].textContent.localeCompare(b.cells[colIdx].textContent) * (asc ? 1 : -1);
    if (isNaN(av)) return 1;
    if (isNaN(bv)) return -1;
    return (av - bv) * (asc ? 1 : -1);
  });
  rows.forEach(r => tbody.appendChild(r));
}
</script>
</body></html>""")

    (output_dir / "comparison.html").write_text("\n".join(html_parts))
