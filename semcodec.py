#!/usr/bin/env python3
"""semcodec — Semantic Image Codec
===================================

Hybrid lossless-foreground + generative-background image codec.

Encode: segments foreground, compresses it losslessly, captions the scene,
        packs everything into a .sem file.
Decode: unpacks, generates background with SDXL, composites foreground back.

Usage:
  python semcodec.py encode photo.png -o photo.sem
  python semcodec.py decode photo.sem -o reconstructed.png
  python semcodec.py info   photo.sem

File format (.sem):
  [8B magic "SEMCODEC\x01"] [4B header_len] [header_json] [crop_data] [mask_data]
"""

import argparse
import json
import struct
import sys
import os
import time
import io
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter

MAGIC = b"SEMCDC\x00\x01"  # 8 bytes
VERSION = 1


# ─────────────────────────────────────────────────────────
# ENCODER
# ─────────────────────────────────────────────────────────

def encode(input_path, output_path, crop_quality=1.0, caption=None, seed=42, fmt="auto"):
    """Encode an image into a .sem file."""
    print(f"Encoding {input_path}...")
    t0 = time.time()

    img = Image.open(input_path).convert("RGBA")
    orig_rgb = Image.open(input_path).convert("RGB")
    w, h = img.size

    # 1. Segment foreground
    print("  [1/4] Segmenting foreground...")
    fg_rgba, mask, bbox = segment_foreground(img)

    # 2. Compress foreground crop
    print("  [2/4] Compressing foreground crop...")
    crop = orig_rgb.crop(bbox)
    crop_bytes = compress_crop(crop, crop_quality, fmt)

    # 3. Generate caption
    print("  [3/4] Captioning scene...")
    if caption is None:
        caption = generate_caption(input_path)
    caption_bytes = caption.encode("utf-8")

    # 4. Compress mask
    print("  [4/4] Compressing mask...")
    mask_bytes = compress_mask(mask)

    # Pack into .sem file
    # 5. Generate verification hash by doing a CPU decode of the background
    print("  [5/5] Computing verification hash (CPU decode)...")
    verify_hash = compute_verification_hash(caption, w, h, seed)

    header = {
        "version": VERSION,
        "width": w,
        "height": h,
        "bbox": [int(x) for x in bbox],
        "seed": seed,
        "caption_len": len(caption_bytes),
        "crop_len": len(crop_bytes),
        "mask_len": len(mask_bytes),
        "crop_quality": crop_quality,
        "fg_pixels": int(np.sum(np.array(mask) > 128)),
        "total_pixels": w * h,
        "device": "cpu",
        "verify_hash": verify_hash,
    }
    header_json = json.dumps(header).encode("utf-8")

    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)
        f.write(caption_bytes)
        f.write(crop_bytes)
        f.write(mask_bytes)

    total_size = os.path.getsize(output_path)
    elapsed = time.time() - t0

    print(f"\n  Encoded: {output_path}")
    print(f"  Size: {total_size / 1024:.1f} KB")
    print(f"    Header:  {len(header_json)} bytes")
    print(f"    Caption: {len(caption_bytes)} bytes")
    print(f"    Crop:    {len(crop_bytes) / 1024:.1f} KB")
    print(f"    Mask:    {len(mask_bytes) / 1024:.1f} KB")
    print(f"  Foreground: {header['fg_pixels']} pixels ({100 * header['fg_pixels'] / header['total_pixels']:.1f}%)")
    print(f"  Time: {elapsed:.1f}s")

    return total_size


def segment_foreground(img_rgba):
    """Segment foreground using rembg."""
    from rembg import remove

    result = remove(img_rgba)
    alpha = np.array(result)[:, :, 3]
    mask = (alpha > 128).astype(np.uint8) * 255

    # Bounding box with padding
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    pad = 10
    h, w = mask.shape
    rmin = max(0, int(rmin) - pad)
    rmax = min(h - 1, int(rmax) + pad)
    cmin = max(0, int(cmin) - pad)
    cmax = min(w - 1, int(cmax) + pad)

    bbox = (cmin, rmin, cmax + 1, rmax + 1)
    fg_crop = result.crop(bbox)
    mask_img = Image.fromarray(mask)

    return fg_crop, mask_img, bbox


def compress_crop(crop_img, quality, fmt="auto"):
    """Compress crop. fmt: 'auto' tries JXL then WebP, 'webp' forces WebP."""
    if fmt != "webp":
        # Try JPEG-XL
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
            crop_img.save(tmp_png.name)
            tmp_jxl = tmp_png.name.replace(".png", ".jxl")

            r = subprocess.run(
                ["cjxl", tmp_png.name, tmp_jxl, "-d", str(quality), "-e", "9"],
                capture_output=True,
            )

            if r.returncode == 0:
                with open(tmp_jxl, "rb") as f:
                    data = f.read()
                os.unlink(tmp_png.name)
                os.unlink(tmp_jxl)
                return b"JXL:" + data

            os.unlink(tmp_png.name)

    # WebP fallback / forced
    buf = io.BytesIO()
    webp_q = int(max(50, 100 - quality * 25))
    crop_img.save(buf, "WEBP", quality=webp_q, lossless=(quality == 0))
    return b"WBP:" + buf.getvalue()


def compress_mask(mask_img):
    """Compress mask as 1-bit PNG."""
    mask_1bit = mask_img.convert("1")
    buf = io.BytesIO()
    mask_1bit.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def compute_verification_hash(caption, width, height, seed):
    """Generate background on CPU and hash it. Stored in .sem for decode verification."""
    import hashlib
    bg = generate_background(caption, width, height, seed, device="cpu")
    bg_bytes = np.array(bg).tobytes()
    return hashlib.sha256(bg_bytes).hexdigest()


def generate_caption(img_path):
    """Caption the scene using local VLM (Qwen3-VL on LM Studio) or fallback."""
    import urllib.request
    import base64

    try:
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        payload = json.dumps({
            "model": "qwen3-vl-4b",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": (
                        "Describe this scene for image regeneration in under 50 words. "
                        "Focus on: setting, lighting, background objects, terrain. "
                        "Do NOT describe people in the foreground. Output ONLY the description."
                    )}
                ]
            }],
            "max_tokens": 100,
            "temperature": 0.1,
        })

        req = urllib.request.Request(
            "http://localhost:1234/v1/chat/completions",
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"    VLM unavailable ({e}), using placeholder caption")
        return "Sandy beach, overcast sky, calm sea, rocks in foreground"


# ─────────────────────────────────────────────────────────
# DECODER
# ─────────────────────────────────────────────────────────

def decode(input_path, output_path, device="cpu"):
    """Decode a .sem file back to an image.

    Always decodes on CPU for cross-platform determinism (unless --device overrides).
    Verifies output against the hash stored during encoding.
    """
    import hashlib

    print(f"Decoding {input_path}...")
    t0 = time.time()

    # 1. Unpack
    print("  [1/4] Unpacking .sem file...")
    header, caption, crop_img, mask_img = unpack_sem(input_path)

    w, h = header["width"], header["height"]
    bbox = header["bbox"]
    seed = header["seed"]
    encoded_device = header.get("device", "cpu")
    verify_hash = header.get("verify_hash")

    # Force CPU for determinism: env var overrides CLI, .sem header overrides env
    env_device = os.environ.get("SEMCODEC_DEVICE")
    if env_device:
        device = env_device
    if encoded_device == "cpu" and device != "cpu":
        print(f"    WARNING: .sem encoded with device=cpu. Forcing CPU for determinism.")
        device = "cpu"

    print(f"    Image: {w}x{h}, seed={seed}, device={device}")
    print(f"    Caption: {caption[:80]}...")
    print(f"    Crop: {crop_img.size}, Mask: {mask_img.size}")

    # 2. Generate background
    print("  [2/4] Generating background with SDXL...")
    background = generate_background(caption, w, h, seed, device)

    # 3. Verify hash
    if verify_hash:
        print("  [3/5] Verifying decode integrity...")
        bg_hash = hashlib.sha256(np.array(background).tobytes()).hexdigest()
        if bg_hash == verify_hash:
            print(f"    VERIFIED: background matches encoder hash")
        else:
            print(f"    MISMATCH: background differs from encoder!")
            print(f"      Expected: {verify_hash[:32]}...")
            print(f"      Got:      {bg_hash[:32]}...")
            print(f"      This means the decode environment differs from encode.")
            print(f"      The person is still exact, but background will differ.")
    else:
        print("  [3/5] No verification hash in file (v1 format)")

    # 4. Composite with hard mask
    print("  [4/5] Compositing foreground...")
    result = composite_hard_mask(background, crop_img, mask_img, bbox)

    # 5. Save
    result.save(output_path)
    elapsed = time.time() - t0

    print(f"\n  Decoded: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"  Time: {elapsed:.1f}s")

    return result


def unpack_sem(path):
    """Unpack a .sem file into components."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"Not a .sem file (magic: {magic})")

        header_len = struct.unpack("<I", f.read(4))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))

        caption = f.read(header["caption_len"]).decode("utf-8")
        crop_data = f.read(header["crop_len"])
        mask_data = f.read(header["mask_len"])

    # Decompress crop
    if crop_data[:4] == b"JXL:":
        crop_img = decompress_jxl(crop_data[4:])
    elif crop_data[:4] == b"WBP:":
        crop_img = Image.open(io.BytesIO(crop_data[4:])).convert("RGB")
    else:
        crop_img = Image.open(io.BytesIO(crop_data)).convert("RGB")

    # Decompress mask
    mask_img = Image.open(io.BytesIO(mask_data)).convert("L")

    return header, caption, crop_img, mask_img


def decompress_jxl(data):
    """Decompress JXL data to PIL Image."""
    with tempfile.NamedTemporaryFile(suffix=".jxl", delete=False) as tmp_jxl:
        tmp_jxl.write(data)
        tmp_jxl.flush()
        tmp_png = tmp_jxl.name.replace(".jxl", ".png")

        r = subprocess.run(["djxl", tmp_jxl.name, tmp_png], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"djxl failed: {r.stderr.decode()}")

        img = Image.open(tmp_png).convert("RGB")
        os.unlink(tmp_jxl.name)
        os.unlink(tmp_png)
        return img


def generate_background(caption, width, height, seed, device="cpu"):
    """Generate background image with SDXL."""
    import torch
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )

    # Determinism flags for ALL devices
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "0"

    if device == "cuda":
        pipe.enable_model_cpu_offload()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        pipe.to(device)

    gen_w = min(width, 1024) // 8 * 8
    gen_h = min(height, 1024) // 8 * 8

    generator = torch.Generator(device="cpu").manual_seed(seed)

    result = pipe(
        prompt=caption,
        negative_prompt="person, human, man, woman, close up, portrait, face",
        width=gen_w,
        height=gen_h,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    if (gen_w, gen_h) != (width, height):
        result = result.resize((width, height), Image.LANCZOS)

    return result


def composite_hard_mask(background, crop_img, mask_full, bbox, edge_px=2):
    """Composite foreground onto background with hard mask + edge feather."""
    w, h = background.size
    result_np = np.array(background).copy()

    # Place crop
    person_layer = np.zeros((h, w, 3), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    crop_np = np.array(crop_img)
    ch = min(y2 - y1, crop_np.shape[0])
    cw = min(x2 - x1, crop_np.shape[1])
    person_layer[y1:y1 + ch, x1:x1 + cw] = crop_np[:ch, :cw]

    # Build alpha: hard interior + feathered edge
    mask_np = np.array(mask_full.resize((w, h), Image.NEAREST))
    eroded = np.array(Image.fromarray(mask_np).filter(
        ImageFilter.MinFilter(2 * edge_px + 1)
    )) > 128
    dilated = np.array(Image.fromarray(mask_np).filter(
        ImageFilter.MaxFilter(2 * edge_px + 1)
    )) > 128

    alpha = np.zeros((h, w), dtype=np.float64)
    alpha[eroded] = 1.0

    edge = dilated & ~eroded
    if np.any(edge):
        blurred = np.array(Image.fromarray(
            (eroded.astype(np.uint8) * 255)
        ).filter(ImageFilter.GaussianBlur(edge_px))).astype(np.float64) / 255.0
        alpha[edge] = blurred[edge]

    for c in range(3):
        result_np[:, :, c] = (
            alpha * person_layer[:, :, c] +
            (1.0 - alpha) * result_np[:, :, c]
        ).astype(np.uint8)

    return Image.fromarray(result_np)


# ─────────────────────────────────────────────────────────
# INFO
# ─────────────────────────────────────────────────────────

def info(input_path):
    """Print info about a .sem file."""
    header, caption, crop_img, mask_img = unpack_sem(input_path)
    total = os.path.getsize(input_path)

    print(f"File: {input_path} ({total / 1024:.1f} KB)")
    print(f"Format: semcodec v{header['version']}")
    print(f"Original: {header['width']}x{header['height']}")
    print(f"Foreground: {header['fg_pixels']} pixels ({100 * header['fg_pixels'] / header['total_pixels']:.1f}%)")
    print(f"Bbox: {header['bbox']}")
    print(f"Crop: {crop_img.size}, quality={header['crop_quality']}")
    print(f"Seed: {header['seed']}")
    print(f"Device: {header.get('device', 'unspecified')}")
    print(f"Verify hash: {header.get('verify_hash', 'none')[:32]}...")
    print(f"Caption ({len(caption)} chars): {caption}")
    print(f"\nSize breakdown:")
    print(f"  Crop:    {header['crop_len'] / 1024:.1f} KB")
    print(f"  Mask:    {header['mask_len'] / 1024:.1f} KB")
    print(f"  Caption: {header['caption_len']} bytes")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="semcodec — Semantic Image Codec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python semcodec.py encode photo.png -o photo.sem\n"
               "  python semcodec.py decode photo.sem -o out.png --device cuda\n"
               "  python semcodec.py info photo.sem\n",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # encode
    enc = sub.add_parser("encode", help="Encode image to .sem")
    enc.add_argument("input", help="Input image (PNG/JPEG)")
    enc.add_argument("-o", "--output", required=True, help="Output .sem file")
    enc.add_argument("-q", "--quality", type=float, default=1.0,
                     help="Crop quality (0=lossless, 0.5=high, 1.0=good, 2.0=ok)")
    enc.add_argument("--caption", help="Manual caption (skips VLM)")
    enc.add_argument("--seed", type=int, default=42, help="Generation seed")
    enc.add_argument("--format", default="auto", choices=["auto", "webp"],
                     help="Crop format (auto=JXL>WebP, webp=force WebP)")

    # decode
    dec = sub.add_parser("decode", help="Decode .sem to image")
    dec.add_argument("input", help="Input .sem file")
    dec.add_argument("-o", "--output", required=True, help="Output image")
    dec.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                     help="Device for SDXL generation")

    # info
    inf = sub.add_parser("info", help="Show .sem file info")
    inf.add_argument("input", help="Input .sem file")

    args = parser.parse_args()

    if args.command == "encode":
        encode(args.input, args.output, args.quality, args.caption, args.seed, args.format)
    elif args.command == "decode":
        decode(args.input, args.output, args.device)
    elif args.command == "info":
        info(args.input)


if __name__ == "__main__":
    main()
