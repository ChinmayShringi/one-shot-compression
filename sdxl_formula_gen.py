"""
Generate a clean beach scene with SDXL and verify determinism.
Two identical runs with seed=42 must produce pixel-identical output.

Uses enable_model_cpu_offload() to fit SDXL in 8 GB VRAM (RTX 2080 SUPER).
Deterministic: torch.use_deterministic_algorithms + fixed CUBLAS workspace.
"""

import os
import time
import hashlib
import numpy as np
from PIL import Image

import torch
from diffusers import StableDiffusionXLPipeline

# ── Determinism flags ───────────────────────────────────────────────
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Config ──────────────────────────────────────────────────────────
PROMPT = (
    "A man in a blue t-shirt sitting on a smooth gray rock at a sandy beach, "
    "a decorated camel in the background, overcast sky, calm sea. "
    "Clean digital art, smooth surfaces, minimal noise, solid colors, "
    "smooth gradients, cel-shaded style"
)
SEED = 42
STEPS = 20
GUIDANCE = 7.5
WIDTH = 1024
HEIGHT = 1024
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

OUT_DIR = os.path.expanduser("~/compress/v3/semantic_results")
OUT_A = os.path.join(OUT_DIR, "sdxl_formula_seed42.png")
OUT_B = os.path.join(OUT_DIR, "sdxl_formula_seed42_v2.png")
LOG = os.path.expanduser("~/compress/sdxl_formula_gen.log")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def generate(pipe, seed: int) -> Image.Image:
    """Generate a single image with a fixed seed (CPU generator for portability)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        width=WIDTH,
        height=HEIGHT,
        generator=generator,
    )
    return result.images[0]


def sha256_of_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Clear log
    with open(LOG, "w") as f:
        f.write("")

    log("Loading SDXL pipeline (cpu-offload for 8GB VRAM)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.enable_model_cpu_offload()
    log("Pipeline loaded.")

    # ── Run 1 ───────────────────────────────────────────────────────
    log(f"Generating run 1 (seed={SEED})...")
    t0 = time.time()
    img_a = generate(pipe, SEED)
    elapsed_a = time.time() - t0
    img_a.save(OUT_A)
    log(f"Run 1 saved to {OUT_A}  ({elapsed_a:.1f}s)")

    # ── Run 2 ───────────────────────────────────────────────────────
    log(f"Generating run 2 (seed={SEED})...")
    t0 = time.time()
    img_b = generate(pipe, SEED)
    elapsed_b = time.time() - t0
    img_b.save(OUT_B)
    log(f"Run 2 saved to {OUT_B}  ({elapsed_b:.1f}s)")

    # ── Compare ─────────────────────────────────────────────────────
    sha_a = sha256_of_file(OUT_A)
    sha_b = sha256_of_file(OUT_B)
    log(f"SHA-256  run1: {sha_a}")
    log(f"SHA-256  run2: {sha_b}")

    arr_a = np.array(img_a)
    arr_b = np.array(img_b)
    pixel_match = np.array_equal(arr_a, arr_b)

    log(f"Pixel-identical: {pixel_match}")
    if not pixel_match:
        diff = np.abs(arr_a.astype(int) - arr_b.astype(int))
        log(f"Max pixel diff: {diff.max()}, Mean: {diff.mean():.4f}")
    else:
        log("DETERMINISM CONFIRMED: both runs produced identical pixels.")

    log("Done.")


if __name__ == "__main__":
    main()
