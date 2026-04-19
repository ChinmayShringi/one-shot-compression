"""Test CPU-CPU determinism: decode same .sem file twice, compare."""
import os, hashlib
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
from PIL import Image
from semcodec import decode

print("=== CPU DETERMINISM TEST ===")
print("Decoding output/photo.sem twice on CPU...\n")

print("--- Run 1 ---")
decode("output/photo.sem", "output/cpu_run1.png", "cpu")

print("\n--- Run 2 ---")
decode("output/photo.sem", "output/cpu_run2.png", "cpu")

# Compare
r1 = np.array(Image.open("output/cpu_run1.png").convert("RGB"))
r2 = np.array(Image.open("output/cpu_run2.png").convert("RGB"))
h1 = hashlib.sha256(r1.tobytes()).hexdigest()
h2 = hashlib.sha256(r2.tobytes()).hexdigest()

print(f"\n=== RESULTS ===")
print(f"Run1 SHA-256: {h1}")
print(f"Run2 SHA-256: {h2}")
print(f"IDENTICAL: {h1 == h2}")

if h1 != h2:
    diff = np.abs(r1.astype(int) - r2.astype(int))
    print(f"Max diff: {diff.max()}, Mean: {diff.mean():.3f}")
    print(f"Exact: {100*np.sum(diff==0)/diff.size:.1f}%")
