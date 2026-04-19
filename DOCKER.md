# semcodec Docker Environment

Deterministic decode environment for the semantic image codec.
Pins the exact software versions proven to produce bit-identical SDXL output.

## Proven Deterministic Stack

| Component        | Version                                   |
|------------------|-------------------------------------------|
| Base OS          | Ubuntu 24.04                              |
| CUDA             | 12.6                                      |
| Python           | 3.12                                      |
| PyTorch          | 2.11.0                                    |
| diffusers        | 0.37.1                                    |
| SDXL model       | stabilityai/stable-diffusion-xl-base-1.0  |
| Tested GPU       | RTX 2080 SUPER (alon)                     |

## Build

```bash
# From the compress/ directory:
docker build -t semcodec .
```

The build downloads the SDXL model (~7 GB) into the image.
The final image is approximately 10 GB. This is intentional:
decode runs fully offline with no network access required.

Build time: 10-20 minutes depending on network speed (model download).

## Run (decode)

```bash
# With docker run:
docker run --gpus all \
    -v "$(pwd)/data:/data" \
    semcodec decode /data/photo.sem -o /data/output.png --device cuda

# With docker compose:
docker compose run --rm semcodec \
    decode /data/photo.sem -o /data/output.png --device cuda
```

Place `.sem` files in `./data/` before running. Output will appear in the same directory.

## Run (info)

```bash
docker run --gpus all \
    -v "$(pwd)/data:/data" \
    semcodec info /data/photo.sem
```

## Determinism Guarantees

The following environment variables are baked into the image:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` -- forces cuBLAS deterministic mode
- `PYTHONHASHSEED=0` -- reproducible Python hash ordering
- `HF_HUB_OFFLINE=1` -- prevents any model re-download
- `TRANSFORMERS_OFFLINE=1` -- prevents online config fetches
- `DIFFUSERS_OFFLINE=1` -- prevents online pipeline fetches

The codec also sets at runtime (in `semcodec.py`):
- `torch.use_deterministic_algorithms(True)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

**Same `.sem` file + same Docker image = identical output, every time.**

Note: determinism is GPU-architecture-dependent. An image decoded on an RTX 2080
will not be bit-identical to one decoded on an RTX 4090, even with all flags set.
For cross-GPU reproducibility, decode on the same GPU model used to validate.

## Encode (not recommended in container)

Encoding requires additional dependencies (rembg, VLM access) that are not
included in this decode-focused image. Encode on the host, decode in the container.

## Troubleshooting

**CUDA out of memory:** The RTX 2080 SUPER has 8 GB VRAM. SDXL with float16
fits within this. If you see OOM, ensure no other GPU processes are running.

**Slow first run:** The model is loaded from disk on each decode. Expect ~30s
load time on a standard SSD. Subsequent runs in the same container are faster
if the OS page cache is warm.

**djxl not found:** The container includes libjxl-tools. If a `.sem` file uses
JXL-compressed crops, they will decompress correctly. WebP crops also work.
