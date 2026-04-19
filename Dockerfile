# Dockerfile — Deterministic semcodec decode environment (CPU-only)
#
# CPU-only for cross-platform determinism. Proven: same SHA-256
# across repeated decodes on the same x86 machine.
#
# Build:  docker build -t semcodec .
# Decode: docker run -v $(pwd)/data:/data semcodec decode /data/photo.sem -o /data/output.png
# Info:   docker run -v $(pwd)/data:/data semcodec info /data/photo.sem
#
# Image is ~8 GB (includes SDXL model weights). No GPU required.

# ============================================================
# Stage 1: builder — install packages, download model
# ============================================================
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3-pip \
        libjpeg-turbo8 \
        libpng16-16t64 \
        libwebp7 \
        libjxl-tools \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-decode.txt /tmp/requirements-decode.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements-decode.txt

# Pre-download SDXL in float32 (CPU mode — no float16)
ENV HF_HOME=/opt/models
RUN python3 -c "\
from diffusers import StableDiffusionXLPipeline; \
import torch; \
StableDiffusionXLPipeline.from_pretrained( \
    'stabilityai/stable-diffusion-xl-base-1.0', \
    torch_dtype=torch.float32, \
); \
print('SDXL model cached (float32, CPU)')"

# ============================================================
# Stage 2: runtime — lean CPU-only image
# ============================================================
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        libjpeg-turbo8 \
        libpng16-16t64 \
        libwebp7 \
        libjxl-tools \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /opt/models

ENV PATH="/opt/venv/bin:$PATH"
ENV HF_HOME=/opt/models

# ── Determinism: CPU-only, all flags set ─────────────────
ENV CUBLAS_WORKSPACE_CONFIG=":4096:8"
ENV PYTHONHASHSEED=0
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV DIFFUSERS_OFFLINE=1
ENV DO_NOT_TRACK=1
# Force CPU — never use GPU even if available
ENV SEMCODEC_DEVICE=cpu

WORKDIR /workspace
COPY semcodec.py /workspace/semcodec.py

ENTRYPOINT ["python3", "/workspace/semcodec.py"]
CMD ["--help"]
