# The `pytorch` container — PyTorch/ROCm 10 engines for Voicebox (VoxCPM2 now).
# OPTIONAL: audio.cpp does the same voices on far fewer resources. PyTorch has its own
# perks (on Radeons it is faster per sentence), and it is the ONLY road to training —
# fine-tuning, LoRA — at the cost of a ~13 GB image and more RAM/VRAM.
#
# Build: through docker-compose.pytorch.yml (see the README).
#
# ⭐ ROCm 10 = AMD's "whl-next" layout: one `torch` (host) + a separate prebuilt kernel
#    package for EACH card (`amd-torch-device-gfxXXXX`, ~0.6-0.7 GB each). Nothing is
#    compiled, but every card listed below costs that much disk.
#    gfx1201 = RX 9070 XT, gfx1100 = RX 7900 XTX. Another card = add its package.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# `gcc` + `libc6-dev` — `torch.compile` (voxcpm warm-up) makes Triton compile a C
# launcher; without a compiler startup dies with "Failed to find C compiler".
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg gcc libc6-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG TORCH=2.14.0+rocm10.1.0
ARG ROCM_INDEX=https://stable.repo.amd.com/rocm/whl-next/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url "$ROCM_INDEX" \
        "torch==$TORCH" torchaudio \
        "amd-torch-device-gfx1201==$TORCH" \
        "amd-torch-device-gfx1100==$TORCH" \
    && pip freeze | grep -i -e '^torch' -e '^amd-torch' -e '^rocm' > /opt/torch-pinned.txt

# ⚠️ Torch from whl-next does NOT ship the aotriton kernels (`torch/lib/aotriton.images`)
# — without them `scaled_dot_product_attention` (FLASH/EFFICIENT) dies with
# `hipErrorInvalidImage`, and MATH is ~9x slower. Images from aotriton's GitHub;
# the version MUST match `libaotriton_v2.so.<version>` in torch (checked below).
ARG AOTRITON=0.14.50tp
RUN L=/usr/local/lib/python3.12/site-packages/torch/lib \
    && test -e "$L/libaotriton_v2.so.${AOTRITON%tp}" \
    && for g in gfx110x gfx120x; do \
        python -c "import urllib.request as u; u.urlretrieve('https://github.com/ROCm/aotriton/releases/download/$AOTRITON/aotriton-$AOTRITON-images-amd-$g.tar.gz', '/tmp/a.tgz')" \
        && tar xzf /tmp/a.tgz -C /tmp && rm /tmp/a.tgz; \
    done \
    && mv /tmp/aotriton/lib/aotriton.images "$L/" && rm -rf /tmp/aotriton

# ⚠️ `--no-deps` on voxcpm itself, the rest with `-c` (pinned torch): a plain
# `pip install voxcpm` would pull torch from PyPI and swap the ROCm one for a CPU one.
# `pillow`, `sortedcontainers` — modelscope (the denoiser) imports them at startup.
RUN pip install --no-cache-dir -c /opt/torch-pinned.txt \
        transformers einops inflect addict wetext "modelscope>=1.22.0" \
        huggingface-hub pydantic tqdm soundfile fastapi uvicorn \
        librosa argbind datasets simplejson pillow sortedcontainers \
    && pip install --no-cache-dir --no-deps voxcpm

# torchcodec — voxcpm lists it, `--no-deps` above skipped it. torchaudio ≥ 2.9 loads audio
# ONLY through it (`torchaudio.load` → the denoiser crashed: "TorchCodec is required").
# The CPU build from PyTorch's index: the PyPI wheel links CUDA (libnvrtc) and does not
# load on ROCm. Decoding runs on the CPU anyway; FFmpeg comes from apt above.
RUN pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cpu \
        "torchcodec==0.17.0+cpu"

COPY manager.py /app/manager.py
COPY workers/ /app/workers/

# Models from /models (mounted); the encoded voice-sample cache is the only thing written.
ENV VOXCPM_WEIGHTS=/models/voxcpm-pt/weights \
    VOXCPM_PREBUILT=/models/voxcpm-pt/prebuilt \
    VOXCPM_PROMPT_CACHE=/cache/prompt-cache

EXPOSE 8080 8081

CMD ["python", "/app/manager.py"]
