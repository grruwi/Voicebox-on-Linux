# The speech engine: audio.cpp with Voicebox's patches (VoxCPM2 + OmniVoice) plus the
# manager (manager.py), which starts and stops the engine process on Voicebox's call.
# Upstream: https://github.com/0xShug0/audio.cpp @ 497c7f0 + voicebox.patch (Apache 2.0).
#
# Two variants (target):
#   vulkan  DEFAULT — runs on any card with Vulkan
#   rocm    OPTIONAL — vulkan + a HIP binary on ROCm 10 (TheRock); Radeons only.
#           Compiled for the GPU_TARGETS below — every extra card = a longer build and
#           ~0.5 GB more in the image, plus a ~2.4 GB ROCm download per card family.
#           Enabled by docker-compose.rocm.yml.

FROM debian:trixie-slim AS source
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ca-certificates curl \
        libvulkan-dev glslc spirv-headers \
 && rm -rf /var/lib/apt/lists/*
ARG AUDIOCPP_COMMIT=497c7f02cd0a941f80b19b7e4f966814b6ce0844
WORKDIR /src
# no submodules — the only one (server-frontends) is optional and off by default
RUN git clone https://github.com/0xShug0/audio.cpp.git . && git checkout ${AUDIOCPP_COMMIT}
COPY voicebox.patch /tmp/voicebox.patch
RUN git apply /tmp/voicebox.patch


FROM source AS build-vulkan
# NATIVE_CPU=ON — CPU kernels for the processor the image is built on.
# audiocpp_gguf = the weight converter (make_weights.py).
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_VULKAN=ON -DENGINE_ENABLE_HIP=OFF \
        -DENGINE_ENABLE_NATIVE_CPU=ON -DENGINE_ENABLE_LLAMAFILE=ON \
        -DENGINE_BUILD_EXAMPLES=OFF -DENGINE_BUILD_TESTS=OFF -DENGINE_BUILD_WARMBENCH=OFF \
        -DAUDIOCPP_DEPLOYMENT_BUILD=ON -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF \
        -DAUDIOCPP_USE_SYSTEM_OPENSSL=OFF \
        -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=voxcpm2,omnivoice \
 && cmake --build build --parallel --target audiocpp_server --target audiocpp_gguf


# ── ROCm (the `rocm` target only) ─────────────────────────────────────────────
FROM source AS build-hip
# Card families from https://rocm.prereleases.amd.com/tarball-multi-arch/ (gfx120X-all = RX 9000,
# gfx110X-all = RX 7000, gfx103X-all = RX 6000) and their GPU_TARGETS.
ARG ROCM_VERSION=10.0.0rc4
ARG ROCM_FAMILIES="gfx120X-all gfx110X-all"
ARG GPU_TARGETS="gfx1201;gfx1200;gfx1100;gfx1101;gfx1102"
ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIPCXX=/opt/rocm/lib/llvm/bin/clang++ PATH=/opt/rocm/bin:$PATH
RUN mkdir -p /opt/rocm && for f in ${ROCM_FAMILIES}; do \
        curl -fsSL "https://rocm.prereleases.amd.com/tarball-multi-arch/therock-dist-linux-${f}-${ROCM_VERSION}.tar.gz" \
        | tar -xz -C /opt/rocm; \
    done
RUN cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_VULKAN=OFF -DENGINE_ENABLE_HIP=ON \
        -DENGINE_ENABLE_NATIVE_CPU=ON -DENGINE_ENABLE_LLAMAFILE=ON \
        -DENGINE_BUILD_EXAMPLES=OFF -DENGINE_BUILD_TESTS=OFF -DENGINE_BUILD_WARMBENCH=OFF \
        -DAUDIOCPP_DEPLOYMENT_BUILD=ON -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF \
        -DAUDIOCPP_USE_SYSTEM_OPENSSL=OFF \
        -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=voxcpm2,omnivoice \
        -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
        "-DGPU_TARGETS=${GPU_TARGETS}" \
        "-DCMAKE_BUILD_RPATH=\$ORIGIN/rocm;\$ORIGIN/rocm/llvm/lib" \
 && cmake --build build-hip --parallel --target audiocpp_server
# Only what the binary uses goes into the image (ldd + rocBLAS/hipBLASLt data + .kpack)
# instead of the whole ~10 GB of ROCm. The layout mirrors ROCm: rocm/ = lib, .kpack next to it.
RUN set -e; D=/out/bin-docker; mkdir -p $D/rocm; \
    cp build-hip/bin/audiocpp_server $D/; \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib ldd $D/audiocpp_server | awk '$3 ~ /^\/opt\/rocm\/lib\// {print $3}' | while read -r so; do \
        rel=${so#/opt/rocm/lib/}; mkdir -p "$D/rocm/$(dirname "$rel")"; cp -L "$so" "$D/rocm/$rel"; \
    done; \
    cp -a /opt/rocm/lib/rocblas /opt/rocm/lib/hipblaslt $D/rocm/; \
    cp -a /opt/rocm/.kpack $D/.kpack


# ── images ────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS image
# Mesa from trixie-backports: the newer RADV exposes cooperative matrix on new Radeons
# (the stable 25.0.7 did matmuls without the matrix units, ~1.7x slower).
RUN echo "deb http://deb.debian.org/debian trixie-backports main" > /etc/apt/sources.list.d/backports.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends libvulkan1 libgomp1 \
 && apt-get install -y --no-install-recommends -t trixie-backports mesa-vulkan-drivers \
 && rm -rf /var/lib/apt/lists/*
# `gguf` — make_weights.py reads the tensor names of the original file
RUN pip install --no-cache-dir gguf
COPY --from=build-vulkan /src/build/bin/audiocpp_server /opt/audiocpp/bin-docker-vulkan/audiocpp_server
COPY --from=build-vulkan /src/build/bin/audiocpp_gguf /opt/audiocpp/bin/audiocpp_gguf
COPY manager.py make_weights.py /app/
# Mesa's shader cache — without it Vulkan compiles its pipelines on every start
ENV XDG_CACHE_HOME=/cache \
    PYTHONUNBUFFERED=1
EXPOSE 17495 17496
CMD ["python", "/app/manager.py"]


# The Voicebox panel (Generation → General → Backend) switches Vulkan / ROCm without a rebuild.
FROM image AS rocm
COPY --from=build-hip /out/bin-docker /opt/audiocpp/bin-docker
# there is no /opt/rocm here — every library must be found next to the binary
RUN ! ldd /opt/audiocpp/bin-docker/audiocpp_server | grep 'not found'


# The last one = the default for a plain `docker build`.
FROM image AS vulkan
