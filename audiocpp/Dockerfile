# Silnik mowy: audio.cpp z latkami Voiceboxa (VoxCPM2 + OmniVoice) plus zarzadca
# (zarzadca.py), ktory podnosi i zdejmuje proces na polecenie Voiceboxa.
# Upstream: https://github.com/0xShug0/audio.cpp @ 497c7f0 + latki-voicebox.patch (Apache 2.0).
#
# Dwa warianty (target):
#   vulkan  DOMYSLNY — dziala na kazdej karcie z Vulkanem
#   rocm    OPCJONALNY — vulkan + binarka HIP na ROCm 10 (TheRock); tylko Radeony,
#           build pobiera ~2,4 GB na rodzine kart. Wlacza sie docker-compose.rocm.yml.

FROM debian:trixie-slim AS zrodla
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ca-certificates curl \
        libvulkan-dev glslc spirv-headers \
 && rm -rf /var/lib/apt/lists/*
ARG AUDIOCPP_COMMIT=497c7f02cd0a941f80b19b7e4f966814b6ce0844
WORKDIR /src
# bez submodulow — jedyny (server-frontends) jest opcjonalny i domyslnie wylaczony
RUN git clone https://github.com/0xShug0/audio.cpp.git . && git checkout ${AUDIOCPP_COMMIT}
COPY latki-voicebox.patch /tmp/latki-voicebox.patch
RUN git apply /tmp/latki-voicebox.patch


FROM zrodla AS build-vulkan
# NATIVE_CPU=ON — kernele CPU pod procesor, na ktorym sie buduje.
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_VULKAN=ON -DENGINE_ENABLE_HIP=OFF \
        -DENGINE_ENABLE_NATIVE_CPU=ON -DENGINE_ENABLE_LLAMAFILE=ON \
        -DENGINE_BUILD_EXAMPLES=OFF -DENGINE_BUILD_TESTS=OFF -DENGINE_BUILD_WARMBENCH=OFF \
        -DAUDIOCPP_DEPLOYMENT_BUILD=ON -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF \
        -DAUDIOCPP_USE_SYSTEM_OPENSSL=OFF \
        -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=voxcpm2,omnivoice \
 && cmake --build build --parallel --target audiocpp_server


# ── ROCm (tylko target `rocm`) ───────────────────────────────────────────────
FROM zrodla AS build-hip
# Rodziny kart z https://rocm.prereleases.amd.com/tarball-multi-arch/ (gfx120X-all = RX 9000,
# gfx110X-all = RX 7000, gfx103X-all = RX 6000) i odpowiadajace im GPU_TARGETS.
ARG ROCM_WERSJA=10.0.0rc4
ARG ROCM_RODZINY="gfx120X-all gfx110X-all"
ARG GPU_TARGETS="gfx1201;gfx1200;gfx1100;gfx1101;gfx1102"
ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIPCXX=/opt/rocm/lib/llvm/bin/clang++ PATH=/opt/rocm/bin:$PATH
RUN mkdir -p /opt/rocm && for r in ${ROCM_RODZINY}; do \
        curl -fsSL "https://rocm.prereleases.amd.com/tarball-multi-arch/therock-dist-linux-${r}-${ROCM_WERSJA}.tar.gz" \
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
# Do obrazu idzie tylko to, czego binarka uzywa (ldd + dane rocBLAS/hipBLASLt + .kpack),
# zamiast calych ~10 GB ROCm. Uklad jak w ROCm: rocm/ = lib, .kpack obok.
RUN set -e; G=/gniazdo/bin-docker; mkdir -p $G/rocm; \
    cp build-hip/bin/audiocpp_server $G/; \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib ldd $G/audiocpp_server | awk '$3 ~ /^\/opt\/rocm\/lib\// {print $3}' | while read -r so; do \
        rel=${so#/opt/rocm/lib/}; mkdir -p "$G/rocm/$(dirname "$rel")"; cp -L "$so" "$G/rocm/$rel"; \
    done; \
    cp -a /opt/rocm/lib/rocblas /opt/rocm/lib/hipblaslt $G/rocm/; \
    cp -a /opt/rocm/.kpack $G/.kpack


# ── obrazy ───────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS obraz
# Mesa z trixie-backports: nowszy RADV wystawia cooperative matrix na nowych Radeonach
# (stabilna 25.0.7 liczyla matmule bez jednostek macierzowych, ~1,7x wolniej).
RUN echo "deb http://deb.debian.org/debian trixie-backports main" > /etc/apt/sources.list.d/backports.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends libvulkan1 libgomp1 \
 && apt-get install -y --no-install-recommends -t trixie-backports mesa-vulkan-drivers \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build-vulkan /src/build/bin/audiocpp_server /opt/audiocpp/bin-docker-vulkan/audiocpp_server
COPY zarzadca.py /app/zarzadca.py
# cache shaderow Mesy — bez niego Vulkan kompiluje pipeline'y przy kazdym starcie
ENV AUDIOCPP_GNIAZDA=/opt/audiocpp \
    XDG_CACHE_HOME=/cache \
    PYTHONUNBUFFERED=1
EXPOSE 17495 17496
CMD ["python", "/app/zarzadca.py"]


# Panel Voiceboxa (zakladka Ogolne → Backend) przelacza Vulkan / ROCm bez przebudowy.
FROM obraz AS rocm
COPY --from=build-hip /gniazdo/bin-docker /opt/audiocpp/bin-docker
# tu nie ma /opt/rocm — kazda biblioteka musi sie znalezc w gniezdzie
RUN ! ldd /opt/audiocpp/bin-docker/audiocpp_server | grep 'not found'


# Ostatni = domyslny przy golym `docker build`.
FROM obraz AS vulkan
