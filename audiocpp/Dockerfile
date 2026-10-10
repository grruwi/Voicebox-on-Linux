# Silnik mowy: audio.cpp z latkami Voiceboxa (VoxCPM2 + OmniVoice), backend Vulkan,
# plus zarzadca (zarzadca.py), ktory podnosi i zdejmuje proces na polecenie Voiceboxa.
# Upstream: https://github.com/0xShug0/audio.cpp @ 497c7f0 + latki-voicebox.patch.

FROM debian:trixie-slim AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ca-certificates \
        libvulkan-dev glslc spirv-headers \
 && rm -rf /var/lib/apt/lists/*

ARG AUDIOCPP_COMMIT=497c7f02cd0a941f80b19b7e4f966814b6ce0844
WORKDIR /src
# bez submodulow — jedyny (server-frontends) jest opcjonalny i domyslnie wylaczony
RUN git clone https://github.com/0xShug0/audio.cpp.git . && git checkout ${AUDIOCPP_COMMIT}

COPY latki-voicebox.patch /tmp/latki-voicebox.patch
RUN git apply /tmp/latki-voicebox.patch

# NATIVE_CPU=ON — kernele CPU pod procesor, na ktorym sie buduje.
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_VULKAN=ON -DENGINE_ENABLE_HIP=OFF \
        -DENGINE_ENABLE_NATIVE_CPU=ON -DENGINE_ENABLE_LLAMAFILE=ON \
        -DENGINE_BUILD_EXAMPLES=OFF -DENGINE_BUILD_TESTS=OFF -DENGINE_BUILD_WARMBENCH=OFF \
        -DAUDIOCPP_DEPLOYMENT_BUILD=ON -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF \
        -DAUDIOCPP_USE_SYSTEM_OPENSSL=OFF \
        -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=voxcpm2,omnivoice \
 && cmake --build build --parallel --target audiocpp_server


FROM python:3.12-slim
# Mesa z trixie-backports: nowszy RADV wystawia cooperative matrix na nowych Radeonach
# (stabilna 25.0.7 liczyla matmule bez jednostek macierzowych, ~1,7x wolniej).
RUN echo "deb http://deb.debian.org/debian trixie-backports main" > /etc/apt/sources.list.d/backports.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends libvulkan1 libgomp1 \
 && apt-get install -y --no-install-recommends -t trixie-backports mesa-vulkan-drivers \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/bin/audiocpp_server /opt/audiocpp/bin-docker-vulkan/audiocpp_server
COPY zarzadca.py /app/zarzadca.py
# cache shaderow Mesy — bez niego Vulkan kompiluje pipeline'y przy kazdym starcie
ENV AUDIOCPP_GNIAZDA=/opt/audiocpp \
    XDG_CACHE_HOME=/cache \
    PYTHONUNBUFFERED=1
EXPOSE 17495 17496
CMD ["python", "/app/zarzadca.py"]
