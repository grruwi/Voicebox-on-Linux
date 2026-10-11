# Speech recognition: whisper.cpp (whisper-server) on Vulkan + the manager (manager.py).
# The model, quantization and GPU/CPU are chosen in the Voicebox panel (Whisper tab).

FROM debian:trixie-slim AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libvulkan-dev glslc spirv-headers \
 && rm -rf /var/lib/apt/lists/*
ARG WHISPER_TAG=v1.9.5
WORKDIR /src
RUN git clone --depth 1 --branch ${WHISPER_TAG} https://github.com/ggml-org/whisper.cpp.git .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON \
        -DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_SDL2=OFF \
 && cmake --build build --parallel --target whisper-server


FROM python:3.12-slim
RUN echo "deb http://deb.debian.org/debian trixie-backports main" > /etc/apt/sources.list.d/backports.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends libvulkan1 libgomp1 \
 && apt-get install -y --no-install-recommends -t trixie-backports mesa-vulkan-drivers \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/bin/whisper-server /whisper/bin/whisper-server
COPY manager.py /app/manager.py
# weights: /whisper/weights (mounted); the selection mirror and the shader cache: /data
ENV XDG_CACHE_HOME=/data/cache \
    PYTHONUNBUFFERED=1
EXPOSE 17480
CMD ["python", "/app/manager.py"]
