#!/usr/bin/env bash
# Weights into ./models. Resumes interrupted downloads (curl -C -).
#   ./download-models.sh          minimum: VoxCPM2 + OmniVoice q8 + Whisper large-v3 q5 (~7.4 GB)
#   ./download-models.sh all      + OmniVoice f16 and the other Whispers from the panel (~13 GB)
#   ./download-models.sh pytorch  + VoxCPM2 for the PyTorch engine (~4.7 GB, see the README)
#
# VoxCPM2 orig  4.96 GB — BF16 + AudioVAE F32; the panel picks the precision (q8/f16/f32)
#               when loading. Prebuilt sets load faster and lighter — see the README.
# OmniVoice     q8 1.35 GB / f16 1.64 GB — the official audio.cpp upstream packages
# Whisper       ggml from ggerganov/whisper.cpp (large-v3 q8 is not on HF — greyed out in the panel)
set -euo pipefail
cd "$(dirname "$0")"

get() {  # url target
    mkdir -p "$(dirname "$2")"
    echo "== $2"
    curl -L --fail -C - -o "$2" "$1"
}

AC=https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main
WH=https://huggingface.co/ggerganov/whisper.cpp/resolve/main

get "$AC/VoxCPM2-GGUF/voxcpm2-orig.gguf"         models/audiocpp/weights/voxcpm2-orig.gguf
get "$AC/OmniVoice-GGUF/omnivoice-q8_0.gguf"     models/audiocpp/weights/omnivoice/omnivoice-q8_0.gguf
get "$WH/ggml-large-v3-q5_0.bin"                 models/whisper-cpp/weights/ggml-large-v3-q5_0.bin

case "${1:-}" in
  all)
    get "$AC/OmniVoice-GGUF/omnivoice-f16.gguf"  models/audiocpp/weights/omnivoice/omnivoice-f16.gguf
    for w in large-v3-turbo-q5_0 large-v3-turbo-q8_0 medium-q5_0 medium-q8_0; do
        get "$WH/ggml-$w.bin"                    models/whisper-cpp/weights/ggml-$w.bin
    done ;;
  pytorch)
    # the original repo (BF16 safetensors + configs), as VoxCPM's from_pretrained expects it
    command -v hf >/dev/null || { echo "needs the HF CLI: pip install -U huggingface_hub"; exit 1; }
    hf download openbmb/VoxCPM2 --local-dir models/voxcpm-pt/weights ;;
esac
du -sh models/*/weights
