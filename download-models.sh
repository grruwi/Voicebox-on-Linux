#!/usr/bin/env bash
# Weights into ./models. Resumes interrupted downloads (curl -C -). Arguments combine.
#   ./download-models.sh              minimum (~11 GB): VoxCPM2 + its q8/f16 prebuilt set,
#                                     OmniVoice q8, Whisper large-v3 q5
#   ./download-models.sh all          + OmniVoice f16, the other Whispers, all 4 prebuilt sets
#   ./download-models.sh pytorch      + VoxCPM2 for the PyTorch engine + its q8/f16 set (~8.4 GB)
#   ./download-models.sh pytorch-all  + all 4 PyTorch sets (~25 GB)
#
# VoxCPM2 orig  4.96 GB — BF16 + AudioVAE F32; the panel picks the precision when loading
# prebuilt      our sets, cut the way the panel splits the model (BaseLM × acoustic):
#               https://huggingface.co/grruwi/VoxCPM2-audiocpp-prebuilt
#               https://huggingface.co/grruwi/VoxCPM2-pytorch-prebuilt
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
PA=https://huggingface.co/grruwi/VoxCPM2-audiocpp-prebuilt/resolve/main
PP=https://huggingface.co/grruwi/VoxCPM2-pytorch-prebuilt/resolve/main
VX=https://huggingface.co/openbmb/VoxCPM2/resolve/main

get "$AC/VoxCPM2-GGUF/voxcpm2-orig.gguf"         models/audiocpp/weights/voxcpm2-orig.gguf
get "$PA/q8_0-f16/voxcpm2.gguf"                  models/audiocpp/prebuilt/q8_0-f16/voxcpm2.gguf
get "$AC/OmniVoice-GGUF/omnivoice-q8_0.gguf"     models/audiocpp/weights/omnivoice/omnivoice-q8_0.gguf
get "$WH/ggml-large-v3-q5_0.bin"                 models/whisper-cpp/weights/ggml-large-v3-q5_0.bin

for arg in "$@"; do case "$arg" in
  all)
    get "$AC/OmniVoice-GGUF/omnivoice-f16.gguf"  models/audiocpp/weights/omnivoice/omnivoice-f16.gguf
    for w in large-v3-turbo-q5_0 large-v3-turbo-q8_0 medium-q5_0 medium-q8_0; do
        get "$WH/ggml-$w.bin"                    models/whisper-cpp/weights/ggml-$w.bin
    done
    for s in q8_0-f32 f16-f16 f16-f32; do
        get "$PA/$s/voxcpm2.gguf"                models/audiocpp/prebuilt/$s/voxcpm2.gguf
    done ;;
  pytorch|pytorch-all)
    # the original repo: config + tokenizer (the worker reads them) + the BF16 weights
    for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json \
             tokenization_voxcpm2.py model.safetensors audiovae.pth; do
        get "$VX/$f"                             models/voxcpm-pt/weights/$f
    done
    sets=q8-f16; [ "$arg" = pytorch-all ] && sets="q8-f16 q8-f32 f16-f16 f16-f32"
    for s in $sets; do
        get "$PP/$s/model.pt"                    models/voxcpm-pt/prebuilt/$s/model.pt
    done ;;
  *) echo "unknown argument: $arg (all | pytorch | pytorch-all)"; exit 1 ;;
esac; done
du -sh models/*/*
