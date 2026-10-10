#!/usr/bin/env bash
# Wagi do ./modele. Wznawia przerwane pobieranie (curl -C -).
#   ./pobierz-modele.sh            minimum: VoxCPM2 + OmniVoice q8 + Whisper large-v3 q5 (~7,4 GB)
#   ./pobierz-modele.sh wszystko   + OmniVoice f16 i pozostale Whispery z panelu (~13 GB)
#
# VoxCPM2 orig  4,96 GB — BF16 + AudioVAE F32; precyzje (q8/f16/f32) wybiera panel przy ladowaniu
# OmniVoice     q8 1,35 GB / f16 1,64 GB — oficjalne paczki upstreamu audio.cpp
# Whisper       ggml z ggerganov/whisper.cpp (large-v3 q8 nie ma na HF — w panelu zostanie szary)
set -euo pipefail
cd "$(dirname "$0")"

pobierz() {  # url cel
    mkdir -p "$(dirname "$2")"
    echo "== $2"
    curl -L --fail -C - -o "$2" "$1"
}

AC=https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main
WH=https://huggingface.co/ggerganov/whisper.cpp/resolve/main

pobierz "$AC/VoxCPM2-GGUF/voxcpm2-orig.gguf"         modele/audiocpp/wagi/voxcpm2-orig.gguf
pobierz "$AC/OmniVoice-GGUF/omnivoice-q8_0.gguf"     modele/audiocpp/wagi/omnivoice/omnivoice-q8_0.gguf
pobierz "$WH/ggml-large-v3-q5_0.bin"                 modele/whisper-cpp/wagi/ggml-large-v3-q5_0.bin

if [ "${1:-}" = wszystko ]; then
    pobierz "$AC/OmniVoice-GGUF/omnivoice-f16.gguf"  modele/audiocpp/wagi/omnivoice/omnivoice-f16.gguf
    for w in large-v3-turbo-q5_0 large-v3-turbo-q8_0 medium-q5_0 medium-q8_0; do
        pobierz "$WH/ggml-$w.bin"                    modele/whisper-cpp/wagi/ggml-$w.bin
    done
fi
du -sh modele/*/wagi
