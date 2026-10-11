"""Voice prompt cache — the PyTorch twin of audio.cpp patch 8 (`prompt_disk_cache.h`).

Every request carries the profile sample as base64 and upstream re-encodes it
each time in `build_prompt_cache`: librosa resample + AudioVAE encode. The result
is a few small CPU tensors that depend only on the samples and the transcript,
so they are kept in RAM and on disk under a content hash. A profile is encoded
once and survives container restarts.

The key hashes DECODED samples, like audio.cpp — not file bytes. Voicebox rewrites
a multi-sample profile's combined WAV before every request, and libsndfile puts
a timestamp into the PEAK chunk of float WAVs: same audio, new bytes each time.

⚠️ The key does not know the AudioVAE code or weights. Every weight set shares
the same F32 AudioVAE; if the encoder ever changes, clear the directory.
"""

import hashlib
import time
from pathlib import Path

import soundfile as sf
import torch

_memory: dict = {}


def _key(prompt_text, prompt_wav_path, reference_wav_path, trim_silence_vad) -> str:
    h = hashlib.sha256()
    for path in (reference_wav_path, prompt_wav_path):
        if path is None:
            h.update(b"0")
            continue
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        h.update(b"1" + f"{rate}:{samples.shape[1]}:".encode() + samples.tobytes())
    h.update(b"0" if prompt_text is None else b"1" + prompt_text.encode("utf-8"))
    h.update(b"1" if trim_silence_vad else b"0")
    return h.hexdigest()


def enable_prompt_cache(tm, directory: Path) -> None:
    """Wrap `tm.build_prompt_cache` (instance attribute, the class stays upstream)."""
    directory.mkdir(parents=True, exist_ok=True)
    upstream = tm.build_prompt_cache

    def build_prompt_cache(prompt_text=None, prompt_wav_path=None, reference_wav_path=None,
                           trim_silence_vad=False):
        start = time.perf_counter()
        key = _key(prompt_text, prompt_wav_path, reference_wav_path, trim_silence_vad)
        cache, source = _memory.get(key), "RAM"
        if cache is None:
            path = directory / f"{key}.pt"
            if path.exists():
                cache, source = torch.load(path, weights_only=True), "disk"
            else:
                cache, source = upstream(prompt_text=prompt_text, prompt_wav_path=prompt_wav_path,
                                         reference_wav_path=reference_wav_path,
                                         trim_silence_vad=trim_silence_vad), "encoded"
                tmp = path.with_suffix(".tmp")
                torch.save(cache, tmp)
                tmp.replace(path)
            _memory[key] = cache
        print(f"[prompt] {source} {key[:12]} {(time.perf_counter() - start) * 1e3:.0f} ms", flush=True)
        # Upstream only reads the tensors (torch.cat); a fresh dict keeps ours untouched.
        return dict(cache)

    tm.build_prompt_cache = build_prompt_cache
