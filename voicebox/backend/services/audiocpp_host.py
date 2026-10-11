"""AudioCPP Host — the client of the `audiocpp-backend` container (the voice engine outside Voicebox).

⭐ SINCE 10.10 (grruwi's call): `audiocpp_server` runs in a SEPARATE CONTAINER
`audiocpp` (`audiocpp/` in this repo), so Voicebox and the models
move separately. The BRAIN stays here: the model entry is built from the panel. The
container's manager starts and stops the process (an Engine Host, see `engine_host.py`:
`/state`, `/load`, `/unload`), and we talk to the engine itself (`AUDIOCPP_ENGINE`)
directly: `/v1/audio/speech`, `/v1/models`.

⭐ ONE MODEL AT A TIME: the process always has exactly one entry in `models` (the manager
enforces it), so loading another one = restarting the process and VRAM surely comes back.
Unloading = the manager kills the process; the container keeps running.

The engine comes up with Voicebox with a LAZY Vox (`lazy`, empty VRAM). The weight
precision from the panel sits in `session_options`, and those are in the process config —
so "load" or a precision change = restarting the process with `lazy: false` (the model
loads BEFORE the port opens).
⛔ `ui_management` (`/v1/models/load` on the fly) needs a build with
`AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON` (BoringSSL + an installer from the network) — we don't have it.

Whisper is NOT here — a separate `whisper` container (`services/whisper_host.py`).

Binaries: baked into the `audiocpp` image (`audiocpp/Dockerfile`, targets vulkan | rocm)
(mounted into the engine container). The MANAGER checks they exist, not Voicebox — the
Voicebox container does not need to see them (public repo: the binary lives only in the
engine image).
"""

import json
import logging
import os
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from . import engine_host

logger = logging.getLogger(__name__)

HOST = "audiocpp"
_lock = threading.Lock()

VOX_ID = "voxcpm2"
OMNI_ID = "omnivoice"

# ⭐ Profile samples are up to ~30 s (10.10: the longest is 29.4 s). The default AudioVAE
# encoder capacity ends at ~15 s and the request dies with
# "AudioVAE encoder sample capacity exceeded". 512000 samples at 16 kHz = 32 s,
# the encoder graph gets 2048 MB for it. Measured on Kleks (29 s).
_ENCODER_CAPACITY = {
    "voxcpm2.audiovae_encoder_sample_capacity": 512000,
    "voxcpm2.audiovae_encoder_graph_context_mb": 2048,
}


# ⭐ Sample cache ON DISK (our audio.cpp patch, `prompt_disk_cache.h`): AudioVAE features
# + LocEnc embeddings, ~2.4 s of compute on every profile change (RAM keeps 1 profile and
# dies with the process). The key is the sample content, the transcript and the weight
# type; the directory separates weight files (name, size, mtime).
_SAMPLE_CACHE = Path("/app/data/cache/audiocpp-samples")


def _cache_dir() -> str:
    tag = "-".join(
        f"{p.stem}-{p.stat().st_size}-{int(p.stat().st_mtime)}"
        for p in sorted(weights().glob("*.gguf"))
    ) or "no-weights"
    return str(_SAMPLE_CACHE / tag)


def _omni_cache_dir(path: Path) -> str:
    """OmniVoice's sample cache (the same patch): the reference audio tokens, a directory
    per weight file. Its own directory — Vox and OmniVoice encode a sample differently."""
    try:
        st = path.stat()
        tag = f"{path.stem}-{st.st_size}-{int(st.st_mtime)}"
    except OSError:
        tag = path.stem
    return str(_SAMPLE_CACHE / tag)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def address() -> tuple:
    u = urlparse(_env("AUDIOCPP_ENGINE", "http://audiocpp-backend:17496"))
    return (u.hostname or "audiocpp-backend", u.port or 17496)


def _manager(path: str, data: Optional[dict] = None, timeout: float = 30) -> dict:
    """The container comes up together with Voicebox, so a refused connection right
    after start = wait (up to ~20 s), don't fail."""
    try:
        return engine_host.request(HOST, path, data, timeout=timeout, wait_s=20)
    except engine_host.EngineHostError as e:
        raise RuntimeError(str(e)) from e


def _url(path: str) -> str:
    h, p = address()
    return f"http://{h}:{p}{path}"


def process_backend() -> Optional[str]:
    """hip | vulkan — which binary the engine ACTUALLY runs on (per the manager)."""
    current = engine_host.state(HOST)
    return (current or {}).get("backend")


def _same_card() -> bool:
    """The process sees exactly the card chosen in the panel (`gpu_card`)."""
    from . import gpu_card

    config = (engine_host.state(HOST) or {}).get("config") or {}
    return (config.get("env") or {}) == gpu_card.engine_env()


def weights() -> Path:
    """The VoxCPM2 GGUF directory — audio.cpp takes a directory, not a file."""
    from ..config import VOICE_MODELS

    return Path(_env("AUDIOCPP_WEIGHTS", str(VOICE_MODELS / "audiocpp" / "weights")))


def prebuilt() -> Path:
    """Prebuilt weight files, one directory per precision set (`<base_lm>-<acoustic>/`)."""
    return Path(_env("AUDIOCPP_PREBUILT", str(weights().parent / "prebuilt")))


def omni_weights(dtype: str) -> Path:
    """The ready OmniVoice GGUF from upstream (`omnivoice-q8_0.gguf` / `omnivoice-f16.gguf`) —
    config, text tokenizer and audio tokenizer sit inside, the engine takes the file alone
    (it unpacks the sidecars into the container's /tmp)."""
    return weights() / "omnivoice" / f"omnivoice-{dtype}.gguf"


def _weights_dir(options: dict) -> Path:
    """The PREBUILT file for this generator precision, if it is there — otherwise the original.

    ⭐ `prebuilt/<type>/` is made by `audiocpp/make_weights.py` (see the README): tensors
    already in the types the engine would produce anyway, so loading is just a read +
    upload (10.10: 2.2 s instead of 7.3; from a cold disk 4.2 instead of ~23). Without a
    prebuilt file the engine converts the BF16 original on every start — it works, just
    slowly. The sample cache stays keyed by the original: the prebuilt file gives the
    same weights bit for bit, and the entry key holds the weight type anyway.
    """
    name = f'{options["voxcpm2.base_lm_weight_type"]}-{options["voxcpm2.weight_type"]}'
    ready = prebuilt() / name
    if (ready / "voxcpm2.gguf").is_file():
        return ready
    logger.info("audiocpp: no prebuilt file %s — the engine converts the original while loading", ready)
    return weights()


def up() -> bool:
    """The engine port accepts connections. A host that is not there answers in 0.5 s,
    not after Docker's ~4 s DNS give-up (see `engine_host._resolves`)."""
    if not engine_host._resolves(_url("")):
        return False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(address()) == 0
    except OSError:
        return False


def ask(path: str, data: Optional[dict] = None, timeout: float = 120) -> dict:
    """GET (no data) or POST JSON to the engine. A server error → RuntimeError."""
    req = urllib.request.Request(
        _url(path),
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"audiocpp {path}: {e.read().decode('utf-8', 'replace')[:300]}") from e


def _vox_entry(options: dict, lazy: bool) -> dict:
    """The Vox entry of the engine config — precision from the panel, sample cache, encoder capacity."""
    return {
        "id": VOX_ID,
        "family": "voxcpm2",
        "task": "tts",
        "mode": "streaming",
        "path": str(_weights_dir(options)),
        "lazy": lazy,
        "session_options": {
            **options,
            "voxcpm2.prompt_disk_cache_dir": _cache_dir(),
            **_ENCODER_CAPACITY,
        },
    }


def _start(entry: dict, backend: Optional[str] = None, force: bool = False) -> None:
    """The manager starts the engine with THIS ONE model (the old process goes). `lazy: false`
    = weights load before the port opens, so we wait longer (prebuilt Vox ~2 s, from a cold
    disk ~4 s; the original with conversion ~7 / ~23 s)."""
    from . import audiocpp_settings, gpu_card

    backend = backend or audiocpp_settings.selected_backend()
    timeout = 30 if entry.get("lazy") else 300
    reply = _manager("/load", {
        "backend": backend, "model": entry,
        "flags": audiocpp_settings.extra_flags(), "env": gpu_card.engine_env(),
        "timeout_s": timeout, "force": force,
    }, timeout=timeout + 30)
    options = entry.get("session_options", {})
    logger.info("audio.cpp engine up (%s, %s, %s) in %s ms — %s",
                entry["id"], "lazy" if entry.get("lazy") else "loaded", backend,
                reply.get("ms"), {k: v for k, v in options.items() if k.endswith("weight_type")})


def ensure_running() -> None:
    """The engine is up — if not, start it with a LAZY Vox (precision from the panel)."""
    with _lock:
        if up():
            return
        from . import audiocpp_settings

        _start(_vox_entry(audiocpp_settings.weight_options(*audiocpp_settings.selected_precision()), lazy=True))


def state() -> dict:
    """{'vox': {'session_options'}|None, 'omni': {'variant', 'flash'}|None} — LOADED only.
    One model at a time, so at most one of vox/omni is not None."""
    from . import audiocpp_settings

    empty = {"vox": None, "omni": None}
    if not up():
        return empty
    try:
        models = ask("/v1/models?include_session_options=true", timeout=3).get("data", [])
    except Exception as e:
        logger.debug("audiocpp: /v1/models not answering (%s)", e)
        return empty
    result = dict(empty)
    for m in models:
        if not m.get("loaded"):
            continue
        if m.get("id") == VOX_ID:
            result["vox"] = {"session_options": m.get("session_options") or {}}
        elif m.get("id") == OMNI_ID:
            # the variant from the entry the process ACTUALLY started with (file path at the manager)
            entry = ((engine_host.state(HOST) or {}).get("config") or {}).get("model") or {}
            variant = next((k for k, t in audiocpp_settings.OMNI_TYPES.items()
                            if entry.get("path", "") == str(omni_weights(t))), None)
            flash = (entry.get("session_options") or {}).get("omnivoice.perf_mode") == "flash_attention"
            result["omni"] = {"variant": variant, "flash": flash}
    return result


def has_vox(options: dict, backend: Optional[str] = None) -> bool:
    """Vox sits with EXACTLY these weight types and on this backend (empty = from the panel)."""
    from . import audiocpp_settings

    if process_backend() != (backend or audiocpp_settings.selected_backend()) or not _same_card():
        return False
    loaded = (state().get("vox") or {}).get("session_options", {})
    return all(loaded.get(k) == v for k, v in options.items())


def load_vox(options: dict, backend: Optional[str] = None) -> None:
    """Vox sits with this precision and backend → nothing. Otherwise restart with `lazy: false`."""
    with _lock:
        if has_vox(options, backend):
            return
        _start(_vox_entry(options, lazy=False), backend)


def unload() -> bool:
    """Stops the engine process (Vox, OmniVoice, or a lazy one). True = something was running."""
    try:
        return engine_host.unload(HOST)
    except engine_host.EngineHostError as e:
        raise RuntimeError(str(e)) from e


def _omni_entry(dtype: str, flash: bool, lazy: bool) -> dict:
    """The OmniVoice entry. Weight types as in the file (upstream defaults); `flash` =
    `omnivoice.perf_mode=flash_attention` — without it attention holds a T×T matrix."""
    return {
        "id": OMNI_ID,
        "family": "omnivoice",
        "task": "tts",
        # upstream's pseudo-streaming: one audio chunk per text chunk
        # (`text_chunk_size`); without it the whole utterance is one chunk
        "mode": "streaming",
        "path": str(omni_weights(dtype)),
        "lazy": lazy,
        "session_options": {
            **({"omnivoice.perf_mode": "flash_attention"} if flash else {}),
            "omnivoice.prompt_disk_cache_dir": _omni_cache_dir(omni_weights(dtype)),
        },
    }


def has_omni(variant: str, flash: bool, backend: Optional[str] = None) -> bool:
    """OmniVoice sits with EXACTLY this weight file, attention mode and backend (empty = from the panel)."""
    from . import audiocpp_settings

    if process_backend() != (backend or audiocpp_settings.selected_backend()) or not _same_card():
        return False
    o = state().get("omni") or {}
    return o.get("variant") == variant and o.get("flash") == flash


def load_omni(variant: str, flash: bool = True, backend: Optional[str] = None,
              force: bool = False) -> None:
    """OmniVoice sits with this file → nothing. Otherwise restart the engine with IT ALONE
    (Vox goes with the old process — one model at a time)."""
    from . import audiocpp_settings

    dtype = audiocpp_settings.OMNI_TYPES[variant]
    if not omni_weights(dtype).is_file():
        raise RuntimeError(f"no OmniVoice weights: {omni_weights(dtype)}")
    with _lock:
        if not force and has_omni(variant, flash, backend):
            return
        _start(_omni_entry(dtype, flash, lazy=False), backend, force=force)
