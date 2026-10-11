"""What the panel picked for the audio.cpp engines, what actually sits there, and the
Reload / Unload buttons — the only place that decides Vox's weight precision.

Since 10.10 the engine is `audiocpp_server` (`services/audiocpp_host.py`). The weights
live in ONE GGUF (BF16 + AudioVAE F32), and the precision is chosen when the model loads
(`session_options`). When a PREBUILT file exists for the chosen generator precision
(`prebuilt/<type>/`, `audiocpp/make_weights.py`), the engine converts nothing
(~2 s, from a cold disk ~4 s); without it it converts the original (~7 s / ~23 s).
A change in the panel = restarting the engine with new `session_options`.
"""

import logging
import os
import shlex
from typing import Optional

logger = logging.getLogger(__name__)

# ⭐ SINCE 10.10 (grruwi's call, item 4 of the list): two panel fields, three model parts.
#   BaseLM   (`vox_base_lm`)   → `voxcpm2.base_lm_weight_type` — ONLY the BaseLM (our patch 11):
#            text → the plan of the utterance (words, stress, intonation); q8 = -1.3 GiB VRAM.
#   acoustic (`vox_acoustic`)  → `voxcpm2.weight_type` — the rest of the generator: ResidualLM,
#            LocEnc, DiT, projections (like "Acoustic" in the old Vox).
#   head     = AudioVAE → ALWAYS f32 (grruwi's call "F32 all the way"), no field.
# Prebuilt files for every combination: `prebuilt/<base_lm>-<acoustic>/`
# (`audiocpp/make_weights.py`); without the file the engine converts the original while loading.
BASE_LM_TYPES = {
    "q8": "q8_0",
    "f16": "f16",
    "f32": "f32",
}

# ⛔ There is NO q8 for the acoustic part (DiT diffusion).
ACOUSTIC_TYPES = {
    "f16": "f16",
    "f32": "f32",
}

AUDIOVAE_TYPE = "f32"

# Engine backend (`vox_backend`, 10.10): separate binaries, no cost for the unused one
# (one binary with both would register both and wake the ROCm runtime under Vulkan too).
# Built into the `audiocpp` image (target `rocm` adds HIP). A change = restarting the engine.
BACKENDS = {
    "hip": "bin-docker",
    "vulkan": "bin-docker-vulkan",
}

# OmniVoice (`omni_weights`, 10.10): ready GGUFs from upstream (HF `audio-cpp/audio.cpp-gguf`,
# `OmniVoice-GGUF/`) in `weights/omnivoice/`. q8 = upstream's default (`recommended_package`).
OMNI_TYPES = {
    "q8": "q8_0",
    "f16": "f16",
}

# Stream granularity (`vox_stream_min` / `vox_stream_every`) — the panel fields stay for
# compatibility, the engine does not know them (audio.cpp emits a chunk per patch).
_STREAM_MIN = 12
_STREAM_EVERY = 16


def weight_options(base_lm: str, acoustic: str) -> dict:
    """Panel keys → the engine's `session_options`. The only place of this mapping."""
    return {
        "voxcpm2.base_lm_weight_type": BASE_LM_TYPES.get(base_lm, BASE_LM_TYPES["q8"]),
        "voxcpm2.weight_type": ACOUSTIC_TYPES.get(acoustic, ACOUSTIC_TYPES["f16"]),
        "voxcpm2.audiovae_weight_type": AUDIOVAE_TYPE,
    }


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _settings_row():
    """The generation settings row, or None. Read fresh every time — the panel wins."""
    from ..database import GenerationSettings
    from ..database import session as db_session

    s = db_session.SessionLocal()
    try:
        return s.query(GenerationSettings).first()
    finally:
        s.close()


def _setting(name: str):
    try:
        row = _settings_row()
        return getattr(row, name, None) if row is not None else None
    except Exception as e:
        logger.debug("%s not from the database (%s) — using the fallback", name, e)
        return None


def selected_precision() -> tuple:
    """(base_lm, acoustic) from the DATABASE — the panel wins. Fallback: env, then defaults.

    One source of truth for both roads to the engine (the backend and `reload_vox`).
    """
    base_lm = (_setting("vox_base_lm") or "").lower() or None
    acoustic = (_setting("vox_acoustic") or "").lower() or None
    return (
        base_lm or _env("VOXCPM_BASE_LM", "q8").lower(),
        acoustic or "f16",
    )


def selected_backend() -> str:
    """hip | vulkan from the DATABASE — the panel wins; vulkan by default (the only one in the public image)."""
    backend = (_setting("vox_backend") or "").lower() or None
    return backend if backend in BACKENDS else "vulkan"


def selected_omni() -> str:
    """q8 | f16 from the DATABASE — the panel wins; q8 by default (like upstream)."""
    variant = (_setting("omni_weights") or "").lower() or None
    return variant if variant in OMNI_TYPES else "q8"


def selected_omni_flash() -> bool:
    """OmniVoice flash attention from the DATABASE — YES by default (grruwi's call "FA on everything")."""
    flash = _setting("omni_flash")
    return True if flash is None else bool(flash)


def selected_stream() -> tuple:
    """(min, every) from the DATABASE — only to show in the panel."""
    def number(value, variable: str, default: int) -> int:
        if value:
            return int(value)
        from_env = _env(variable)
        return int(from_env) if from_env.isdigit() and int(from_env) > 0 else default

    return (
        number(_setting("vox_stream_min"), "VOXCPM_STREAM_MIN", _STREAM_MIN),
        number(_setting("vox_stream_every"), "VOXCPM_STREAM_EVERY", _STREAM_EVERY),
    )


def extra_flags() -> list:
    """Any flags appended to `audiocpp_server` from `AUDIOCPP_EXTRA_FLAGS`.

    A free slot: anything `audiocpp_server --help` understands goes here as one string.
    `shlex` knows quotes. Empty = nothing appended.
    """
    raw = _env("AUDIOCPP_EXTRA_FLAGS")
    if not raw:
        return []
    try:
        split = shlex.split(raw)
    except ValueError as e:
        logger.warning("AUDIOCPP_EXTRA_FLAGS cannot be parsed (%s) — skipping", e)
        return []
    logger.info("the TTS engine gets flags from the settings: %s", " ".join(split))
    return split


def _in_use() -> bool:
    """`AUDIOCPP_ENGINE` empty = the audio.cpp engine is not used at all."""
    return bool(_env("AUDIOCPP_ENGINE", "http://audiocpp:17496"))


def loaded_precision() -> tuple:
    """(base_lm, acoustic) Vox ACTUALLY sits with in the engine — from its `/v1/models`,
    not from the panel. (None, None) = it does not sit there."""
    from . import audiocpp_host

    vox = audiocpp_host.state().get("vox")
    if vox is None:
        return (None, None)
    options = vox.get("session_options", {})
    base_type = options.get("voxcpm2.base_lm_weight_type", options.get("voxcpm2.weight_type"))
    base = next((k for k, t in BASE_LM_TYPES.items() if base_type == t), None)
    acoustic = next((k for k, t in ACOUSTIC_TYPES.items() if options.get("voxcpm2.weight_type") == t), None)
    return (base, acoustic)


def state() -> dict:
    from . import audiocpp_host

    host = audiocpp_host.state()
    vox_loaded = host.get("vox") is not None
    loaded_base, loaded_acoustic = loaded_precision() if vox_loaded else (None, None)
    selected_base, selected_acoustic = selected_precision()
    stream_min, stream_every = selected_stream()
    omni = host.get("omni")
    loaded_backend = audiocpp_host.process_backend() if (vox_loaded or omni) else None
    return {
        "available": _in_use(),
        "vox_loaded": vox_loaded,
        "loaded_base_lm": loaded_base,
        "loaded_acoustic": loaded_acoustic,
        "selected_base_lm": selected_base,
        "selected_acoustic": selected_acoustic,
        "loaded_backend": loaded_backend,
        "selected_backend": selected_backend(),
        # OmniVoice sits in the same engine — one model at a time, so either it or Vox
        "omni_loaded": omni is not None,
        "loaded_omni": (omni or {}).get("variant"),
        "loaded_omni_flash": (omni or {}).get("flash"),
        "selected_omni": selected_omni(),
        "omni": [_describe_file(k, audiocpp_host.omni_weights(t)) for k, t in OMNI_TYPES.items()],
        "base_lm": _describe(BASE_LM_TYPES),
        "acoustic": _describe(ACOUSTIC_TYPES),
        "loaded_stream_min": None,
        "loaded_stream_every": None,
        "selected_stream_min": stream_min,
        "selected_stream_every": stream_every,
    }


def _describe_file(key: str, path) -> dict:
    on_disk = path.is_file()
    return {
        "key": key,
        "file": path.name,
        "on_disk": on_disk,
        "size_mib": int(path.stat().st_size / 1048576) if on_disk else 0,
    }


def _describe(types: dict) -> list:
    """Every variant can be loaded from the original (the engine converts it); prebuilt
    files exist per COMBINATION of fields, so only the type and the original here."""
    from . import audiocpp_host

    files = sorted(audiocpp_host.weights().glob("*.gguf"))
    path = files[0] if files else None
    return [{
        "key": key,
        "file": f"{path.name} → {dtype}" if path else f"(no GGUF) → {dtype}",
        "on_disk": path is not None,
        "size_mib": int(path.stat().st_size / 1048576) if path else 0,
    } for key, dtype in types.items()]


def reload_vox(base_lm: Optional[str] = None, acoustic: Optional[str] = None,
               backend: Optional[str] = None) -> dict:
    """Loads Vox in the engine with the given precision. Empty = what the panel has."""
    import time

    default_base, default_acoustic = selected_precision()
    base_lm = (base_lm or default_base).lower()
    acoustic = (acoustic or default_acoustic).lower()
    if base_lm not in BASE_LM_TYPES:
        raise ValueError(f"unknown base_lm variant: {base_lm}")
    if acoustic not in ACOUSTIC_TYPES:
        raise ValueError(f"unknown acoustic variant: {acoustic}")
    backend = (backend or selected_backend()).lower()
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend: {backend}")
    if not _in_use():
        raise RuntimeError("the audio.cpp engine is not used (empty AUDIOCPP_ENGINE) — nothing to reload")

    from . import audiocpp_host

    start = time.monotonic()
    audiocpp_host.load_vox(weight_options(base_lm, acoustic), backend)
    ms = int((time.monotonic() - start) * 1000)
    logger.info("Vox reloaded in the engine: base_lm=%s acoustic=%s backend=%s in %d ms",
                base_lm, acoustic, backend, ms)
    return {**state(), "ms": ms}


def reload_omni() -> dict:
    """The "Reload" button on the OmniVoice tab: restart the engine with OmniVoice from the
    panel (weights, flash attention, backend) — also with the same set. Vox goes."""
    import time

    if not _in_use():
        raise RuntimeError("the audio.cpp engine is not used (empty AUDIOCPP_ENGINE)")
    from . import audiocpp_host

    start = time.monotonic()
    audiocpp_host.load_omni(selected_omni(), selected_omni_flash(), selected_backend(), force=True)
    ms = int((time.monotonic() - start) * 1000)
    logger.info("OmniVoice reloaded: weights=%s flash=%s backend=%s in %d ms",
                selected_omni(), selected_omni_flash(), selected_backend(), ms)
    return {**state(), "ms": ms}


def unload_omni() -> dict:
    """The "Unload" button on the OmniVoice tab — the engine process stops, VRAM comes back."""
    from . import audiocpp_host

    if audiocpp_host.state().get("omni") is not None:
        audiocpp_host.unload()
    return state()
