"""Whisper (whisper.cpp) — the CLIENT of the separate `whisper` container (since 10.10, grruwi's call).

`whisper-server` is no longer born in Voicebox. It lives in the `whisper` container
(`whisper/` in this repo), run by `manager.py`; this module only sends
it commands. The dictation talks to the container directly (127.0.0.1:17480), so
Voicebox can be down or busy speaking.

The panel's choice (`generation_settings`): model (medium/large/turbo), quantization
(q5/q8), device (gpu/cpu), "on startup". THE PANEL IS THE TRUTH — the manager only keeps
its mirror (`whisper/dane/selection.json`), so it can come up with the right model
without Voicebox. The mirror is sent on every panel change, at Voicebox's start and
before every load. Beam: `WHISPER_BEAM_SIZE` from Voicebox's env.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from ..config import VOICE_MODELS

logger = logging.getLogger(__name__)

_WEIGHTS = Path(os.environ.get("WHISPER_WEIGHTS") or VOICE_MODELS / "whisper-cpp" / "weights")

# (model, quant) -> ggml file. large-v3-q8_0 is not on HF — quantized by hand.
FILES = {
    ("medium", "q5"): "ggml-medium-q5_0.bin",
    ("medium", "q8"): "ggml-medium-q8_0.bin",
    ("large", "q5"): "ggml-large-v3-q5_0.bin",
    ("large", "q8"): "ggml-large-v3-q8_0.bin",
    ("turbo", "q5"): "ggml-large-v3-turbo-q5_0.bin",
    ("turbo", "q8"): "ggml-large-v3-turbo-q8_0.bin",
}
MODELS = ("medium", "large", "turbo")
QUANTS = ("q5", "q8")
DEVICES = ("gpu", "cpu")

# Defaults = what the host's whisper ran on (large-v3-q5_0 on the GPU).
_DEFAULTS = ("large", "q5", "gpu")

# The `glos` Docker network (shared with the `whisper` container). From the host: 127.0.0.1:17480.
MANAGER = os.environ.get("WHISPER_MANAGER", "http://whisper:17480").strip().rstrip("/")
TRANSCRIBE_PATH = "/v1/audio/transcriptions"
URL = MANAGER + TRANSCRIBE_PATH


def _http(method: str, path: str, data: Optional[dict] = None, timeout: float = 5.0) -> dict:
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(
        MANAGER + path, data=body, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("error")
        except Exception:
            detail = None
        raise RuntimeError(f"whisper container: {detail or e}") from e
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"whisper container not answering ({e}) — `docker start whisper`") from e


def file(model: str, quant: str) -> Path:
    return _WEIGHTS / FILES[(model, quant)]


def _settings_row():
    from ..database import GenerationSettings
    from ..database import session as db_session

    s = db_session.SessionLocal()
    try:
        return s.query(GenerationSettings).first()
    finally:
        s.close()


def selected() -> tuple:
    """(model, quant, device) from the DATABASE — the panel wins. Empty = defaults."""
    model, quant, device = _DEFAULTS
    try:
        row = _settings_row()
        if row is not None:
            model = getattr(row, "whisper_model", None) or model
            quant = getattr(row, "whisper_quant", None) or quant
            device = getattr(row, "whisper_device", None) or device
    except Exception as e:
        logger.debug("whisper: selection not from the database (%s) — using defaults", e)
    return (model, quant, device)


def on_startup() -> bool:
    """The panel switch: whether whisper comes up by itself (now: when the CONTAINER starts)."""
    try:
        row = _settings_row()
        return bool(row is not None and getattr(row, "whisper_on_startup", False))
    except Exception as e:
        logger.debug("whisper: the startup switch not from the database (%s)", e)
        return False


def flash() -> bool:
    """whisper-server flash attention from the panel — ON by default (like whisper.cpp)."""
    try:
        row = _settings_row()
        if row is not None and getattr(row, "whisper_fa", None) is not None:
            return bool(row.whisper_fa)
    except Exception as e:
        logger.debug("whisper: flash not from the database (%s)", e)
    return True


def _set(model: str, quant: str, device: str) -> dict:
    from . import gpu_card

    return {
        "model": model,
        "quant": quant,
        "device": device,
        "file": FILES[(model, quant)],
        # ⚠️ env_file is read ONLY when the Voicebox container is created
        "beam_size": os.environ.get("WHISPER_BEAM_SIZE", "").strip(),
        "fa": flash(),
        "env": gpu_card.engine_env(),
    }


def sync() -> None:
    """The panel's mirror → the manager. Loads nothing."""
    model, quant, device = selected()
    _http("PUT", "/selection", {**_set(model, quant, device), "on_startup": on_startup()})


def _manager_state() -> Optional[dict]:
    try:
        return _http("GET", "/state", timeout=2.0)
    except RuntimeError as e:
        logger.debug("whisper: %s", e)
        return None


def loaded() -> Optional[tuple]:
    """(model, quant, device) the process ACTUALLY holds. None = not running."""
    config = (_manager_state() or {}).get("config")
    if not config:
        return None
    return (config.get("model"), config.get("quant"), config.get("device"))


def stop() -> bool:
    """Unloads whisper in the container (VRAM comes back only when the process ends)."""
    try:
        return bool(_http("POST", "/unload", timeout=20.0).get("unloaded"))
    except RuntimeError as e:
        logger.warning("whisper: %s", e)
        return False


def start(model: Optional[str] = None, quant: Optional[str] = None,
          device: Optional[str] = None, force: bool = False) -> dict:
    """Starts whisper in the container. Empty arguments = what the panel has.

    When it already runs with exactly this set — the manager does nothing.
    """
    default_model, default_quant, default_device = selected()
    model = model or default_model
    quant = quant or default_quant
    device = device or default_device
    if (model, quant) not in FILES:
        raise ValueError(f"unknown whisper: {model}/{quant}")
    if device not in DEVICES:
        raise ValueError(f"unknown device: {device}")

    sync()
    s = _http("POST", "/load", {**_set(model, quant, device), "force": force}, timeout=70.0)
    if "ms" in s:
        logger.info("whisper: %s/%s on %s up in %d ms (container)", model, quant, device, s["ms"])
        return {**state(s), "ms": s["ms"]}
    return state(s)


def state(s: Optional[dict] = None) -> dict:
    if s is None:
        s = _manager_state()
    config = (s or {}).get("config")
    # weights from the container (it loads them); without the container — from our mount
    weights = (s or {}).get("weights")
    model, quant, device = selected()

    def on_disk(name: str) -> tuple:
        if weights is not None:
            return (name in weights, weights.get(name, 0))
        p = _WEIGHTS / name
        return (p.is_file(), int(p.stat().st_size / 1048576) if p.is_file() else 0)

    return {
        "loaded": bool(config),
        "container": s is not None,
        "loaded_model": config.get("model") if config else None,
        "loaded_quant": config.get("quant") if config else None,
        "loaded_device": config.get("device") if config else None,
        "loaded_fa": config.get("fa", True) if config else None,
        "selected_fa": flash(),
        "selected_model": model,
        "selected_quant": quant,
        "selected_device": device,
        "files": [
            {
                "model": m,
                "quant": q,
                "file": name,
                "on_disk": on_disk(name)[0],
                "size_mib": on_disk(name)[1],
            }
            for (m, q), name in FILES.items()
        ],
    }
