"""Engine Host — the contract every engine container meets, and the engine-agnostic Offloader.

An Engine Host is a container whose manager keeps AT MOST ONE model loaded, each model
in its own process:

    GET  /state   → {"loaded": bool, "model": str | None, ...}
    POST /unload  → kills the model's process (all of its VRAM comes back);
                    the container keeps running
    POST /load    → host-specific body (see `audiocpp_host.py`, `pytorch_host.py`)

Hosts come from `VOICEBOX_ENGINE_HOSTS` ("name=url,name=url"). The Offloader knows
nothing about engines: before any model loads, every OTHER host that holds something
is unloaded and checked. Within one host, `/load` swaps the model itself. The
containers themselves always stay up (empty, a few MB) — Voicebox never needs Docker.
"Keep all models in VRAM" in the panel turns the Offloader off.

Adding an engine:
  - a new audio.cpp or PyTorch model → nothing here, it lives in an existing host;
  - a new kind of container → implement `/state` + `/unload` + `/load` and add it
    to `VOICEBOX_ENGINE_HOSTS`.

Whisper is deliberately NOT a host here: it is the dictation ear, and unloading it
before every spoken reply would make each dictation wait for a reload.
"""

import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DEFAULT_HOSTS = "audiocpp=http://audiocpp-backend:17495,pytorch=http://pytorch-backend:8080"


def hosts() -> dict[str, str]:
    raw = os.environ.get("VOICEBOX_ENGINE_HOSTS", _DEFAULT_HOSTS)
    pairs = (item.split("=", 1) for item in raw.split(",") if "=" in item)
    return {name.strip(): url.strip().rstrip("/") for name, url in pairs if name.strip() and url.strip()}


class EngineHostError(RuntimeError):
    pass


_unresolved_until: dict[str, float] = {}


def _resolves(url: str, timeout: float = 0.5) -> bool:
    """⚠️ A stopped container disappears from Docker's DNS and `getaddrinfo` only gives
    up after ~4 s (Docker asks upstream). Every Offloader pass asks every host, so
    without a deadline one missing container added seconds to each generation.
    A miss is remembered for 30 s — a host that is not there costs once, not per sentence."""
    host = urlparse(url).hostname
    if not host or _unresolved_until.get(host, 0) > time.monotonic():
        return False
    result: list = []

    def resolve():
        try:
            socket.getaddrinfo(host, None)
            result.append(True)
        except OSError:
            result.append(False)

    thread = threading.Thread(target=resolve, daemon=True)
    thread.start()
    thread.join(timeout)
    if result and result[0]:
        _unresolved_until.pop(host, None)
        return True
    _unresolved_until[host] = time.monotonic() + 30
    return False


def request(host: str, path: str, data: Optional[dict] = None, timeout: float = 30,
            wait_s: float = 0) -> dict:
    """GET (no data) or POST JSON to a host's manager. `wait_s` > 0 = keep retrying a
    refused connection that long (the container may be starting together with Voicebox)."""
    base = hosts().get(host)
    if not base:
        raise EngineHostError(f"unknown engine host {host!r} (VOICEBOX_ENGINE_HOSTS)")
    req = urllib.request.Request(
        base + path,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    deadline = time.monotonic() + wait_s
    while True:
        if not _resolves(base):
            raise EngineHostError(f"{host}: container not running ({base})")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            try:
                body = json.loads(body).get("error", body)
            except ValueError:
                pass
            raise EngineHostError(f"{host} {path}: {body}") from e
        except (urllib.error.URLError, ConnectionError) as e:
            if time.monotonic() >= deadline:
                raise EngineHostError(f"{host} manager not answering ({base}): {e}") from e
            time.sleep(0.5)


def keep_all_models() -> bool:
    """The panel's "Keep all models in VRAM" (`keep_all_models`): the Offloader is off,
    models of different hosts may sit on the card side by side. Unreadable = off."""
    try:
        from ..database import session as db_session
        from . import settings as settings_service

        db = db_session.SessionLocal()
        try:
            return bool(settings_service.get_generation_settings(db).keep_all_models)
        finally:
            db.close()
    except Exception as e:
        logger.warning("keep_all_models unreadable (%s) — treating it as off", e)
        return False


def state(host: str) -> Optional[dict]:
    """None = the host is unreachable (a container that is down holds nothing)."""
    try:
        return request(host, "/state", timeout=3)
    except EngineHostError as e:
        logger.debug("engine host %s: %s", host, e)
        return None


def unload(host: str) -> bool:
    """True = something was loaded and is gone now."""
    current = state(host)
    if not current or not current.get("loaded"):
        return False
    request(host, "/unload", {}, timeout=60)
    return True


def offload_except(keep: Optional[str]) -> list[str]:
    """Unload every host except `keep` and VERIFY it is empty. Returns what went down.
    Off when the panel says "Keep all models in VRAM".

    ⚠️ THIS IS A CRITICAL PATH. The machine is used REMOTELY: an OOM or a hang is not
    a failed generation, it is losing access to the whole computer until someone is
    physically back at it. So every failure is hard — if memory was not freed, the
    next model does NOT load next to the old one."""
    if keep is not None and keep_all_models():
        return []
    freed, failed = [], []
    for host in hosts():
        if host == keep:
            continue
        current = state(host)
        if not current:  # container down — holds nothing
            continue
        name = f"{host}:{current.get('model')}"
        if not current.get("loaded"):
            continue
        try:
            request(host, "/unload", {}, timeout=60)
            after = state(host)
            if after and after.get("loaded"):
                logger.error("Offloader: %s reported unloaded but still holds a model", name)
                failed.append(name)
            else:
                freed.append(name)
        except EngineHostError as e:
            logger.error("Offloader: unloading %s failed: %s", name, e)
            failed.append(name)
    if freed:
        logger.info("Offloader: unloaded %s (room for %s)", ", ".join(freed), keep or "an in-process model")
    if failed:
        raise EngineHostError("memory not freed by: " + ", ".join(failed))
    return freed


def unload_all() -> list[str]:
    """Voicebox's last word: every engine host drops its model (the containers stay up)."""
    return offload_except(None)
