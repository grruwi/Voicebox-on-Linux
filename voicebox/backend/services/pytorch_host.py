"""PyTorch Host — the client of the `pytorch-backend` container (PyTorch engines outside Voicebox).

The container's manager (an Engine Host, see `engine_host.py`) runs at most one model,
each in its own worker process; Voicebox talks to that worker directly
(`PYTORCH_WORKER`, `/generate`, `/generate/stream`). Unloading kills the worker, so all
of its VRAM comes back and the container keeps running.

A model is a directory `workers/<name>/` in the container; its options (e.g. the
precision set) go with `/load`, and a different set = a new worker process.
"""

import logging
import os
from typing import Optional

from . import engine_host

logger = logging.getLogger(__name__)

HOST = "pytorch"


def worker_url() -> str:
    return os.environ.get("PYTORCH_WORKER", "http://pytorch-backend:8081").strip().rstrip("/")


def state() -> Optional[dict]:
    """None = the container is unreachable."""
    return engine_host.state(HOST)


def has(model: str, options: Optional[dict] = None) -> bool:
    """`model` is loaded (and with exactly these options and the chosen card, when given)."""
    from . import gpu_card

    current = state() or {}
    if not current.get("loaded") or current.get("model") != model:
        return False
    return options is None or (current.get("options") == options
                               and (current.get("env") or {}) == gpu_card.engine_env())


def load(model: str, options: dict, force: bool = False, timeout_s: float = 900) -> dict:
    """Blocks until the worker is up (weights loaded, compiled). Call from a thread pool."""
    from . import gpu_card

    try:
        reply = engine_host.request(
            HOST, "/load", {"model": model, "options": options, "env": gpu_card.engine_env(),
                            "timeout_s": timeout_s, "force": force},
            timeout=timeout_s + 30, wait_s=20,
        )
    except engine_host.EngineHostError as e:
        raise RuntimeError(str(e)) from e
    logger.info("pytorch: %s %s up in %s ms", model, options, reply.get("ms"))
    return reply


def unload() -> bool:
    """True = a model was loaded and is gone now."""
    try:
        return engine_host.unload(HOST)
    except engine_host.EngineHostError as e:
        raise RuntimeError(str(e)) from e
