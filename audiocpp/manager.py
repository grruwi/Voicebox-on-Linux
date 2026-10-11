"""Manager of `audiocpp_server` — the `audiocpp-backend` container, separate from Voicebox (10.10, grruwi's call).

The voice engine (Vox, OmniVoice) lives in its own container, so Voicebox and the models
can move separately (another machine, another GPU). The brain stays in Voicebox
(`services/audiocpp_host.py`): it builds the model entry from the panel. Only the
process lives here.

The same contract as the `pytorch` manager, so Voicebox's Offloader treats every engine
container alike (it only needs `/state` and `/unload`).

⭐ ONE MODEL AT A TIME (grruwi's call 10.10 — the stream triggered at odd moments):
the process config always has EXACTLY one entry in `models`, and changing the model =
restarting the process, so VRAM surely comes back. The route takes ONE `model`, not a list.

Routes (port 17495):
  GET  /state     {loaded, backend, model, config}
  POST /load      {backend: hip|vulkan, model: {audio.cpp entry}, flags: [...], env: {...}, timeout_s, force}
                  — stops the old process, starts a new one with this single model and
                  returns once the engine port accepts connections (a `lazy: false` model
                  loads BEFORE the port opens)
  POST /unload    stops the process

The engine itself listens on 17496 in the `glos` network — Voicebox talks to it directly
(`/v1/audio/speech`, `/v1/models`, `/v1/tasks/unload_models`), not through here.
Engine log: /tmp/tts-server.log in the container (`docker exec audiocpp-backend tail ...`).
"""

import json
import logging
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s manager: %(message)s")
log = logging.getLogger("manager")

# The binaries are baked into the image (see the Dockerfile): bin-docker-vulkan always,
# bin-docker (HIP) only in the `rocm` target. `AUDIOCPP_BINARIES` points elsewhere.
_BINARIES = Path(os.environ.get("AUDIOCPP_BINARIES") or "/opt/audiocpp")
BINARIES = {
    "hip": _BINARIES / "bin-docker" / "audiocpp_server",
    "vulkan": _BINARIES / "bin-docker-vulkan" / "audiocpp_server",
}

PORT = 17495
ENGINE_HOST, ENGINE_PORT = "0.0.0.0", 17496
CONFIG = Path("/tmp/audiocpp-server.json")
LOG = Path("/tmp/tts-server.log")

_lock = threading.RLock()
_process = None  # subprocess.Popen
_loaded = None   # {backend, model, flags, env} the process ACTUALLY started with


# Card visibility for the model's process (Voicebox's GPU choice, `services/gpu_card.py`).
_VISIBILITY = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES")


def _visibility(request: dict) -> dict:
    env = request.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError("env: a JSON object")
    return {k: str(v) for k, v in env.items() if k in _VISIBILITY}


def _port_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", ENGINE_PORT)) == 0


def _running() -> bool:
    return _process is not None and _process.poll() is None


def _model_loaded(model_id: str) -> bool:
    """Whether the engine has this model LOADED (`/v1/models`, field `loaded`)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{ENGINE_PORT}/v1/models", timeout=3) as r:
            models = json.loads(r.read().decode("utf-8", "replace")).get("data", [])
    except (OSError, ValueError):
        return False
    return any(m.get("id") == model_id and m.get("loaded") for m in models)


def _log_tail(n: int = 12) -> str:
    try:
        return "\n".join(LOG.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _unload() -> bool:
    """SIGTERM, 10 s to exit, then SIGKILL. The port has to close."""
    global _process, _loaded
    if _process is None:
        return False
    if _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.warning("pid %s ignored SIGTERM for 10 s — SIGKILL", _process.pid)
            _process.kill()
            _process.wait(timeout=5)
    log.info("audiocpp_server stopped (pid %s)", _process.pid)
    _process, _loaded = None, None
    for _ in range(40):
        if not _port_open():
            break
        time.sleep(0.25)
    return True


def _load(request: dict) -> dict:
    global _process, _loaded
    backend = request.get("backend") or "hip"
    model = request.get("model")
    flags = [str(f) for f in (request.get("flags") or [])]
    timeout = float(request.get("timeout_s") or 300)
    if backend not in BINARIES:
        raise ValueError(f"backend: {'|'.join(BINARIES)}, not {backend!r}")
    if not isinstance(model, dict) or not model.get("id") or not model.get("family"):
        raise ValueError("model: ONE audio.cpp entry with `id` and `family`")
    binary = BINARIES[backend]
    if not binary.is_file():
        raise RuntimeError(f"no binary {binary}")
    env = _visibility(request)
    target = {"backend": backend, "model": model, "flags": flags, "env": env}

    # Same entry and the process is up → nothing, BUT only if the model really sits there
    # (or is meant to be lazy). After `/v1/tasks/unload_models` the process keeps the same
    # entry and loses the model — then restart, otherwise "load" would lie and the weights
    # would only come in with the first sentence.
    if not request.get("force") and _running() and _loaded == target and _port_open() and (
            model.get("lazy") or _model_loaded(model["id"])):
        return {"ms": 0}

    start = time.monotonic()
    _unload()
    CONFIG.write_text(json.dumps({
        "host": ENGINE_HOST, "port": ENGINE_PORT, "backend": backend,
        "models": [model],
    }))
    with open(LOG, "ab", buffering=0) as log_file:
        _process = subprocess.Popen(
            [str(binary), "--config", str(CONFIG), "--log"] + flags,
            stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env={**os.environ, **env},
        )
    while time.monotonic() - start < timeout:
        time.sleep(0.25)
        if _process.poll() is not None:
            code = _process.returncode
            _process = None
            raise RuntimeError(f"audiocpp_server died while starting (code {code}):\n{_log_tail()}")
        if _port_open():
            _loaded = target
            ms = int((time.monotonic() - start) * 1000)
            log.info("%s (%s, %s) up in %d ms", model["id"], backend,
                     "lazy" if model.get("lazy") else "loaded", ms)
            return {"ms": ms}
    _unload()
    raise RuntimeError(f"audiocpp_server not up in {int(timeout)} s:\n{_log_tail()}")


def state() -> dict:
    loaded = _loaded if _running() else None
    return {
        "loaded": loaded is not None,
        "backend": loaded["backend"] if loaded else None,
        "model": loaded["model"]["id"] if loaded else None,
        "config": loaded,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _reply(self, code: int, data) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        return json.loads(body) if body.strip() else {}

    def do_GET(self):
        if self.path.split("?")[0] == "/state":
            with _lock:
                return self._reply(200, state())
        self._reply(404, {"error": f"no {self.path}"})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            if path == "/load":
                request = self._json()
                with _lock:
                    extra = _load(request)
                    return self._reply(200, {**state(), **extra})
            if path == "/unload":
                with _lock:
                    unloaded = _unload()
                    return self._reply(200, {**state(), "unloaded": unloaded})
        except ValueError as e:
            return self._reply(400, {"error": str(e)})
        except RuntimeError as e:
            return self._reply(503, {"error": str(e)})
        self._reply(404, {"error": f"no {self.path}"})

    def log_message(self, fmt, *args):
        # The panel polls /state — keep it out of the log.
        if "/state" not in self.requestline:
            log.info("%s %s", self.address_string(), fmt % args)


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True

    def stop(*_):
        _unload()
        os._exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log.info("listening on %d (engine on %d)", PORT, ENGINE_PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
