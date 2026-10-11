"""Manager of the `pytorch-backend` container — PyTorch engines for Voicebox, one model at a time.

The same contract as the `audiocpp` manager, so Voicebox's Offloader can treat every
engine container alike (it only needs `/state` and `/unload`):

  GET  /state    {loaded, model, options, pid}
  POST /load     {model: <dir in workers/>, options: {...}, env: {...}, timeout_s, force}
                 — stops the current worker, starts the requested one and returns when
                 its port accepts connections (workers load weights BEFORE opening it)
  POST /unload   stops the worker

⭐ ONE MODEL AT A TIME, EACH IN ITS OWN PROCESS. Unloading = killing the process, so all
of its VRAM comes back. Freeing a model inside a living torch process does not work
(measured on VoxCPM2: compiled graphs and CUDA graphs keep the old modules on the card).

⭐ NO TORCH HERE. The manager is stdlib only: an idle container touches no GPU and does
not spin a ROCr thread.

Adding an engine = a new directory `workers/<name>/` with `server.py` exposing a FastAPI
`app`. It reads its options from `WORKER_OPTIONS` (JSON), loads its model during startup
and serves on `WORKER_PORT`. Model paths come from the container environment.
"""

import collections
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s manager: %(message)s")
log = logging.getLogger("manager")

PORT = int(os.environ.get("MANAGER_PORT", "8080"))
WORKER_PORT = int(os.environ.get("WORKER_PORT", "8081"))
WORKERS = Path(__file__).resolve().parent / "workers"

_lock = threading.RLock()
_process = None  # subprocess.Popen
_loaded = None   # {model, options, env} the worker ACTUALLY started with


# Card visibility for the worker (Voicebox's GPU choice, `services/gpu_card.py`).
_VISIBILITY = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES")


def _visibility(request: dict) -> dict:
    env = request.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError("env: a JSON object")
    return {k: str(v) for k, v in env.items() if k in _VISIBILITY}
_log_tail = collections.deque(maxlen=40)


def available() -> list[str]:
    return sorted(p.parent.name for p in WORKERS.glob("*/server.py"))


def _port_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", WORKER_PORT)) == 0


def _running() -> bool:
    return _process is not None and _process.poll() is None


def _pump(stream) -> None:
    """Worker output → container log (`docker logs pytorch-backend`) + a tail for error messages."""
    for line in iter(stream.readline, b""):
        text = line.decode("utf-8", "replace").rstrip("\n")
        _log_tail.append(text)
        print(text, flush=True)


def _unload() -> bool:
    """SIGTERM, 15 s to exit, then SIGKILL. The port has to close."""
    global _process, _loaded
    if _process is None:
        return False
    if _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            log.warning("pid %s ignored SIGTERM for 15 s — SIGKILL", _process.pid)
            _process.kill()
            _process.wait(timeout=5)
    log.info("%s unloaded (pid %s)", (_loaded or {}).get("model"), _process.pid)
    _process, _loaded = None, None
    for _ in range(40):
        if not _port_open():
            break
        time.sleep(0.25)
    return True


def _load(request: dict) -> dict:
    global _process, _loaded
    model = request.get("model")
    options = request.get("options") or {}
    timeout = float(request.get("timeout_s") or 900)
    if model not in available():
        raise ValueError(f"model: one of {available()}, not {model!r}")
    if not isinstance(options, dict):
        raise ValueError("options: a JSON object")
    visibility = _visibility(request)
    target = {"model": model, "options": options, "env": visibility}

    if not request.get("force") and _running() and _loaded == target and _port_open():
        return {"ms": 0}

    start = time.monotonic()
    _unload()
    _log_tail.clear()
    env = {**os.environ, **visibility, "WORKER_OPTIONS": json.dumps(options), "WORKER_PORT": str(WORKER_PORT)}
    _process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", str(WORKER_PORT)],
        cwd=WORKERS / model, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    )
    threading.Thread(target=_pump, args=(_process.stdout,), daemon=True).start()
    while time.monotonic() - start < timeout:
        time.sleep(0.25)
        if _process.poll() is not None:
            code = _process.returncode
            _process = None
            raise RuntimeError(f"{model} died while starting (code {code}):\n" + "\n".join(list(_log_tail)[-12:]))
        if _port_open():
            _loaded = target
            ms = int((time.monotonic() - start) * 1000)
            log.info("%s %s up in %d ms", model, options, ms)
            return {"ms": ms}
    _unload()
    raise RuntimeError(f"{model} not up in {int(timeout)} s:\n" + "\n".join(list(_log_tail)[-12:]))


def state() -> dict:
    loaded = _loaded if _running() else None
    return {
        "loaded": loaded is not None,
        "model": loaded["model"] if loaded else None,
        "options": loaded["options"] if loaded else None,
        "env": loaded["env"] if loaded else None,
        "pid": _process.pid if loaded else None,
        "available": available(),
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
    log.info("listening on %d (worker on %d), workers: %s", PORT, WORKER_PORT, ", ".join(available()))
    server.serve_forever()


if __name__ == "__main__":
    main()
