"""Manager of `whisper-server` — the `whisper` container, separate from Voicebox (10.10, grruwi's call).

The choice (model, quantization, GPU/CPU, "on startup") lives in Voicebox's PANEL.
Only its mirror sits here in /data/selection.json, so the container comes up with the
right model even when Voicebox is not there — the dictation talks to it directly.

The same shape as the engine hosts (`/state` with `loaded` and `model`), though Voicebox's
Offloader leaves whisper alone: it is the dictation ear.

Routes (port 17480):
  GET  /state                     what runs, the selection mirror, weights on disk
  PUT  /selection                 Voicebox writes the panel's choice (loads nothing)
  POST /load                      {model, quant, device, file, beam_size, fa, env, force} — starts the process
  POST /unload                    stops it (VRAM comes back only when the process ends)
  POST /v1/audio/transcriptions   like whisper-server; when nothing runs — starts with the selection

Changing the model = restarting the process (checked since 30.09, VRAM surely comes back).
On the GPU the process inherits the dump cores 6,7,14,15 from `command`; on the CPU it
takes the container's whole cpuset (no core 0/8) and as many threads as cores.
"""

import json
import logging
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s manager: %(message)s")
log = logging.getLogger("manager")

BINARY = Path("/whisper/bin/whisper-server")
WEIGHTS = Path("/whisper/weights")
SELECTION_FILE = Path("/data/selection.json")

PORT = 17480
# whisper-server only on the container's loopback — from outside everything goes through the manager.
INNER_HOST, INNER_PORT = "127.0.0.1", 17481
TRANSCRIBE_PATH = "/v1/audio/transcriptions"
DEVICES = ("gpu", "cpu")
_TIMEOUT_S = 600

_lock = threading.RLock()
_process = None  # subprocess.Popen
_loaded = None   # the set the process ACTUALLY started with


# Card visibility for the model's process (Voicebox's GPU choice, `services/gpu_card.py`).
_VISIBILITY = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES")


def _visibility(request: dict) -> dict:
    env = request.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError("env: a JSON object")
    return {k: str(v) for k, v in env.items() if k in _VISIBILITY}


def _read_selection() -> dict:
    try:
        return json.loads(SELECTION_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _write_selection(selection: dict) -> None:
    tmp = SELECTION_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(selection, ensure_ascii=False, indent=1))
    tmp.replace(SELECTION_FILE)


def _port_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((INNER_HOST, INNER_PORT)) == 0


def _running() -> bool:
    return _process is not None and _process.poll() is None


def _container_cores() -> str:
    return Path("/sys/fs/cgroup/cpuset.cpus.effective").read_text().strip()


def _core_count(cores: str) -> int:
    n = 0
    for part in cores.split(","):
        if "-" in part:
            a, b = part.split("-")
            n += int(b) - int(a) + 1
        elif part:
            n += 1
    return n


def _unload() -> bool:
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
    log.info("whisper-server stopped (pid %s)", _process.pid)
    _process, _loaded = None, None
    return True


def _load(request: dict) -> dict:
    global _process, _loaded
    file = str(request.get("file") or "")
    device = request.get("device") or "gpu"
    if not file or "/" in file or not (WEIGHTS / file).is_file():
        raise ValueError(f"no weights {file!r} in {WEIGHTS}")
    if device not in DEVICES:
        raise ValueError(f"unknown device: {device}")
    if not BINARY.is_file():
        raise RuntimeError(f"no binary {BINARY}")
    beam_size = str(request.get("beam_size") or "").strip()
    fa = request.get("fa") is not False  # no field = whisper.cpp's default (on)
    env = _visibility(request)
    target = {"model": request.get("model"), "quant": request.get("quant"),
              "device": device, "file": file, "beam_size": beam_size, "fa": fa, "env": env}

    # `force` (the "Reload" button in the panel) = restart even with the same set
    if not request.get("force") and _running() and _loaded == target and _port_open():
        return {}

    start = time.monotonic()
    _unload()

    argv = [
        str(BINARY),
        "-m", str(WEIGHTS / file),
        "--host", INNER_HOST,
        "--port", str(INNER_PORT),
        "--inference-path", TRANSCRIBE_PATH,
        # language detected, unless the caller gives its own
        "-l", "auto",
    ]
    # Beam from Voicebox's `WHISPER_BEAM_SIZE`. Empty = whisper-server's default.
    if beam_size:
        argv += ["-bs", beam_size]
    # Flash attention from Voicebox's panel (the Whisper tab); whisper.cpp has it on by default.
    if not fa:
        argv += ["-nfa"]
    if device == "cpu":
        cores = _container_cores()
        argv = ["taskset", "-c", cores] + argv + ["-ng", "-t", str(_core_count(cores))]

    # stdout/stderr inherited = `docker logs whisper` / Portainer
    _process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, env={**os.environ, **env})

    for _ in range(240):
        time.sleep(0.25)
        if _process.poll() is not None:
            _process = None
            raise RuntimeError("whisper-server died while starting — `docker logs whisper`")
        if _port_open():
            _loaded = target
            ms = int((time.monotonic() - start) * 1000)
            log.info("%s/%s on %s (beam=%s, fa=%s) up in %d ms", target["model"], target["quant"],
                     device, beam_size or "-", "yes" if fa else "no", ms)
            return {"ms": ms}

    _unload()
    raise RuntimeError("whisper-server not up in 60 s — `docker logs whisper`")


def state() -> dict:
    loaded = _loaded if _running() else None
    try:
        weights = {p.name: int(p.stat().st_size / 1048576) for p in sorted(WEIGHTS.glob("*.bin"))}
    except OSError:
        weights = {}
    return {
        "loaded": loaded is not None,
        "model": loaded["model"] if loaded else None,
        "config": loaded,
        "selection": _read_selection(),
        "weights": weights,
    }


def _transcribe(body: bytes, content_type: str) -> tuple:
    with _lock:
        if not (_running() and _port_open()):
            selection = _read_selection()
            if not selection.get("file"):
                raise RuntimeError("no model selected — load Whisper in Voicebox")
            _load(selection)
    req = urllib.request.Request(f"http://{INNER_HOST}:{INNER_PORT}{TRANSCRIBE_PATH}",
                                 data=body, headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
            return r.status, r.headers.get("Content-Type", "application/json"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", "application/json"), e.read()
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"whisper-server not answering ({e})") from e


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _reply(self, code: int, data, content_type: str = "application/json") -> None:
        if not isinstance(data, (bytes, bytearray)):
            data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _json(self) -> dict:
        body = self._body()
        return json.loads(body) if body.strip() else {}

    def do_GET(self):
        if self.path.split("?")[0] == "/state":
            return self._reply(200, state())
        self._reply(404, {"error": f"no {self.path}"})

    def do_PUT(self):
        if self.path.split("?")[0] != "/selection":
            return self._reply(404, {"error": f"no {self.path}"})
        try:
            _write_selection(self._json())
        except (ValueError, OSError) as e:
            return self._reply(400, {"error": str(e)})
        self._reply(200, state())

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            if path == "/load":
                request = self._json() or _read_selection()
                with _lock:
                    extra = _load(request)
                return self._reply(200, {**state(), **extra})
            if path == "/unload":
                with _lock:
                    unloaded = _unload()
                return self._reply(200, {**state(), "unloaded": unloaded})
            if path == TRANSCRIBE_PATH:
                code, content_type, data = _transcribe(self._body(), self.headers.get("Content-Type", ""))
                return self._reply(code, data, content_type)
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
    SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True

    def stop(*_):
        _unload()
        os._exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    selection = _read_selection()
    if selection.get("on_startup") and selection.get("file"):
        def load_on_startup():
            try:
                with _lock:
                    _load(selection)
            except (ValueError, RuntimeError) as e:
                log.warning("did not come up at startup: %s", e)
        threading.Thread(target=load_on_startup, daemon=True).start()

    log.info("listening on :%d", PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
