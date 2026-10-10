"""Zarzadca `audiocpp_server` — kontener `audiocpp`, osobno od Voiceboxa (10.10, decyzja grruwiego).

Silnik glosu (Vox, OmniVoice) we wlasnym kontenerze, zeby Voicebox i modele daly sie
przenosic osobno (stodola, Ibanez). Mozg zostaje w Voiceboxie (`services/gospodarz.py`):
to on sklada wpis modelu z wyborow w panelu. Tu siedzi tylko proces.

⭐ JEDEN MODEL NARAZ (decyzja grruwiego 10.10 — stream triggeruje w dziwnych momentach):
config procesu ma zawsze DOKLADNIE jeden wpis w `models`, a zmiana modelu = restart
procesu, wiec VRAM na pewno wraca. Trasa przyjmuje JEDEN `model`, nie liste.

Trasy (port 17495):
  GET  /stan        {chodzi, backend, model, konfig}
  POST /uruchom     {backend: hip|vulkan, model: {wpis audio.cpp}, flagi: [...], limit_s, wymus}
                    — zdejmuje stary proces, podnosi nowy z tym jednym modelem i wraca,
                    gdy port silnika przyjmuje polaczenia (model `lazy: false` laduje
                    sie PRZED otwarciem portu)
  POST /zatrzymaj   zdejmuje proces

Sam silnik slucha na 17496 w sieci `glos` — Voicebox gada z nim wprost
(`/v1/audio/speech`, `/v1/models`, `/v1/tasks/unload_models`), bez przechodzenia tedy.
Log silnika: /tmp/tts-server.log w kontenerze (`docker exec audiocpp tail ...`).
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s zarzadca: %(message)s")
log = logging.getLogger("zarzadca")

# Gniazda robi `skrypty/buduj-audiocpp.sh [hip|vulkan]` (montowane :ro, ta sama sciezka).
# `AUDIOCPP_GNIAZDA` — inna maszyna (publiczne repo: binarka wbudowana w obraz, /opt/audiocpp).
_GNIAZDA = Path(os.environ.get("AUDIOCPP_GNIAZDA") or "/klocki/apczan-kutasu/voicebox/voice-models/audiocpp")
BINARKI = {
    "hip": _GNIAZDA / "bin-docker" / "audiocpp_server",
    "vulkan": _GNIAZDA / "bin-docker-vulkan" / "audiocpp_server",
}

PORT = 17495
SILNIK_HOST, SILNIK_PORT = "0.0.0.0", 17496
KONFIG = Path("/tmp/audiocpp-server.json")
LOG = Path("/tmp/tts-server.log")

_blokada = threading.RLock()
_proces = None      # subprocess.Popen
_zaladowany = None  # {backend, model, flagi}, z ktorymi proces FAKTYCZNIE wstal


def _port_nasluchuje() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", SILNIK_PORT)) == 0


def _chodzi() -> bool:
    return _proces is not None and _proces.poll() is None


def _model_siedzi(model_id: str) -> bool:
    """Czy silnik ma ten model ZALADOWANY (`/v1/models`, pole `loaded`)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{SILNIK_PORT}/v1/models", timeout=3) as r:
            modele = json.loads(r.read().decode("utf-8", "replace")).get("data", [])
    except (OSError, ValueError):
        return False
    return any(m.get("id") == model_id and m.get("loaded") for m in modele)


def _ogon_logu(n: int = 12) -> str:
    try:
        return "\n".join(LOG.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _zatrzymaj() -> bool:
    """SIGTERM, 10 s na zejscie, potem SIGKILL. Port musi sie zwolnic."""
    global _proces, _zaladowany
    if _proces is None:
        return False
    if _proces.poll() is None:
        _proces.terminate()
        try:
            _proces.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.warning("pid %s nie zszedl po SIGTERM w 10 s — SIGKILL", _proces.pid)
            _proces.kill()
            _proces.wait(timeout=5)
    log.info("audiocpp_server zdjety (pid %s)", _proces.pid)
    _proces, _zaladowany = None, None
    for _ in range(40):
        if not _port_nasluchuje():
            break
        time.sleep(0.25)
    return True


def _uruchom(z: dict) -> dict:
    global _proces, _zaladowany
    backend = z.get("backend") or "hip"
    model = z.get("model")
    flagi = [str(f) for f in (z.get("flagi") or [])]
    limit = float(z.get("limit_s") or 300)
    if backend not in BINARKI:
        raise ValueError(f"backend: {'|'.join(BINARKI)}, nie {backend!r}")
    if not isinstance(model, dict) or not model.get("id") or not model.get("family"):
        raise ValueError("model: JEDEN wpis audio.cpp z `id` i `family`")
    binarka = BINARKI[backend]
    if not binarka.is_file():
        raise RuntimeError(f"brak binarki {binarka}")
    cel = {"backend": backend, "model": model, "flagi": flagi}

    # Ten sam wpis i proces stoi → nic, ALE tylko gdy model faktycznie siedzi (albo ma byc
    # leniwy). Po `/v1/tasks/unload_models` proces zostaje z tym samym wpisem, a model
    # nie — wtedy restart, inaczej „zaladuj" by klamal i wagi wchodzilyby dopiero z mowa.
    if not z.get("wymus") and _chodzi() and _zaladowany == cel and _port_nasluchuje() and (
            model.get("lazy") or _model_siedzi(model["id"])):
        return {"ms": 0}

    start = time.monotonic()
    _zatrzymaj()
    KONFIG.write_text(json.dumps({
        "host": SILNIK_HOST, "port": SILNIK_PORT, "backend": backend,
        "models": [model],
    }))
    with open(LOG, "ab", buffering=0) as plik_logu:
        _proces = subprocess.Popen(
            [str(binarka), "--config", str(KONFIG), "--log"] + flagi,
            stdout=plik_logu, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )
    while time.monotonic() - start < limit:
        time.sleep(0.25)
        if _proces.poll() is not None:
            kod = _proces.returncode
            _proces = None
            raise RuntimeError(f"audiocpp_server padl przy starcie (kod {kod}):\n{_ogon_logu()}")
        if _port_nasluchuje():
            _zaladowany = cel
            ms = int((time.monotonic() - start) * 1000)
            log.info("%s (%s, %s) wstal w %d ms", model["id"], backend,
                     "lazy" if model.get("lazy") else "zaladowany", ms)
            return {"ms": ms}
    _zatrzymaj()
    raise RuntimeError(f"audiocpp_server nie wstal w {int(limit)} s:\n{_ogon_logu()}")


def stan() -> dict:
    zal = _zaladowany if _chodzi() else None
    return {
        "chodzi": zal is not None,
        "backend": zal["backend"] if zal else None,
        "model": zal["model"]["id"] if zal else None,
        "konfig": zal,
    }


class Obsluga(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _odpowiedz(self, kod: int, dane) -> None:
        dane = json.dumps(dane, ensure_ascii=False).encode()
        self.send_response(kod)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(dane)))
        self.end_headers()
        self.wfile.write(dane)

    def _json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        cialo = self.rfile.read(n) if n else b""
        return json.loads(cialo) if cialo.strip() else {}

    def do_GET(self):
        if self.path.split("?")[0] == "/stan":
            with _blokada:
                return self._odpowiedz(200, stan())
        self._odpowiedz(404, {"blad": f"nie ma {self.path}"})

    def do_POST(self):
        sciezka = self.path.split("?")[0]
        try:
            if sciezka == "/uruchom":
                z = self._json()
                with _blokada:
                    extra = _uruchom(z)
                    return self._odpowiedz(200, {**stan(), **extra})
            if sciezka == "/zatrzymaj":
                with _blokada:
                    zdjety = _zatrzymaj()
                    return self._odpowiedz(200, {**stan(), "zdjety": zdjety})
        except ValueError as e:
            return self._odpowiedz(400, {"blad": str(e)})
        except RuntimeError as e:
            return self._odpowiedz(503, {"blad": str(e)})
        self._odpowiedz(404, {"blad": f"nie ma {self.path}"})

    def log_message(self, fmt, *args):
        # /stan odpytuje panel — nie zasmiecamy logu
        if "/stan" not in self.requestline:
            log.info("%s %s", self.address_string(), fmt % args)


def main() -> None:
    serwer = ThreadingHTTPServer(("0.0.0.0", PORT), Obsluga)
    serwer.daemon_threads = True

    def koniec(*_):
        _zatrzymaj()
        os._exit(0)

    signal.signal(signal.SIGTERM, koniec)
    signal.signal(signal.SIGINT, koniec)
    log.info("nasluchuje na %d (silnik na %d)", PORT, SILNIK_PORT)
    serwer.serve_forever()


if __name__ == "__main__":
    main()
