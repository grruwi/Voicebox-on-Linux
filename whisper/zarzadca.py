"""Zarzadca `whisper-server` — kontener `whisper`, osobno od Voiceboxa (10.10, decyzja grruwiego).

Wybor (model, kwant, karta/procesor, „przy starcie") zyje w PANELU Voiceboxa.
Tu siedzi tylko jego lustro w /dane/wybor.json, zeby kontener wstal z dobrym
modelem takze wtedy, gdy Voiceboxa nie ma — dyktafon gada tu wprost.

Trasy (port 17480):
  GET  /stan                      co chodzi, lustro wyboru, wagi na dysku
  PUT  /wybor                     Voicebox wpisuje wybor z panelu (nic nie laduje)
  POST /zaladuj                   {model, kwant, urzadzenie, plik, bs} — podnosi proces
  POST /zdejmij                   zdejmuje (VRAM wraca dopiero z koncem procesu)
  POST /v1/audio/transcriptions   jak whisper-server; gdy nic nie chodzi — wstaje z wyborem

Zmiana modelu = restart procesu (sprawdzone od 30.09, VRAM na pewno wraca).
Na karcie proces dziedziczy smietnik 6,7,14,15 z `command`; na procesorze bierze
caly cpuset kontenera (bez rdzenia 0/8) i tyle watkow, ile rdzeni.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s zarzadca: %(message)s")
log = logging.getLogger("zarzadca")

BINARKA = Path("/whisper/bin/whisper-server")
WAGI = Path("/whisper/wagi")
PLIK_WYBORU = Path("/dane/wybor.json")

PORT = 17480
# whisper-server tylko w petli kontenera — z zewnatrz wszystko idzie przez zarzadce.
WEWN_HOST, WEWN_PORT = "127.0.0.1", 17481
SCIEZKA = "/v1/audio/transcriptions"
URZADZENIA = ("gpu", "cpu")
_TIMEOUT_S = 600

_blokada = threading.RLock()
_proces = None          # subprocess.Popen
_zaladowany = None      # zestaw, z ktorym proces FAKTYCZNIE wstal


def _czytaj_wybor() -> dict:
    try:
        return json.loads(PLIK_WYBORU.read_text())
    except (OSError, ValueError):
        return {}


def _zapisz_wybor(wybor: dict) -> None:
    tmp = PLIK_WYBORU.with_suffix(".tmp")
    tmp.write_text(json.dumps(wybor, ensure_ascii=False, indent=1))
    tmp.replace(PLIK_WYBORU)


def _port_nasluchuje() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((WEWN_HOST, WEWN_PORT)) == 0


def _chodzi() -> bool:
    return _proces is not None and _proces.poll() is None


def _rdzenie_kontenera() -> str:
    return Path("/sys/fs/cgroup/cpuset.cpus.effective").read_text().strip()


def _ile_rdzeni(lista: str) -> int:
    n = 0
    for kawalek in lista.split(","):
        if "-" in kawalek:
            a, b = kawalek.split("-")
            n += int(b) - int(a) + 1
        elif kawalek:
            n += 1
    return n


def _zatrzymaj() -> bool:
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
    log.info("whisper-server zdjety (pid %s)", _proces.pid)
    _proces, _zaladowany = None, None
    return True


def _zaladuj(z: dict) -> dict:
    global _proces, _zaladowany
    plik = str(z.get("plik") or "")
    urzadzenie = z.get("urzadzenie") or "gpu"
    if not plik or "/" in plik or not (WAGI / plik).is_file():
        raise ValueError(f"brak wag {plik!r} w {WAGI}")
    if urzadzenie not in URZADZENIA:
        raise ValueError(f"nieznane urzadzenie: {urzadzenie}")
    if not BINARKA.is_file():
        raise RuntimeError(f"brak binarki {BINARKA}")
    bs = str(z.get("bs") or "").strip()
    fa = z.get("fa") is not False  # brak pola = default whisper.cpp (wlaczony)
    cel = {"model": z.get("model"), "kwant": z.get("kwant"),
           "urzadzenie": urzadzenie, "plik": plik, "bs": bs, "fa": fa}

    # `wymus` (guzik „Przeladuj" w panelu) = restart takze przy tym samym zestawie
    if not z.get("wymus") and _chodzi() and _zaladowany == cel and _port_nasluchuje():
        return {}

    start = time.monotonic()
    _zatrzymaj()

    argv = [
        str(BINARKA),
        "-m", str(WAGI / plik),
        "--host", WEWN_HOST,
        "--port", str(WEWN_PORT),
        "--inference-path", SCIEZKA,
        # jezyk wykrywany, chyba ze wolajacy poda swoj
        "-l", "auto",
    ]
    # Beam z `WHISPER_BEAM_SIZE` Voiceboxa. Puste = default whisper-servera.
    if bs:
        argv += ["-bs", bs]
    # Flash attention z panelu Voiceboxa (zakladka Whisper); whisper.cpp ma go domyslnie wlaczonego.
    if not fa:
        argv += ["-nfa"]
    if urzadzenie == "cpu":
        rdzenie = _rdzenie_kontenera()
        argv = ["taskset", "-c", rdzenie] + argv + ["-ng", "-t", str(_ile_rdzeni(rdzenie))]

    # stdout/stderr dziedziczone = `docker logs whisper` / Portainer
    _proces = subprocess.Popen(argv, stdin=subprocess.DEVNULL)

    for _ in range(240):
        time.sleep(0.25)
        if _proces.poll() is not None:
            _proces = None
            raise RuntimeError("whisper-server padl przy starcie — `docker logs whisper`")
        if _port_nasluchuje():
            _zaladowany = cel
            ms = int((time.monotonic() - start) * 1000)
            log.info("%s/%s na %s (bs=%s, fa=%s) wstal w %d ms", cel["model"], cel["kwant"],
                     urzadzenie, bs or "-", "tak" if fa else "nie", ms)
            return {"ms": ms}

    _zatrzymaj()
    raise RuntimeError("whisper-server nie wstal w 60 s — `docker logs whisper`")


def stan() -> dict:
    zal = _zaladowany if _chodzi() else None
    try:
        wagi = {p.name: int(p.stat().st_size / 1048576) for p in sorted(WAGI.glob("*.bin"))}
    except OSError:
        wagi = {}
    return {"chodzi": zal is not None, "zaladowany": zal, "wybor": _czytaj_wybor(), "wagi": wagi}


def _transkrybuj(cialo: bytes, typ: str) -> tuple:
    with _blokada:
        if not (_chodzi() and _port_nasluchuje()):
            wybor = _czytaj_wybor()
            if not wybor.get("plik"):
                raise RuntimeError("brak wyboru modelu — zaladuj Whispera w Voiceboxie")
            _zaladuj(wybor)
    req = urllib.request.Request(f"http://{WEWN_HOST}:{WEWN_PORT}{SCIEZKA}",
                                 data=cialo, headers={"Content-Type": typ})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
            return r.status, r.headers.get("Content-Type", "application/json"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", "application/json"), e.read()
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"whisper-server nie odpowiada ({e})") from e


class Obsluga(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _odpowiedz(self, kod: int, dane, typ: str = "application/json") -> None:
        if not isinstance(dane, (bytes, bytearray)):
            dane = json.dumps(dane, ensure_ascii=False).encode()
        self.send_response(kod)
        self.send_header("Content-Type", typ)
        self.send_header("Content-Length", str(len(dane)))
        self.end_headers()
        self.wfile.write(dane)

    def _cialo(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _json(self) -> dict:
        cialo = self._cialo()
        return json.loads(cialo) if cialo.strip() else {}

    def do_GET(self):
        if self.path.split("?")[0] == "/stan":
            return self._odpowiedz(200, stan())
        self._odpowiedz(404, {"blad": f"nie ma {self.path}"})

    def do_PUT(self):
        if self.path.split("?")[0] != "/wybor":
            return self._odpowiedz(404, {"blad": f"nie ma {self.path}"})
        try:
            _zapisz_wybor(self._json())
        except (ValueError, OSError) as e:
            return self._odpowiedz(400, {"blad": str(e)})
        self._odpowiedz(200, stan())

    def do_POST(self):
        sciezka = self.path.split("?")[0]
        try:
            if sciezka == "/zaladuj":
                z = self._json() or _czytaj_wybor()
                with _blokada:
                    extra = _zaladuj(z)
                return self._odpowiedz(200, {**stan(), **extra})
            if sciezka == "/zdejmij":
                with _blokada:
                    zdjety = _zatrzymaj()
                return self._odpowiedz(200, {**stan(), "zdjety": zdjety})
            if sciezka == SCIEZKA:
                kod, typ, dane = _transkrybuj(self._cialo(), self.headers.get("Content-Type", ""))
                return self._odpowiedz(kod, dane, typ)
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
    PLIK_WYBORU.parent.mkdir(parents=True, exist_ok=True)
    serwer = ThreadingHTTPServer(("0.0.0.0", PORT), Obsluga)
    serwer.daemon_threads = True

    def koniec(*_):
        _zatrzymaj()
        os._exit(0)

    signal.signal(signal.SIGTERM, koniec)
    signal.signal(signal.SIGINT, koniec)

    wybor = _czytaj_wybor()
    if wybor.get("przy_starcie") and wybor.get("plik"):
        def _na_start():
            try:
                with _blokada:
                    _zaladuj(wybor)
            except (ValueError, RuntimeError) as e:
                log.warning("przy starcie nie wstal: %s", e)
        threading.Thread(target=_na_start, daemon=True).start()

    log.info("slucham na :%d", PORT)
    serwer.serve_forever()


if __name__ == "__main__":
    main()
