"""Fish Audio S2 Pro przez s2.cpp (ggml/Vulkan) — backend BEZ PyTorcha.

Ta sama zasada co `voxcpm_gguf_backend`: wagi NIE lezą w procesie Voiceboxa.
Inferencja idzie do osobnego procesu (`s2` z s2.cpp), ktory albo konczy sie po
kazdej wypowiedzi (CLI), albo stoi jako serwer i trzyma model na karcie.

⛔ SCIEZKA PYTORCHOWA ODRZUCONA. Zmierzone 31.08 na tej samej probce i tym
samym tekscie:

    PyTorch fp16 (GPU)      2,69 s dzwieku w 49,5 s   RAM 16,5 GB
    s2.cpp q8   (CPU)       7,06 s dzwieku w  105 s   RAM  6,2 GB
    s2.cpp q8   (Vulkan)    7,29 s dzwieku w 11,4 s   RTF 0,95

⚠️ BEZ WSKAZANIA KARTY SILNIK LICZY NA CPU — i jest wtedy 9x wolniejszy.
To nie jest domysl: `--vulkan` domyslnie -1, a `--gpu-layers` 0. Pokretla
`FISH_S2_VULKAN` i `FISH_S2_GPU_LAYERS` w `ustawienia.env` to wlaczaja;
puste = zostaje domyslne zachowanie silnika, czyli procesor.

⚠️ Model q8 (5,6 GB) NIESIE CODEC W SOBIE — nie ma osobnego pliku akustyki,
inaczej niz w VoxCPM2, gdzie BaseLM i Acoustic to zawsze para.
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

from ..config import MODELE_GLOSOWE

import numpy as np

from .base import combine_voice_prompts as _combine_voice_prompts

logger = logging.getLogger(__name__)

# s2.cpp zapisuje WAV w 44,1 kHz — inaczej niz VoxCPM2 (48 kHz).
FISH_SAMPLE_RATE = 44100


def _cfg(nazwa: str, domyslnie: str) -> str:
    return os.environ.get(nazwa, domyslnie).strip()


class FishS2Backend:
    """Fish S2 Pro na ggml. Wagi zyja w osobnym procesie, nie tutaj."""

    def __init__(self):
        self.model_size = "gguf-q8"
        self._sprawdzone = False

    # -------------------------------------------------------------- sciezki

    @property
    def _cli(self) -> Path:
        return Path(_cfg("FISH_S2_CLI", str(MODELE_GLOSOWE / "s2.cpp/build/s2")))

    @property
    def _model(self) -> Path:
        return Path(_cfg(
            "FISH_S2_MODEL",
            str(MODELE_GLOSOWE / "fish-s2-pro/s2-pro-q8_0.gguf"),
        ))

    @property
    def _tokenizer(self) -> Path:
        """Silnik szuka go sam obok modelu — sciezke podajemy tylko, gdy lezy gdzie indziej."""
        return Path(_cfg("FISH_S2_TOKENIZER", str(self._model.parent / "tokenizer.json")))

    @property
    def _katalog_glosow(self) -> str:
        """`--voice-dir`: tu ladują sie zakodowane probki referencyjne.

        Kodowanie probki kosztuje 0,59 s na karcie i 26,8 s na CPU — raz
        zapisana, wchodzi potem za darmo.
        """
        return _cfg("FISH_S2_VOICE_DIR", "")

    # -------------------------------------------------------------- serwer

    @property
    def _serwer(self) -> str:
        """Adres serwera s2.cpp. Pusty = wolamy CLI za kazdym razem."""
        return _cfg("FISH_S2_SERWER", "")

    def _adres_serwera(self) -> Optional[Tuple[str, int]]:
        adres = self._serwer
        if not adres:
            return None
        from urllib.parse import urlparse

        cel = urlparse(adres)
        if not cel.hostname or not cel.port:
            return None
        return cel.hostname, cel.port

    def _serwer_stoi(self) -> bool:
        cel = self._adres_serwera()
        if not cel:
            return False
        import socket

        try:
            with socket.create_connection(cel, timeout=0.3):
                return True
        except OSError:
            return False

    def _pid_serwera(self) -> Optional[int]:
        """Skan /proc zamiast `pkill -f`: ten wzorzec pasuje takze do powloki."""
        cel = self._adres_serwera()
        if not cel:
            return None
        znacznik = f"--port\x00{cel[1]}"
        for wpis in Path("/proc").iterdir():
            if not wpis.name.isdigit():
                continue
            try:
                cmdline = (wpis / "cmdline").read_text()
            except (OSError, UnicodeDecodeError):
                continue
            if "s2" in cmdline and "--server" in cmdline and znacznik in cmdline:
                return int(wpis.name)
        return None

    def _ubij_serwer(self) -> bool:
        import signal
        import time

        pid = self._pid_serwera()
        if pid is None:
            return False
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return False
        for _ in range(20):
            time.sleep(0.5)
            if not self._serwer_stoi():
                logger.info("s2.cpp: serwer %d zszedl, VRAM oddany", pid)
                return True
        logger.warning("s2.cpp: serwer %d nie zszedl w 10 s", pid)
        return False

    def _podnies_serwer(self) -> bool:
        cel = self._adres_serwera()
        if not cel:
            return False

        polecenie = [str(self._cli), "--server", "--host", cel[0], "--port", str(cel[1]),
                     "--model", str(self._model)]
        polecenie += self._flagi_sprzetu()
        if self._katalog_glosow:
            polecenie += ["--voice-dir", self._katalog_glosow]

        with open("/tmp/fish-s2-server.log", "ab") as dziennik:
            subprocess.Popen(
                polecenie,
                stdout=dziennik,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )

        import time

        for _ in range(60):
            time.sleep(0.5)
            if self._serwer_stoi():
                logger.info("s2.cpp: serwer wstal na %s:%d", *cel)
                return True
        logger.error("s2.cpp: serwer nie wstal w 30 s — patrz /tmp/fish-s2-server.log")
        return False

    # -------------------------------------------------------------- zycie

    def is_loaded(self) -> bool:
        return self._sprawdzone

    def _get_model_path(self, model_size: str = "default") -> str:
        return str(self._model)

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return self._model.exists()

    async def load_model(self, model_size: str = "default") -> None:
        brakuje = [str(p) for p in (self._cli, self._model) if not p.exists()]
        if brakuje:
            raise RuntimeError("brak plikow silnika s2.cpp: " + ", ".join(brakuje))
        if not self._sprawdzone:
            self._sprawdzone = True
            logger.info("Fish S2/ggml gotowy: %s", self._model.name)
        if self._adres_serwera() and not self._serwer_stoi():
            await asyncio.to_thread(self._podnies_serwer)

    def unload_model(self) -> None:
        """Oddaje VRAM — czyli ubija serwer, bo tam leza wagi."""
        self._sprawdzone = False
        self._ubij_serwer()

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        return {"ref_audio": str(audio_path), "ref_text": reference_text}, False

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts)

    # -------------------------------------------------------------- pokretla

    # ⭐ POKRETLA SIEDZA W LAUNCHERZE OBOK WAG — nie tutaj.
    # ⛔ PUSTE = NIE DOKLADAMY FLAGI. Silnik bierze wtedy swoja domyslna, a my
    # nie zamrazamy jego wyboru sprzed roku ani nie udajemy, ze ktos ja wybral.
    _POKRETLA: ClassVar[tuple] = (
        ("temperature", "FISH_S2_TEMPERATURE", "--temperature"),
        ("top_p", "FISH_S2_TOP_P", "--top-p"),
        ("top_k", "FISH_S2_TOP_K", "--top-k"),
        ("max_tokens", "FISH_S2_MAX_TOKENS", "--max-tokens"),
        ("codec_context_frames", "FISH_S2_CODEC_CONTEXT_FRAMES", "--codec-context-frames"),
        ("stream_decode_stride", "FISH_S2_DECODE_STRIDE", "--stream-decode-stride"),
        ("threads", "FISH_S2_THREADS", "--threads"),
    )

    def _flagi_sprzetu(self) -> List[str]:
        """Karta albo procesor. ⚠️ Puste = CPU, czyli 9x wolniej."""
        flagi: List[str] = []
        vulkan = _cfg("FISH_S2_VULKAN", "")
        if vulkan:
            flagi += ["--vulkan", vulkan]
        warstwy = _cfg("FISH_S2_GPU_LAYERS", "")
        if warstwy:
            flagi += ["--gpu-layers", warstwy]
        return flagi

    def _flagi_pokretel(self) -> List[str]:
        flagi: List[str] = []
        for _, zmienna, flaga in self._POKRETLA:
            wartosc = _cfg(zmienna, "")
            if wartosc:
                flagi += [flaga, wartosc]
        return flagi

    # -------------------------------------------------------------- mowa

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        await self.load_model()

        ref_audio = voice_prompt.get("ref_audio")
        ref_text = voice_prompt.get("ref_text")
        if ref_audio and not Path(ref_audio).exists():
            logger.warning("nie ma probki referencyjnej: %s", ref_audio)
            ref_audio = None

        if self._serwer_stoi():
            return await asyncio.to_thread(self._generuj_przez_serwer, text, ref_audio, ref_text)
        return await asyncio.to_thread(self._generuj, text, ref_audio, ref_text)

    def _generuj(
        self,
        text: str,
        ref_audio: Optional[str],
        ref_text: Optional[str],
    ) -> Tuple[np.ndarray, int]:
        """Jedno wywolanie CLI — proces konczy sie po wypowiedzi i oddaje karte."""
        with tempfile.TemporaryDirectory() as katalog:
            wyjscie = Path(katalog) / "mowa.wav"
            polecenie = [
                str(self._cli),
                "--model", str(self._model),
                "--text", text,
                "--output", str(wyjscie),
            ]
            if self._tokenizer.exists() and self._tokenizer.parent != self._model.parent:
                polecenie += ["--tokenizer", str(self._tokenizer)]
            if ref_audio:
                polecenie += ["--prompt-audio", ref_audio]
                if ref_text:
                    polecenie += ["--prompt-text", ref_text]
            if self._katalog_glosow:
                polecenie += ["--voice-dir", self._katalog_glosow]
            polecenie += self._flagi_sprzetu()
            polecenie += self._flagi_pokretel()

            wynik = subprocess.run(polecenie, capture_output=True, text=True)
            if wynik.returncode != 0 or not wyjscie.exists():
                ogon = (wynik.stderr or wynik.stdout or "").strip().splitlines()[-5:]
                raise RuntimeError("s2.cpp padl: " + " | ".join(ogon))

            return self._czytaj_wav(wyjscie)

    def _generuj_przez_serwer(
        self,
        text: str,
        ref_audio: Optional[str],
        ref_text: Optional[str],
    ) -> Tuple[np.ndarray, int]:
        """`POST /generate`, multipart. Model zostaje na karcie miedzy zdaniami."""
        import requests

        adres = self._serwer.rstrip("/") + "/generate"
        pola = {"text": (None, text)}

        # ⚠️ Serwer czyta te pola przez `get<int32_t>()` / `get<float>()`, wiec
        # string „0" leci na wyjatek i cale zadanie wraca jako 400 „Invalid params
        # JSON" — bez sladu, ktore pole zawinilo. Nazwy tez nie sa te same co flagi
        # CLI: `--threads` to w JSON-ie `n_threads`.
        _NAZWY_JSON = {"threads": "n_threads"}

        def _na_liczbe(t: str):
            try:
                return int(t)
            except ValueError:
                pass
            try:
                return float(t)
            except ValueError:
                return t

        parametry = {}
        for nazwa, zmienna, _ in self._POKRETLA:
            wartosc = _cfg(zmienna, "")
            if wartosc:
                parametry[_NAZWY_JSON.get(nazwa, nazwa)] = _na_liczbe(wartosc)

        # ⭐ Streaming: dekoder pracuje PRZEPLATANY z generacja zamiast dopiero po niej.
        # Odpowiedz wraca tym samym kanalem — serwer buforuje i oddaje caly WAV,
        # wiec po tej stronie nic sie nie zmienia. Nie ma flagi CLI, tylko to pole.
        strumien = _cfg("FISH_S2_STREAM", "")
        if strumien:
            parametry["stream"] = strumien.lower() not in ("0", "false", "no", "nie")

        if parametry:
            pola["params"] = (None, json.dumps(parametry), "application/json")

        uchwyt = None
        try:
            if ref_audio:
                uchwyt = open(ref_audio, "rb")
                pola["reference"] = (Path(ref_audio).name, uchwyt, "audio/wav")
                if ref_text:
                    pola["reference_text"] = (None, ref_text)

            odpowiedz = requests.post(adres, files=pola, timeout=600)
            odpowiedz.raise_for_status()
        finally:
            if uchwyt is not None:
                uchwyt.close()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as plik:
            plik.write(odpowiedz.content)
            sciezka = Path(plik.name)
        try:
            return self._czytaj_wav(sciezka)
        finally:
            sciezka.unlink(missing_ok=True)

    @staticmethod
    def _czytaj_wav(sciezka: Path) -> Tuple[np.ndarray, int]:
        """⚠️ `soundfile`, NIE `wave`.

        CLI zapisuje PCM 16-bit, ale serwer oddaje 32-bit float — modul `wave`
        zna tylko PCM i pada wtedy na `unknown format: 3`, juz PO wygenerowaniu
        dzwieku. Jedna sciezka odczytu dla obu, zeby nie zalezalo to od tego,
        czy `FISH_S2_SERWER` jest ustawiony.
        """
        import soundfile as sf

        dzwiek, czestotliwosc = sf.read(str(sciezka), dtype="float32", always_2d=False)
        if dzwiek.ndim > 1:
            dzwiek = dzwiek.mean(axis=1)
        return dzwiek.astype(np.float32), czestotliwosc
