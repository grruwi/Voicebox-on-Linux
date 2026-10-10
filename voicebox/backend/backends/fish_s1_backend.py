"""Fish S1-mini (PyTorch/ROCm) w kontenerze — backend sterujacy DOCKEREM.

Ta sama zasada co `fish_s2_backend` i `voxcpm_gguf_backend`: wagi NIE leza w
procesie Voiceboxa. Roznica jest jedna, ale zmienia caly rozdzial „zycie":
tam osobnym procesem byla binarka, tutaj jest KONTENER. Wiec:

    „zaladowany"  = kontener chodzi i `/v1/health` odpowiada
    „zaladuj"     = `fish-s1-up.sh` (tworzy albo podnosi, czeka na health)
    „odladuj"     = `docker stop fish-s1` — proces ginie, VRAM wraca

⛔ NIE odtwarzam tu polecenia `docker run`. Ono zyje w `fish-s1-up.sh` razem ze
zmiennymi, bez ktorych ten model jest 27x wolniejszy (`MIOPEN_FIND_MODE=FAST`:
68 s -> 2,5 s) i startuje 5x dluzej (cache kompilacji na dysku: 112 s -> 22 s).
Dwie kopie tej wiedzy rozjada sie przy pierwszej zmianie i nikt tego nie zauwazy.

⚠️ PROTOKOL TO ZWYKLY JSON, mimo ze oficjalny klient uzywa msgpack.
Sprawdzone: `POST /v1/tts` z `content-type: application/json` -> HTTP 200.
Probka referencyjna idzie jako base64 w `references[0].audio` — serwer sam ja
dekoduje (walidator w `ServeTTSRequest` robi to dla stringow dluzszych niz 255
znakow). Dzieki temu nie ma tu zaleznosci od `ormsgpack`.

Zmierzone 31.08 (RX 7900 XTX, kontener z `MIOPEN_FIND_MODE=FAST`):
    bez referencji      3,25 s dzwieku w 1,62 s
    z probka kleksa    15,09 s dzwieku w 2,64 s   RTF ~0,18
"""

import asyncio
import base64
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

# Kontener oddaje WAV w 44,1 kHz — tak jak s2.cpp, inaczej niz VoxCPM2 (48 kHz).
# Wartosc odczytujemy i tak z pliku, wiec to tylko notatka.
FISH_S1_SAMPLE_RATE = 44100

# ⛔ Nazwa kontenera i adres siedza TUTAJ, nie w `ustawienia.env`. Wpisanie ich
# do env zamroziloby je jako „czyjs wybor", a to sa czesci konstrukcji.
NAZWA_KONTENERA = "fish-s1"
ADRES_SERWERA = "http://127.0.0.1:8080"
LAUNCHER = MODELE_GLOSOWE / "fish-s1-up.sh"
KATALOG_WAG = MODELE_GLOSOWE / "fish-s1-mini"


def _cfg(nazwa: str, domyslnie: str) -> str:
    return os.environ.get(nazwa, domyslnie).strip()


class FishS1Backend:
    """Fish S1-mini w kontenerze. Wagi zyja w Dockerze, nie tutaj."""

    def __init__(self):
        # ⚠️ „default", a nie wlasna nazwa wariantu. Offloader pomija config o TYM
        # SAMYM rozmiarze co ladowany — rozjazd („pytorch" kontra domyslne
        # „default") kazalby mu zdjac wlasnie ten model, ktory za chwile stawia.
        self.model_size = "default"
        self._sprawdzone = False

    # -------------------------------------------------------------- pliki

    @property
    def _model(self) -> Path:
        return KATALOG_WAG / "model.pth"

    @property
    def _codec(self) -> Path:
        return KATALOG_WAG / "codec.pth"

    def _get_model_path(self, model_size: str = "default") -> str:
        return str(KATALOG_WAG)

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return self._model.exists() and self._codec.exists()

    # -------------------------------------------------------------- kontener

    @property
    def _serwer(self) -> str:
        """Czytane przez `check_model_loaded` w `backends/__init__.py`."""
        return ADRES_SERWERA

    def _serwer_stoi(self) -> bool:
        import socket
        from urllib.parse import urlparse

        cel = urlparse(ADRES_SERWERA)
        if not cel.hostname or not cel.port:
            return False
        try:
            with socket.create_connection((cel.hostname, cel.port), timeout=0.3):
                return True
        except OSError:
            return False

    @staticmethod
    def _docker(*argumenty: str, timeout: int = 60) -> subprocess.CompletedProcess:
        """⚠️ Bez `sudo` — grruwi jest w grupie `docker` (GID 950)."""
        return subprocess.run(
            ["docker", *argumenty], capture_output=True, text=True, timeout=timeout
        )

    def _kontener_chodzi(self) -> bool:
        wynik = self._docker("ps", "--filter", f"name={NAZWA_KONTENERA}",
                             "--format", "{{.Names}}", timeout=10)
        return NAZWA_KONTENERA in wynik.stdout.split()

    def _podnies(self) -> bool:
        """Woła launcher — on wie o zmiennych MIOpen i o cache na dysku."""
        if not LAUNCHER.exists():
            raise RuntimeError(f"nie ma launchera kontenera: {LAUNCHER}")

        # ⚠️ Zatrzymany kontener podnosimy przez `docker start` — launcher w tym
        # przypadku kasuje go i tworzy od nowa, co gubi log poprzedniego biegu.
        wynik = self._docker("ps", "-a", "--filter", f"name={NAZWA_KONTENERA}",
                             "--format", "{{.Names}}", timeout=10)
        if NAZWA_KONTENERA in wynik.stdout.split():
            self._docker("start", NAZWA_KONTENERA, timeout=30)
        else:
            wynik = subprocess.run(
                ["bash", str(LAUNCHER), "up"], capture_output=True, text=True, timeout=300
            )
            if wynik.returncode != 0:
                ogon = (wynik.stderr or wynik.stdout or "").strip().splitlines()[-5:]
                raise RuntimeError("fish-s1-up.sh padl: " + " | ".join(ogon))
            return True

        import time

        # Start z cache kompilacji trwa ~22 s; bez cache 112 s. Zapas do 3 min.
        for _ in range(180):
            if self._zdrowy():
                logger.info("fish-s1: kontener gotowy na %s", ADRES_SERWERA)
                return True
            if not self._kontener_chodzi():
                raise RuntimeError("fish-s1: kontener zszedl przy starcie — `docker logs fish-s1`")
            time.sleep(1)
        raise RuntimeError("fish-s1: kontener nie odpowiedzial w 3 min")

    def _zdrowy(self) -> bool:
        """Sam otwarty port nie wystarczy — serwer slucha, zanim wczyta wagi."""
        import requests

        try:
            odpowiedz = requests.get(ADRES_SERWERA + "/v1/health", timeout=2)
            return odpowiedz.status_code == 200
        except Exception:
            return False

    def _ubij(self) -> bool:
        if not self._kontener_chodzi():
            return False
        self._docker("stop", NAZWA_KONTENERA, timeout=60)

        import time

        for _ in range(30):
            if not self._serwer_stoi():
                logger.info("fish-s1: kontener zszedl, VRAM oddany")
                return True
            time.sleep(0.5)
        logger.warning("fish-s1: kontener nie zszedl w 15 s")
        return False

    # -------------------------------------------------------------- zycie

    def is_loaded(self) -> bool:
        # ⚠️ NIE sama flaga. Kontener zyje niezaleznie od tego procesu:
        # podniesiony z shella albo przetrwaly restart backendu trzyma
        # pamiec, a flaga jest wtedy False — offloader by go pominal
        # i zaladowal drugi model obok. `docker ps`, nie health:
        # pytanie brzmi "czy cos trzyma pamiec", nie "czy gotowy".
        return self._sprawdzone or self._kontener_chodzi()

    async def load_model(self, model_size: str = "default") -> None:
        brakuje = [str(p) for p in (self._model, self._codec) if not p.exists()]
        if brakuje:
            raise RuntimeError("brak wag S1-mini: " + ", ".join(brakuje))
        if not self._zdrowy():
            await asyncio.to_thread(self._podnies)
        if not self._sprawdzone:
            self._sprawdzone = True
            logger.info("Fish S1-mini gotowy: %s", KATALOG_WAG)

    def unload_model(self) -> None:
        """Oddaje VRAM — czyli zatrzymuje kontener, bo tam leza wagi."""
        self._sprawdzone = False
        self._ubij()

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

    # ⭐ POKRETLA SIEDZA W LAUNCHERZE `fish-s1-up.sh` — nie tutaj.
    # ⛔ PUSTE = POLA NIE WYSYLAMY. Serwer bierze wtedy swoja domyslna, a my nie
    # zamrazamy jego wyboru sprzed roku ani nie udajemy, ze ktos go dokonal.
    # Nazwa po lewej to POLE W JSON-ie (`ServeTTSRequest`), nie flaga CLI.
    _POKRETLA: ClassVar[tuple] = (
        ("temperature", "FISH_S1_TEMPERATURE"),
        ("top_p", "FISH_S1_TOP_P"),
        ("repetition_penalty", "FISH_S1_REPETITION_PENALTY"),
        ("max_new_tokens", "FISH_S1_MAX_NEW_TOKENS"),
        ("chunk_length", "FISH_S1_CHUNK_LENGTH"),
        ("use_memory_cache", "FISH_S1_MEMORY_CACHE"),
        ("normalize", "FISH_S1_NORMALIZE"),
    )

    @staticmethod
    def _na_wartosc(tekst: str):
        """`normalize` chce boola, `use_memory_cache` stringa „on"/„off",
        reszta liczb. Serwer waliduje przez pydantic i odrzuca zly typ."""
        if tekst.lower() in ("true", "yes", "tak"):
            return True
        if tekst.lower() in ("false", "no", "nie"):
            return False
        try:
            return int(tekst)
        except ValueError:
            pass
        try:
            return float(tekst)
        except ValueError:
            return tekst

    def _parametry(self) -> dict:
        pola = {}
        for nazwa, zmienna in self._POKRETLA:
            wartosc = _cfg(zmienna, "")
            if wartosc:
                pola[nazwa] = self._na_wartosc(wartosc)
        return pola

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

        return await asyncio.to_thread(self._generuj, text, ref_audio, ref_text, seed)

    def _generuj(
        self,
        text: str,
        ref_audio: Optional[str],
        ref_text: Optional[str],
        seed: Optional[int],
    ) -> Tuple[np.ndarray, int]:
        import requests

        zadanie = {"text": text, "format": "wav"}
        zadanie.update(self._parametry())
        if seed is not None:
            zadanie["seed"] = seed

        if ref_audio:
            with open(ref_audio, "rb") as plik:
                zakodowane = base64.b64encode(plik.read()).decode("ascii")
            zadanie["references"] = [{"audio": zakodowane, "text": ref_text or ""}]

        odpowiedz = requests.post(ADRES_SERWERA + "/v1/tts", json=zadanie, timeout=600)
        odpowiedz.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as plik:
            plik.write(odpowiedz.content)
            sciezka = Path(plik.name)
        try:
            return self._czytaj_wav(sciezka)
        finally:
            sciezka.unlink(missing_ok=True)

    @staticmethod
    def _czytaj_wav(sciezka: Path) -> Tuple[np.ndarray, int]:
        """⚠️ `soundfile`, NIE `wave` — modul `wave` zna tylko PCM i pada na
        float32 dopiero PO wygenerowaniu dzwieku."""
        import soundfile as sf

        dzwiek, czestotliwosc = sf.read(str(sciezka), dtype="float32", always_2d=False)
        if dzwiek.ndim > 1:
            dzwiek = dzwiek.mean(axis=1)
        return dzwiek.astype(np.float32), czestotliwosc
