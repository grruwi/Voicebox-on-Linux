"""Fish S2 Pro int8 (PyTorch/ROCm) w kontenerze — backend sterujacy DOCKEREM.

Blizniak `fish_s1_backend`: wagi nie leza w procesie Voiceboxa, tylko w
kontenerze. „Zaladowany" = `/v1/health` odpowiada, „odladuj" = `docker stop`.

⭐ TO NIE JEST TEN SAM S2 CO `fish_s2` — tamten liczy na ggml/Vulkan (s2.cpp),
   ten na PyTorchu, z wag skwantyzowanych do int8. Dwa silniki, jedne wagi
   zrodlowe, rozne sciezki. Stad `_pt` w nazwie, tak jak przy `voxcpm_pt`.

⛔ Polecenia `docker run` NIE ma tu kopii — siedzi w `fish-s2-int8-up.sh` razem
   z trzema latkami na kod fish-speech, bez ktorych ten model albo nie wstanie,
   albo zje dwa razy wiecej VRAM. Dwie kopie tej wiedzy rozjada sie po cichu.

Zmierzone 31.08 (RX 7900 XTX, int8, kontekst 4096):
    7,99 s dzwieku w 6,61 s   RTF ~0,83   VRAM 11,53 GiB
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

# Serwer oddaje WAV 44,1 kHz. Wartosc i tak czytamy z pliku — to notatka.
FISH_S2_PT_SAMPLE_RATE = 44100

# ⛔ Nazwa kontenera, port i sciezki to CZESCI KONSTRUKCJI, nie pokretla.
# W `ustawienia.env` zamrozilyby sie jako „czyjs wybor".
NAZWA_KONTENERA = "fish-s2-int8"
ADRES_SERWERA = "http://127.0.0.1:17499"
LAUNCHER = MODELE_GLOSOWE / "fish-s2-int8-up.sh"
KATALOG_WAG = MODELE_GLOSOWE / "fish-s2-pro/int8"


def _cfg(nazwa: str, domyslnie: str) -> str:
    return os.environ.get(nazwa, domyslnie).strip()


class FishS2PyTorchBackend:
    """Fish S2 Pro int8 w kontenerze. Wagi zyja w Dockerze, nie tutaj."""

    def __init__(self):
        # ⚠️ „default", nie wlasna nazwa wariantu — offloader pomija config o TYM
        # SAMYM rozmiarze co ladowany; rozjazd kazalby mu zdjac wlasnie ten
        # model, ktory za chwile stawia.
        self.model_size = "default"
        self._sprawdzone = False

    # -------------------------------------------------------------- pliki

    @property
    def _indeks(self) -> Path:
        return KATALOG_WAG / "model.safetensors.index.json"

    @property
    def _codec(self) -> Path:
        return KATALOG_WAG / "codec.pth"

    def _get_model_path(self, model_size: str = "default") -> str:
        return str(KATALOG_WAG)

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return self._indeks.exists() and self._codec.exists()

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
        """Woła launcher — on wie o latkach i o cache kompilacji na dysku."""
        if not LAUNCHER.exists():
            raise RuntimeError(f"nie ma launchera kontenera: {LAUNCHER}")

        # ⚠️ Zatrzymany kontener podnosimy przez `docker start`; launcher w tym
        # przypadku kasuje go i tworzy od nowa, gubiac log poprzedniego biegu.
        wynik = self._docker("ps", "-a", "--filter", f"name={NAZWA_KONTENERA}",
                             "--format", "{{.Names}}", timeout=10)
        if NAZWA_KONTENERA in wynik.stdout.split():
            self._docker("start", NAZWA_KONTENERA, timeout=30)
        else:
            wynik = subprocess.run(
                ["bash", str(LAUNCHER), "up"], capture_output=True, text=True, timeout=600
            )
            if wynik.returncode != 0:
                ogon = (wynik.stderr or wynik.stdout or "").strip().splitlines()[-5:]
                raise RuntimeError("fish-s2-int8-up.sh padl: " + " | ".join(ogon))
            return True

        import time

        # ⚠️ Start jest DLUZSZY niz u pozostalych: `--compile` buduje kernele przy
        # pierwszym przebiegu (119 s), potem czyta je z inductor/triton cache.
        for _ in range(300):
            if self._zdrowy():
                logger.info("fish-s2-int8: kontener gotowy na %s", ADRES_SERWERA)
                return True
            if not self._kontener_chodzi():
                raise RuntimeError(
                    "fish-s2-int8: kontener zszedl przy starcie — `docker logs fish-s2-int8`"
                )
            time.sleep(1)
        raise RuntimeError("fish-s2-int8: kontener nie odpowiedzial w 5 min")

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
                logger.info("fish-s2-int8: kontener zszedl, VRAM oddany")
                return True
            time.sleep(0.5)
        logger.warning("fish-s2-int8: kontener nie zszedl w 15 s")
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
        brakuje = [str(p) for p in (self._indeks, self._codec) if not p.exists()]
        if brakuje:
            raise RuntimeError("brak wag S2 Pro int8: " + ", ".join(brakuje))
        if not self._zdrowy():
            await asyncio.to_thread(self._podnies)
        if not self._sprawdzone:
            self._sprawdzone = True
            logger.info("Fish S2 Pro int8 gotowy: %s", KATALOG_WAG)

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

    # ⭐ POKRETLA SIEDZA W LAUNCHERZE OBOK WAG — nie tutaj.
    # ⛔ PUSTE = POLA NIE WYSYLAMY. Serwer bierze wtedy swoja domyslna.
    # Nazwa po lewej to POLE W JSON-ie (`ServeTTSRequest`), nie flaga CLI.
    _POKRETLA: ClassVar[tuple] = (
        ("temperature", "FISH_S2_PT_TEMPERATURE"),
        ("top_p", "FISH_S2_PT_TOP_P"),
        ("repetition_penalty", "FISH_S2_PT_REPETITION_PENALTY"),
        ("max_new_tokens", "FISH_S2_PT_MAX_NEW_TOKENS"),
        ("chunk_length", "FISH_S2_PT_CHUNK_LENGTH"),
        ("use_memory_cache", "FISH_S2_PT_MEMORY_CACHE"),
        ("normalize", "FISH_S2_PT_NORMALIZE"),
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

        logger.info(
            "[FishS2/int8] %s | ref=%s", self._parametry() or "domyslne",
            "tak" if ref_audio else "nie",
        )

        odpowiedz = requests.post(ADRES_SERWERA + "/v1/tts", json=zadanie, timeout=900)
        odpowiedz.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as plik:
            plik.write(odpowiedz.content)
            sciezka = Path(plik.name)
        try:
            return self._czytaj_wav(sciezka)
        finally:
            sciezka.unlink(missing_ok=True)

    # -------------------------------------------------------------- strumien

    SAMPLE_RATE: ClassVar[int] = FISH_S2_PT_SAMPLE_RATE

    async def generate_stream(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ):
        """Kawalki dzwieku w miare powstawania, jako `np.float32` @ 44,1 kHz.

        ⚠️ CALKOWITY CZAS TAKI SAM jak `generate()` — zysk to PIERWSZY DZWIEK
        zamiast ciszy do konca. ⛔ Efekty i normalizacja poziomow na strumieniu
        nie dzialaja: obie potrzebuja calego przebiegu.

        ⚠️ GRANULACJA JEST GRUBA. Fish tnie tekst na SEGMENTY (zdania) i oddaje
        gotowy segment, nie pojedyncze tokeny — przy jednym zdaniu strumien
        rowna sie zwyklej generacji. Zysk rosnie z dlugoscia tekstu.

        ⛔ FORMAT INNY NIZ U VOXA: serwer wysyla najpierw NAGLOWEK WAV (44 B),
        potem surowe probki **int16**. Stad odciecie naglowka i dzielenie przez
        32768 (`AMPLITUDE` w `tools/server/inference.py`). Bloki HTTP nie
        respektuja granicy probki, wiec nieparzysty ogon trzeba przeniesc do
        nastepnego bloku — inaczej dzwiek rozjezdza sie w szum.
        """
        await self.load_model()

        ref_audio = voice_prompt.get("ref_audio")
        ref_text = voice_prompt.get("ref_text")
        if ref_audio and not Path(ref_audio).exists():
            logger.warning("nie ma probki referencyjnej: %s", ref_audio)
            ref_audio = None

        zadanie = {"text": text, "format": "wav", "streaming": True}
        zadanie.update(self._parametry())
        if seed is not None:
            zadanie["seed"] = seed

        if ref_audio:
            with open(ref_audio, "rb") as plik:
                zakodowane = base64.b64encode(plik.read()).decode("ascii")
            zadanie["references"] = [{"audio": zakodowane, "text": ref_text or ""}]

        logger.info(
            "[FishS2/int8] STRUMIEN %s | ref=%s", self._parametry() or "domyslne",
            "tak" if ref_audio else "nie",
        )

        # ⛔ BEZ LIMITU (30.09). Watek wklada przez `call_soon_threadsafe(put_nowait)`,

        # wiec pelna kolejka NIE hamuje watku, tylko rzuca QueueFull w petli — ginie

        # kawalek albo KONIEC i generacja wisi na zawsze, a za nia cala kolejka zadan.

        # Silnik liczy ~3x szybciej niz ffplay gra; od strumienia co krok (upstream)

        # dluga wypowiedz przebijala 64 kawalki. Cala wypowiedz w RAM to kilka MB.

        kolejka: asyncio.Queue = asyncio.Queue()
        petla = asyncio.get_running_loop()
        KONIEC = object()

        def pobieraj():
            """Blokujace czytanie strumienia HTTP w osobnym watku."""
            import requests

            ogon = b""
            naglowek_zjedzony = 0
            try:
                with requests.post(ADRES_SERWERA + "/v1/tts",
                                   json=zadanie, stream=True, timeout=900) as odpowiedz:
                    odpowiedz.raise_for_status()
                    for blok in odpowiedz.iter_content(chunk_size=None):
                        if not blok:
                            continue
                        dane = ogon + blok

                        # Naglowek RIFF ma 44 bajty i moze przyjsc sklejony
                        # z pierwszym segmentem — odliczamy, ile jeszcze zostalo.
                        if naglowek_zjedzony < 44:
                            brakuje = 44 - naglowek_zjedzony
                            zjadamy = min(brakuje, len(dane))
                            naglowek_zjedzony += zjadamy
                            dane = dane[zjadamy:]
                            if not dane:
                                ogon = b""
                                continue

                        uzyteczne = len(dane) - (len(dane) % 2)
                        ogon = dane[uzyteczne:]
                        if uzyteczne:
                            probki = (
                                np.frombuffer(dane[:uzyteczne], dtype="<i2")
                                .astype(np.float32) / 32768.0
                            )
                            petla.call_soon_threadsafe(kolejka.put_nowait, probki)
            except Exception as blad:
                petla.call_soon_threadsafe(kolejka.put_nowait, blad)
            finally:
                petla.call_soon_threadsafe(kolejka.put_nowait, KONIEC)

        zadanie_watku = asyncio.create_task(asyncio.to_thread(pobieraj))
        try:
            while True:
                pozycja = await kolejka.get()
                if pozycja is KONIEC:
                    break
                if isinstance(pozycja, Exception):
                    raise pozycja
                yield pozycja
        finally:
            await zadanie_watku

    @staticmethod
    def _czytaj_wav(sciezka: Path) -> Tuple[np.ndarray, int]:
        """⚠️ `soundfile`, NIE `wave` — modul `wave` zna tylko PCM i pada na
        float32 dopiero PO wygenerowaniu dzwieku."""
        import soundfile as sf

        dzwiek, czestotliwosc = sf.read(str(sciezka), dtype="float32", always_2d=False)
        if dzwiek.ndim > 1:
            dzwiek = dzwiek.mean(axis=1)
        return dzwiek.astype(np.float32), czestotliwosc
