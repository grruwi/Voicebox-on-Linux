"""VoxCPM2 na PyTorchu/ROCm w KONTENERZE — drugi vox, obok tego z ggml.

⚠️ TO JEST TEN SAM MODEL CO `voxcpm`, TYLKO INNY SILNIK. Rozdzielone celowo,
zeby dalo sie je przelaczyc w GUI i uslyszec roznice:

    voxcpm     — ggml/Vulkan, `llama-tts-server` na :17496, RTF ~0,54-0,59
    voxcpm_pt  — PyTorch/ROCm, kontener `voxcpm-pt` na :17498, RTF ~0,69-0,72

PyTorch jest wolniejszy i to zmierzone (31.08, cfg 2.0 / timesteps 15, ta sama
karta, `MIOPEN_FIND_MODE=FAST`). Zostaje mimo to, bo tedy — i tylko tedy —
prowadzi droga do LoRA i fine-tuningu na polskim; ggml tego nie umie.

Serwer po drugiej stronie to NASZ plik (`/klocki/apczan-kutasu/voicebox/voice-models/voxcpm-pt/serwer.py`),
nie upstream: repo VoxCPM2 na HuggingFace niesie same wagi, bez kodu serwera.
⛔ Serwer nie ma zadnych domyslnych wartosci — pole nieprzyslane jest polem
nieprzekazanym do `generate()`.
"""

import asyncio
import base64
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

from ..config import MODELE_GLOSOWE

import numpy as np

from .base import combine_voice_prompts as _combine_voice_prompts

logger = logging.getLogger(__name__)

VOXCPM_SAMPLE_RATE = 48000

NAZWA_KONTENERA = "voxcpm-pt"
ADRES_SERWERA = "http://127.0.0.1:17498"
LAUNCHER = MODELE_GLOSOWE / "voxcpm-pt-up.sh"
KATALOG_WAG = MODELE_GLOSOWE / "voxcpm-pt/wagi"

# Nawias na poczatku tekstu = opis stylu. Ta sama regula co w silniku ggml.
_PREFIKS_STYLU = re.compile(r"^\s*\([^)]{1,200}\)")


def _cfg(nazwa: str, domyslnie: str) -> str:
    return os.environ.get(nazwa, domyslnie).strip()


class VoxCPMPyTorchBackend:
    """VoxCPM2 w kontenerze. Wagi zyja w Dockerze, nie tutaj."""

    def __init__(self):
        self.model_size = "default"
        self._sprawdzone = False

    # -------------------------------------------------------------- pliki

    @property
    def _model(self) -> Path:
        return KATALOG_WAG / "model.safetensors"

    @property
    def _audiovae(self) -> Path:
        return KATALOG_WAG / "audiovae.pth"

    def _get_model_path(self, model_size: str = "default") -> str:
        return str(KATALOG_WAG)

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return self._model.exists() and self._audiovae.exists()

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
        """⚠️ Bez `sudo` — grruwi jest w grupie `docker`."""
        return subprocess.run(
            ["docker", *argumenty], capture_output=True, text=True, timeout=timeout
        )

    def _kontener_chodzi(self) -> bool:
        wynik = self._docker("ps", "--filter", f"name={NAZWA_KONTENERA}",
                             "--format", "{{.Names}}", timeout=10)
        return NAZWA_KONTENERA in wynik.stdout.split()

    def _zdrowy(self) -> bool:
        """Sam otwarty port nie wystarczy — uvicorn slucha, zanim wczyta wagi."""
        import requests

        try:
            odpowiedz = requests.get(ADRES_SERWERA + "/health", timeout=2)
            return odpowiedz.status_code == 200 and '"ok"' in odpowiedz.text
        except Exception:
            return False

    def _podnies(self) -> bool:
        if not LAUNCHER.exists():
            raise RuntimeError(f"nie ma launchera kontenera: {LAUNCHER}")

        # ⚠️ Zatrzymany kontener podnosimy przez `docker start` — launcher w tym
        # przypadku kasuje go i tworzy od nowa, co gubi log poprzedniego biegu.
        wynik = self._docker("ps", "-a", "--filter", f"name={NAZWA_KONTENERA}",
                             "--format", "{{.Names}}", timeout=10)
        if NAZWA_KONTENERA not in wynik.stdout.split():
            wynik = subprocess.run(
                ["bash", str(LAUNCHER), "up"], capture_output=True, text=True, timeout=400
            )
            if wynik.returncode != 0:
                ogon = (wynik.stderr or wynik.stdout or "").strip().splitlines()[-5:]
                raise RuntimeError("voxcpm-pt-up.sh padl: " + " | ".join(ogon))
            return True

        self._docker("start", NAZWA_KONTENERA, timeout=30)

        import time

        for _ in range(180):
            if self._zdrowy():
                logger.info("voxcpm-pt: kontener gotowy na %s", ADRES_SERWERA)
                return True
            if not self._kontener_chodzi():
                raise RuntimeError("voxcpm-pt: kontener zszedl przy starcie — `docker logs voxcpm-pt`")
            time.sleep(1)
        raise RuntimeError("voxcpm-pt: kontener nie odpowiedzial w 3 min")

    def _ubij(self) -> bool:
        if not self._kontener_chodzi():
            return False
        self._docker("stop", NAZWA_KONTENERA, timeout=60)

        import time

        for _ in range(30):
            if not self._serwer_stoi():
                logger.info("voxcpm-pt: kontener zszedl, VRAM oddany")
                return True
            time.sleep(0.5)
        logger.warning("voxcpm-pt: kontener nie zszedl w 15 s")
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
        brakuje = [str(p) for p in (self._model, self._audiovae) if not p.exists()]
        if brakuje:
            raise RuntimeError("brak wag VoxCPM2 (PyTorch): " + ", ".join(brakuje))
        if not self._zdrowy():
            await asyncio.to_thread(self._podnies)
        if not self._sprawdzone:
            self._sprawdzone = True
            logger.info("VoxCPM2/PyTorch gotowy: %s", KATALOG_WAG)

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

    # ⭐ POKRETLA SIEDZA W LAUNCHERZE. ⛔ PUSTE = POLA NIE WYSYLAMY.
    # Osobne od `VOXCPM_*` (ggml) celowo: to inny silnik i moze potrzebowac
    # innego punktu pracy przy tych samych wagach.
    _POKRETLA: ClassVar[tuple] = (
        ("cfg_value", "VOXCPM_PT_CFG"),
        ("inference_timesteps", "VOXCPM_PT_TIMESTEPS"),
        ("normalize", "VOXCPM_PT_NORMALIZE"),
        ("denoise", "VOXCPM_PT_DENOISE"),
        ("retry_badcase", "VOXCPM_PT_RETRY_BADCASE"),
    )

    @staticmethod
    def _na_wartosc(tekst: str):
        if tekst.lower() in ("true", "yes", "tak", "1"):
            return True
        if tekst.lower() in ("false", "no", "nie", "0"):
            return False
        try:
            return int(tekst)
        except ValueError:
            pass
        try:
            return float(tekst)
        except ValueError:
            return tekst

    def _z_bazy(self) -> dict:
        """⭐ SUWAKI Z GUI. Te same kolumny co silnik ggml (`vox_cfg`,
        `vox_timesteps`) — ten sam model, ten sam sens parametru, wiec jeden
        suwak steruje obydwoma. Bez tego kręcenie w GUI nie robilo tu NIC.

        ⚠️ MODUL, nie nazwa: `SessionLocal` wypelnia sie dopiero w `init_db()`.
        """
        pola: dict = {}
        try:
            from ..database import session as db_session
            from ..database import GenerationSettings

            s = db_session.SessionLocal()
            try:
                w = s.query(GenerationSettings).first()
                if w is not None:
                    if w.vox_cfg is not None:
                        pola["cfg_value"] = float(w.vox_cfg)
                    if w.vox_timesteps is not None:
                        pola["inference_timesteps"] = int(w.vox_timesteps)
            finally:
                s.close()
        except Exception as blad:
            logger.debug("pokretla nie z bazy (%s) — biore ze srodowiska", blad)
        return pola

    def _parametry(self) -> dict:
        """Kolejnosc: `ustawienia.env` → BAZA (suwaki w GUI). Suwak wygrywa, bo
        ma dzialac od nastepnej generacji, bez restartu backendu — tak samo jak
        w silniku ggml.
        """
        pola = {}
        for nazwa, zmienna in self._POKRETLA:
            wartosc = _cfg(zmienna, "")
            if not wartosc:
                continue
            # cfg i timesteps to liczby — „1" ma zostac jedynka, nie True.
            if nazwa in ("cfg_value", "inference_timesteps"):
                pola[nazwa] = float(wartosc) if nazwa == "cfg_value" else int(wartosc)
            else:
                pola[nazwa] = self._na_wartosc(wartosc)

        pola.update(self._z_bazy())
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

        zadanie = {"text": text}
        zadanie.update(self._parametry())
        if seed is not None:
            zadanie["seed"] = seed

        if ref_audio:
            with open(ref_audio, "rb") as plik:
                zakodowane = base64.b64encode(plik.read()).decode("ascii")

            # Barwa — dziala bez transkryptu.
            zadanie["reference_wav"] = zakodowane

            # ⚠️ Sposob mowienia — wazny TYLKO w parze z transkryptem, upstream
            # rzuca przy samym `prompt_wav_path`. I odpada, gdy tekst niesie
            # nawias stylu: upstream zabrania obu instrukcji naraz. Ta sama
            # regula co w silniku ggml — patrz `voxcpm_backend.py`.
            if ref_text and not _PREFIKS_STYLU.match(text):
                zadanie["prompt_wav"] = zakodowane
                zadanie["prompt_text"] = ref_text

        logger.info(
            "[VoxCPM2/pt] %s | ref=%s | tryb=%s",
            {k: v for k, v in zadanie.items() if k not in ("text", "reference_wav", "prompt_wav")},
            "tak" if ref_audio else "nie",
            "styl" if _PREFIKS_STYLU.match(text) else "kontynuacja",
        )

        odpowiedz = requests.post(ADRES_SERWERA + "/generuj", json=zadanie, timeout=900)
        odpowiedz.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as plik:
            plik.write(odpowiedz.content)
            sciezka = Path(plik.name)
        try:
            return self._czytaj_wav(sciezka)
        finally:
            sciezka.unlink(missing_ok=True)

    # -------------------------------------------------------------- strumien

    # ⭐ Konsument rozpoznaje ten silnik przez `hasattr(backend, "generate_stream")`.
    # Silnik bez tej metody leci stara droga — zaden inny backend nie wymaga zmian.
    SAMPLE_RATE: ClassVar[int] = VOXCPM_SAMPLE_RATE

    async def generate_stream(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ):
        """Kawalki dzwieku w miare powstawania, jako `np.float32` @ 48 kHz.

        ⚠️ CALKOWITY CZAS TAKI SAM jak `generate()` — zysk to PIERWSZY DZWIEK
        po ~0,25 s zamiast po calosci. Sens ma tam, gdzie ktos slucha na zywo.
        ⛔ Efekty i normalizacja poziomow NIE dzialaja na strumieniu — obie
        potrzebuja calego przebiegu. Konsument musi to wiedziec.
        """
        await self.load_model()

        ref_audio = voice_prompt.get("ref_audio")
        ref_text = voice_prompt.get("ref_text")
        if ref_audio and not Path(ref_audio).exists():
            logger.warning("nie ma probki referencyjnej: %s", ref_audio)
            ref_audio = None

        zadanie = {"text": text}
        zadanie.update(self._parametry())
        if seed is not None:
            zadanie["seed"] = seed

        if ref_audio:
            with open(ref_audio, "rb") as plik:
                zakodowane = base64.b64encode(plik.read()).decode("ascii")
            zadanie["reference_wav"] = zakodowane
            if ref_text and not _PREFIKS_STYLU.match(text):
                zadanie["prompt_wav"] = zakodowane
                zadanie["prompt_text"] = ref_text

        logger.info(
            "[VoxCPM2/pt] STRUMIEN %s | ref=%s",
            {k: v for k, v in zadanie.items()
             if k not in ("text", "reference_wav", "prompt_wav", "prompt_text")},
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

            try:
                with requests.post(ADRES_SERWERA + "/generuj/strumien",
                                   json=zadanie, stream=True, timeout=900) as odpowiedz:
                    odpowiedz.raise_for_status()
                    for blok in odpowiedz.iter_content(chunk_size=None):
                        if blok:
                            petla.call_soon_threadsafe(
                                kolejka.put_nowait, np.frombuffer(blok, dtype=np.float32)
                            )
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
        import soundfile as sf

        dzwiek, czestotliwosc = sf.read(str(sciezka), dtype="float32", always_2d=False)
        if dzwiek.ndim > 1:
            dzwiek = dzwiek.mean(axis=1)
        return dzwiek.astype(np.float32), czestotliwosc
