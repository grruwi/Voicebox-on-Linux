"""
VoxCPM2 przez audio.cpp (ggml, HIP/ROCm) — backend, ktory NIE LADUJE ZADNYCH WAG.

Wagi siedza w gospodarzu (`services/gospodarz.py`, proces `audiocpp_server`),
tutaj leca tylko zadania HTTP: `/v1/audio/speech` (strumien surowego PCM
albo caly WAV). Gospodarz stoi pusty od startu Voiceboxa, vox laduje sie
w nim przy pierwszej wypowiedzi albo z ekranu Modele.

Od 10.10 zamiast `llama-tts-server` z llama.cpp-omni (kopia: tag `przed-audiocpp`
w `native/`). Zmierzone 10.10 na 9070 XT, Kleks 29 s, 21 krokow, CFG 1,8, F16:
RTF 0,40, pierwszy bajt 0,22–0,25 s (stary vox: RTF 0,43, 0,12–0,32 s).

⚠️ Pola panelu bez odpowiednika w audio.cpp (silnik ich nie zna, nie wysylamy):
temperatura, `stop_on_predictor`, `append_audio_start`, `target_sr`,
ziarnistosc strumienia (`vox_stream_min/co`).
"""

import asyncio
import io
import logging
import os
import random
import re
import wave
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

import numpy as np

from .base import combine_voice_prompts as _combine_voice_prompts

logger = logging.getLogger(__name__)

VOXCPM_SAMPLE_RATE = 48000

# Ten sam wzorzec co w `voxcpm_backend` — nawias na poczatku PRZELACZA tryb
# z kontynuacji na sterowanie stylem. Limit 200 znakow jest twardy po stronie
# modelu, nie nasz.
_CONTROL_PREFIX = re.compile(r"^\s*\([^)]{1,200}\)")


def _cfg(nazwa: str, domyslnie: str) -> str:
    return os.environ.get(nazwa, domyslnie).strip()


class VoxCPMGGUFBackend:
    """VoxCPM2 na audio.cpp. Wagi zyja w gospodarzu, nie tutaj."""

    def __init__(self):
        # Nazwa zostaje dla rejestru modeli (`backends/__init__.py`).
        self.model_size = "gguf-q8"
        self._sprawdzone = False

    # ─────────────────────────────────────────────── wagi i gospodarz

    @staticmethod
    def _precyzja() -> dict:
        """Typy wag (`session_options`) z panelu — JEDNO zrodlo:
        `services/tts_serwer.opcje_wag(wybrane())`, wspolne z przyciskiem przeladowania."""
        from ..services import tts_serwer

        return tts_serwer.opcje_wag(*tts_serwer.wybrane())

    def _serwer_stoi(self) -> bool:
        """Czy VOX siedzi w gospodarzu (nie: czy gospodarz stoi)."""
        from ..services import gospodarz

        return gospodarz.stan().get("vox") is not None

    def _serwer_ma_wagi(self, opcje: dict) -> bool:
        from ..services import gospodarz

        return gospodarz.ma_wagi(opcje)

    def _ubij_serwer(self) -> bool:
        """Zdejmuje voxa z gospodarza — VRAM wraca, backend zostaje."""
        from ..services import gospodarz

        if not gospodarz.stoi():
            return False
        byl = gospodarz.zdejmij_voxa()
        if byl:
            logger.info("vox zdjety z gospodarza — VRAM oddany")
        return byl

    def _podnies_serwer(self) -> bool:
        """Laduje voxa (precyzja z panelu), jesli nie siedzi tam juz ten sam."""
        from ..services import gospodarz

        if not self._serwer:
            return False
        gospodarz.upewnij()
        opcje = self._precyzja()
        if self._serwer_ma_wagi(opcje):
            return False
        gospodarz.zaladuj_voxa(opcje)
        logger.info("vox zaladowany do gospodarza (%s)", opcje)
        return True

    # ------------------------------------------------------------- protokol

    def is_loaded(self) -> bool:
        # Wagi siedza w gospodarzu, nie w tym procesie — pytamy jego.
        return self._serwer_stoi()

    def _get_model_path(self, model_size: str = "default") -> str:
        from ..services import gospodarz

        return str(gospodarz.wagi())

    def _is_model_cached(self, model_size: str = "default") -> bool:
        from ..services import gospodarz

        return any(gospodarz.wagi().glob("*.gguf"))

    async def load_model(self, model_size: str = "default") -> None:
        """Sprawdza, czy jest czym mowic, i laduje voxa do gospodarza.

        ⛔ NIE `and not self._serwer_stoi()`. Siedzacy vox moze miec INNA
        precyzje, niz wybrano w panelu — o tym wie dopiero `_podnies_serwer`,
        ktory porownuje `session_options` i w razie rozjazdu przeladowuje.
        """
        from ..services import gospodarz

        # binarke sprawdza zarzadca kontenera silnika — tu tylko wagi
        if not gospodarz.wagi().exists():
            raise RuntimeError(f"brak wag VoxCPM2: {gospodarz.wagi()}")
        if not self._sprawdzone:
            self._sprawdzone = True
            logger.info("VoxCPM2/audio.cpp gotowy: %s", gospodarz.wagi())
        if self._serwer:
            await asyncio.to_thread(self._podnies_serwer)

    def unload_model(self) -> None:
        """Oddaje VRAM — zdejmuje voxa z gospodarza."""
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
        # Sklejanie probek to numpy na CPU — dziala tak samo, silnik nieistotny.
        return await _combine_voice_prompts(audio_paths, reference_texts)

    # ⭐ POKRETLA: panel (baza) → `ustawienia.env` → awaryjne.
    #   cfg       — jak sztywno model trzyma sie nawiasu. Wyzej dokladniej ale
    #               plasko, nizej zywiej ale z wiekszym rozrzutem.
    #   timesteps — kroki dyfuzji. 30 do PROBEK REFERENCYJNYCH, 10 do gadania.
    _AWARYJNE: ClassVar[dict] = {"cfg": "2.0", "timesteps": "30"}

    def _pokretla(self) -> dict:
        """Kolejnosc: BAZA (suwaki w GUI) → zmienne z `ustawienia.env` → awaryjne.

        GUI wygrywa, bo zmiana suwaka ma dzialac od nastepnej generacji, bez
        restartu backendu. Gdy tabeli nie ma (stara baza), cicho spadamy nizej.
        """
        z_bazy: dict = {}
        try:
            # ⚠️ MODUL, nie nazwa. `SessionLocal` to globalna zmienna wypelniana
            # dopiero w `init_db()` — `from ..database import SessionLocal`
            # zlapalby `None` z chwili ladowania i padal na wywolaniu.
            from ..database import session as db_session
            from ..database import GenerationSettings

            s = db_session.SessionLocal()
            try:
                w = s.query(GenerationSettings).first()
                if w is not None:
                    if w.vox_cfg is not None:
                        z_bazy["cfg"] = str(w.vox_cfg)
                    if w.vox_timesteps is not None:
                        z_bazy["timesteps"] = str(w.vox_timesteps)
                    if getattr(w, "vox_seed", None) is not None:
                        z_bazy["seed"] = str(w.vox_seed)
            finally:
                s.close()
        except Exception as e:  # brak kolumny, brak wiersza, cokolwiek
            logger.debug("pokretla nie z bazy (%s) — biore ze srodowiska", e)

        return {
            "cfg": z_bazy.get("cfg") or _cfg("VOXCPM_CFG", self._AWARYJNE["cfg"]),
            "timesteps": z_bazy.get("timesteps") or _cfg("VOXCPM_TIMESTEPS", self._AWARYJNE["timesteps"]),
            # Pusty ciag = ziarno losowe (patrz `_zadanie`).
            "seed": z_bazy.get("seed") or _cfg("VOXCPM_SEED", ""),
            # ⭐ Bez pola w panelu — tylko `ustawienia.env`. PUSTE = silnik bierze swoje.
            #   max_steps → `max_tokens`: sufit krokow (patchy). Silnik: max_length modelu.
            #   min_steps → `min_tokens`: ile krokow, zanim wolno skonczyc. Silnik: 2.
            "max_steps": _cfg("VOXCPM_MAX_STEPS", ""),
            "min_steps": _cfg("VOXCPM_MIN_STEPS", ""),
        }

    def _zadanie(self, text: str, ref_audio: Optional[str], ref_text: Optional[str],
                 seed: Optional[int]) -> dict:
        """Cialo `/v1/audio/speech` wspolne dla strumienia i calosci."""
        stylizowany = bool(_CONTROL_PREFIX.match(text))
        p = self._pokretla()
        # ⛔ PUSTE POLE MUSI DAWAC LOSOWE ZIARNO, A NIE BRAK ZIARNA — staly seed
        # zabija „wygeneruj jeszcze raz, moze wyjdzie lepiej".
        ziarno = seed if seed is not None else p["seed"]
        if ziarno == "":
            ziarno = random.randrange(1, 2**31)
        zad = {
            "model": "voxcpm2",
            "input": text,
            "guidance_scale": float(p["cfg"]),
            "num_inference_steps": int(p["timesteps"]),
            "seed": int(ziarno),
            # Model stoi w trybie `streaming` — audio.cpp odrzuca wtedy KAZDE
            # zadanie bez `retry_badcase=false`, takze to na caly WAV
            # (powtorka zlego przypadku istnieje tylko w trybie offline).
            "options": {"retry_badcase": False},
            # ⭐ Probki float32 prosto z AudioVAE, bez kwantyzacji do 16 bitow
            # (nasza latka w `app/server/parallel_runtime.cpp`, 10.10). Serwer
            # potwierdza format naglowkiem `X-Sample-Format`.
            "sample_format": "f32",
        }
        if p["max_steps"]:
            zad["max_tokens"] = int(p["max_steps"])
        if p["min_steps"]:
            zad["options"]["min_tokens"] = int(p["min_steps"])
        if ref_audio:
            # Sciezka, nie base64: serwer siedzi w tym samym kontenerze i widzi
            # /app/data (base64 ma limit 5 MiB). ⚠️ ABSOLUTNA — wzgledna serwer
            # liczy od katalogu swojego configu (/tmp), nie od CWD backendu.
            zad["voice_ref"] = {"type": "path", "path": str(Path(ref_audio).resolve())}
            # ⭐ Transkrypt = tryb KONTYNUACJI (ta sama probka jako prompt; nasza
            # latka w `voxcpm2/session.cpp`). Tylko BEZ nawiasu — upstream
            # zabrania laczyc instrukcji stylu z prompt_text.
            if ref_text and not stylizowany:
                zad["reference_text"] = ref_text
        logger.info(
            "[VoxCPM2/audio.cpp] cfg=%s kroki=%s seed=%s | ref=%s tryb=%s",
            zad["guidance_scale"], zad["num_inference_steps"], zad["seed"],
            "tak" if ref_audio else "nie",
            "styl" if stylizowany else ("kontynuacja" if "reference_text" in zad else "barwa"),
        )
        return zad

    @staticmethod
    def _probka(voice_prompt: dict) -> Tuple[Optional[str], Optional[str]]:
        ref_audio = voice_prompt.get("ref_audio")
        if ref_audio and not Path(ref_audio).exists():
            logger.warning("nie ma probki referencyjnej: %s", ref_audio)
            ref_audio = None
        return ref_audio, voice_prompt.get("ref_text")

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        await self.load_model()
        ref_audio, ref_text = self._probka(voice_prompt)
        zad = self._zadanie(text, ref_audio, ref_text, seed)
        zad["response_format"] = "wav"
        return await asyncio.to_thread(self._generuj, zad)

    def _generuj(self, zad: dict) -> Tuple[np.ndarray, int]:
        import requests

        if not self._serwer:
            raise RuntimeError("vox wymaga gospodarza — VOXCPM_SERWER jest pusty")
        odp = requests.post(self._serwer.rstrip("/") + "/v1/audio/speech", json=zad, timeout=900)
        if odp.status_code != 200 or not odp.content.startswith(b"RIFF"):
            raise RuntimeError(f"audio.cpp: HTTP {odp.status_code} {odp.content[:300]!r}")
        # ⚠️ `soundfile`, NIE `wave` — `wave` zna tylko PCM i pada na WAV-ie
        # float32 (WAVE_FORMAT_IEEE_FLOAT).
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(odp.content), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return audio, sr

    SAMPLE_RATE: ClassVar[int] = VOXCPM_SAMPLE_RATE

    async def generate_stream(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ):
        """Kawalki dzwieku w miare powstawania, jako `np.float32`.

        `response_format=pcm` + `stream_format=audio` = gole probki mono 48 kHz
        po chunked, bez naglowka WAV: float32 LE przy `sample_format=f32`
        (naglowek `X-Sample-Format: f32`), inaczej s16le.
        """
        await self.load_model()

        adres = self._serwer
        if not adres:
            raise RuntimeError("strumien wymaga gospodarza — VOXCPM_SERWER jest pusty")

        ref_audio, ref_text = self._probka(voice_prompt)
        zad = self._zadanie(text, ref_audio, ref_text, seed)
        zad["response_format"] = "pcm"
        zad["stream_format"] = "audio"

        # ⛔ BEZ LIMITU (30.09). Watek wklada przez `call_soon_threadsafe(put_nowait)`,
        # wiec pelna kolejka NIE hamuje watku, tylko rzuca QueueFull w petli — ginie
        # kawalek albo KONIEC i generacja wisi na zawsze, a za nia cala kolejka zadan.
        # Cala wypowiedz w RAM to kilka MB.
        kolejka: asyncio.Queue = asyncio.Queue()
        petla = asyncio.get_running_loop()
        KONIEC = object()

        def pobieraj():
            """Blokujace czytanie strumienia HTTP w osobnym watku."""
            import requests

            reszta = b""
            try:
                with requests.post(adres.rstrip("/") + "/v1/audio/speech",
                                   json=zad, stream=True, timeout=900) as odp:
                    if odp.status_code != 200:
                        raise RuntimeError(f"audio.cpp: HTTP {odp.status_code} {odp.text[:300]}")
                    # Bez naglowka = binarka sprzed latki f32 → s16le, jak upstream.
                    f32 = odp.headers.get("X-Sample-Format") == "f32"
                    bajtow = 4 if f32 else 2
                    for blok in odp.iter_content(chunk_size=None):
                        if not blok:
                            continue
                        # Probka nie moze sie przeciac miedzy blokami HTTP;
                        # niepelna koncowka czeka na doklejenie do nastepnego.
                        blok = reszta + blok
                        uciete = len(blok) % bajtow
                        reszta = blok[len(blok) - uciete:] if uciete else b""
                        if uciete:
                            blok = blok[:len(blok) - uciete]
                        if not blok:
                            continue
                        if f32:
                            probki = np.frombuffer(blok, dtype="<f4").copy()
                        else:
                            probki = np.frombuffer(blok, dtype="<i2").astype(np.float32) / 32768.0
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

    # ── serwer ────────────────────────────────────────────────────────────

    @property
    def _serwer(self) -> str:
        """Adres gospodarza audio.cpp (`VOXCPM_SERWER`)."""
        return _cfg("VOXCPM_SERWER", "")
