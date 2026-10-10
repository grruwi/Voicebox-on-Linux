"""
OmniVoice (k2-fsa, 600+ jezykow) przez audio.cpp — backend, ktory NIE LADUJE ZADNYCH WAG.

Ten sam silnik co vox (`audiocpp_server` w kontenerze `audiocpp`, `services/gospodarz.py`),
inna rodzina modelu. ⭐ JEDEN MODEL NARAZ (decyzja grruwiego 10.10): zaladowanie OmniVoice
= restart silnika z samym nim, vox schodzi — i odwrotnie. Offloader Voiceboxa
(`evict_other_models`) i tak zdejmuje inne TTS przed ladowaniem; zarzadca kontenera
pilnuje tego drugi raz, na poziomie procesu.

Wagi: gotowe GGUF-y upstreamu (HF `audio-cpp/audio.cpp-gguf`, `OmniVoice-GGUF/`)
w `voice-models/audiocpp/wagi/omnivoice/`, wariant z panelu (`omni_wagi`: q8 | f16).

Klonowanie: probka profilu (`voice_ref`, sciezka) + transkrypt (`reference_text`).
Bez probki: `instruct` = voice design („female, young adult, moderate pitch"),
bez obu = glos automatyczny.

⭐ WLASNE SUWAKI (zakladka OmniVoice w ustawieniach, 11.10): `omni_cfg` → `guidance_scale`,
`omni_kroki` → `num_inference_steps`, `omni_seed` → `seed`. Puste = default upstreamu
(guidance 2,0, 32 kroki, ziarno losowe). `omni_flash` = `omnivoice.perf_mode=flash_attention`
(sesja, zmiana = restart silnika przy nastepnym ladowaniu) — bez niego uwaga liczy pelna
macierz T×T (T = probka + tekst + mowa, x2 przez CFG) i VRAM rosnie z najdluzsza wypowiedzia.
Reszta pokretel (speed, duration_sec, shift, denoise, temperatury, ...) = defaulty upstreamu. OmniVoice to model niekauzalny: liczy cala wypowiedz naraz
i dlugosc zgaduje z gory (z tempa probki), wiec strumien = jeden kawalek na koniec,
chyba ze ktos ustawi `text_chunk_size` (pseudo-streaming upstreamu).
"""

import asyncio
import io
import logging
import os
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

import numpy as np

from .base import combine_voice_prompts as _combine_voice_prompts

logger = logging.getLogger(__name__)

OMNIVOICE_SAMPLE_RATE = 24000

# Pseudo-strumien (zakladka OmniVoice, 11.10 — odpowiednik `--stream-by-line` z omnivoice.cpp):
# tekst ciety na kawalki, kazdy liczony i wypuszczany osobno, wiec pierwszy dzwiek przychodzi
# po pierwszym kawalku zamiast po calosci. „brak" = nic nie wysylamy (upstream sam tnie
# dopiero powyzej 30 s mowy). 160 znakow = wartosc z przykladow strumienia w docs upstreamu.
_TRYBY_CIECIA = {"linie": "endline", "zdania": "tag_aware"}
_ZNAKI_CIECIA = 160


class OmniVoiceBackend:
    """OmniVoice na audio.cpp. Wagi zyja w silniku (kontener `audiocpp`), nie tutaj."""

    def __init__(self):
        self.model_size = "default"

    # ─────────────────────────────────────────────── wagi i silnik

    @staticmethod
    def _wariant() -> str:
        from ..services import tts_serwer

        return tts_serwer.wybrany_omni()

    def _plik(self) -> Path:
        from ..services import gospodarz, tts_serwer

        return gospodarz.wagi_omni(tts_serwer.OMNI_WARIANTY[self._wariant()])

    def is_loaded(self) -> bool:
        from ..services import gospodarz

        return gospodarz.stan().get("omni") is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return str(self._plik())

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return self._plik().is_file()

    async def load_model(self, model_size: str = "default") -> None:
        """Laduje OmniVoice do silnika (wariant z panelu), jesli nie siedzi tam juz ten sam."""
        from ..services import gospodarz

        if not self._serwer:
            raise RuntimeError("OmniVoice wymaga silnika audio.cpp — VOXCPM_SERWER jest pusty")
        from ..services import tts_serwer

        await asyncio.to_thread(gospodarz.zaladuj_omni, self._wariant(), tts_serwer.wybrany_omni_flash())

    def unload_model(self) -> None:
        """Oddaje VRAM — zdejmuje OmniVoice z silnika, proces zostaje."""
        from ..services import gospodarz

        if gospodarz.stoi() and gospodarz.zdejmij_omni():
            logger.info("OmniVoice zdjety z silnika — VRAM oddany")

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

    # ─────────────────────────────────────────────── zadanie

    @staticmethod
    def _suwaki() -> dict:
        """cfg / kroki / seed z zakladki OmniVoice. Brak wartosci = nie wysylamy (default upstreamu)."""
        wynik: dict = {}
        try:
            # MODUL, nie nazwa — `SessionLocal` wypelnia dopiero `init_db()`.
            from ..database import session as db_session
            from ..database import GenerationSettings

            s = db_session.SessionLocal()
            try:
                w = s.query(GenerationSettings).first()
                if w is not None:
                    if getattr(w, "omni_cfg", None) is not None:
                        wynik["guidance_scale"] = float(w.omni_cfg)
                    if getattr(w, "omni_kroki", None) is not None:
                        wynik["num_inference_steps"] = int(w.omni_kroki)
                    if getattr(w, "omni_seed", None) is not None:
                        wynik["seed"] = int(w.omni_seed)
                    if getattr(w, "omni_tempo", None) is not None:
                        wynik["speed"] = float(w.omni_tempo)
                    tryb = getattr(w, "omni_chunk_tryb", None) or "brak"
                    if tryb in _TRYBY_CIECIA:
                        wynik["text_chunk_mode"] = _TRYBY_CIECIA[tryb]
                        wynik["text_chunk_size"] = int(getattr(w, "omni_chunk_znaki", None) or _ZNAKI_CIECIA)
            finally:
                s.close()
        except Exception as e:
            logger.debug("suwaki OmniVoice nie z bazy (%s) — defaulty upstreamu", e)
        return wynik

    def _zadanie(self, text: str, voice_prompt: dict, language: Optional[str],
                 seed: Optional[int], instruct: Optional[str]) -> dict:
        """Cialo `/v1/audio/speech` wspolne dla strumienia i calosci."""
        zad = {
            "model": "omnivoice",
            "input": text,
            # probki float32 (nasza latka w `app/server/runtime.cpp`, wspolna dla rodzin)
            "sample_format": "f32",
            "options": {},
        }
        ref_audio = voice_prompt.get("ref_audio")
        if ref_audio and not Path(ref_audio).exists():
            logger.warning("nie ma probki referencyjnej: %s", ref_audio)
            ref_audio = None
        if ref_audio:
            # Sciezka ABSOLUTNA — silnik widzi /app/data pod ta sama sciezka (kontener `audiocpp`).
            zad["voice_ref"] = {"type": "path", "path": str(Path(ref_audio).resolve())}
            if voice_prompt.get("ref_text"):
                zad["reference_text"] = voice_prompt["ref_text"]
        if instruct:
            zad["options"]["instruction"] = instruct
        if language:
            zad["options"]["language"] = language
        suwaki = self._suwaki()
        if "guidance_scale" in suwaki:
            zad["guidance_scale"] = suwaki["guidance_scale"]
        if "num_inference_steps" in suwaki:
            zad["num_inference_steps"] = suwaki["num_inference_steps"]
        # Ziarno: z wywolania, potem z panelu. Puste = silnik ciagnie dalej swoj RNG sesji
        # (rozne przy kazdej generacji).
        if "speed" in suwaki:
            # tempo = mnoznik przy szacowaniu dlugosci mowy (model zgaduje ja z tempa probki)
            zad["options"]["speed"] = suwaki["speed"]
        if "text_chunk_mode" in suwaki:
            zad["options"]["text_chunk_mode"] = suwaki["text_chunk_mode"]
            zad["options"]["text_chunk_size"] = suwaki["text_chunk_size"]
        ziarno = seed if seed is not None else suwaki.get("seed")
        if ziarno is not None:
            zad["seed"] = int(ziarno)
        logger.info("[OmniVoice/audio.cpp] wagi=%s cfg=%s kroki=%s tempo=%s ciecie=%s/%s seed=%s jezyk=%s | ref=%s instruct=%s",
                    self._wariant(), zad.get("guidance_scale", "-"), zad.get("num_inference_steps", "-"),
                    zad["options"].get("speed", "-"), zad["options"].get("text_chunk_mode", "-"), zad["options"].get("text_chunk_size", "-"),
                    zad.get("seed", "-"), language or "auto",
                    "tak" if ref_audio else "nie", "tak" if instruct else "nie")
        return zad

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        await self.load_model()
        zad = self._zadanie(text, voice_prompt, language, seed, instruct)
        zad["response_format"] = "wav"
        return await asyncio.to_thread(self._generuj, zad)

    def _generuj(self, zad: dict) -> Tuple[np.ndarray, int]:
        import requests
        import soundfile as sf

        odp = requests.post(self._serwer.rstrip("/") + "/v1/audio/speech", json=zad, timeout=900)
        if odp.status_code != 200 or not odp.content.startswith(b"RIFF"):
            raise RuntimeError(f"audio.cpp: HTTP {odp.status_code} {odp.content[:300]!r}")
        # `soundfile`, NIE `wave` — `wave` zna tylko PCM i pada na WAV-ie float32.
        audio, sr = sf.read(io.BytesIO(odp.content), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return audio, sr

    SAMPLE_RATE: ClassVar[int] = OMNIVOICE_SAMPLE_RATE

    async def generate_stream(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ):
        """Kawalki dzwieku jako `np.float32` (mono 24 kHz) — tak jak u voxa:
        `response_format=pcm` + `stream_format=audio`, float32 LE przy naglowku
        `X-Sample-Format: f32`, inaczej s16le."""
        await self.load_model()
        adres = self._serwer
        zad = self._zadanie(text, voice_prompt, language, seed, instruct)
        zad["response_format"] = "pcm"
        zad["stream_format"] = "audio"

        # BEZ LIMITU — jak u voxa (pelna kolejka gubila KONIEC i wieszala generacje).
        kolejka: asyncio.Queue = asyncio.Queue()
        petla = asyncio.get_running_loop()
        KONIEC = object()

        def pobieraj():
            import requests

            reszta = b""
            try:
                with requests.post(adres.rstrip("/") + "/v1/audio/speech",
                                   json=zad, stream=True, timeout=900) as odp:
                    if odp.status_code != 200:
                        raise RuntimeError(f"audio.cpp: HTTP {odp.status_code} {odp.text[:300]}")
                    f32 = odp.headers.get("X-Sample-Format") == "f32"
                    bajtow = 4 if f32 else 2
                    for blok in odp.iter_content(chunk_size=None):
                        if not blok:
                            continue
                        # probka nie moze sie przeciac miedzy blokami HTTP
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

    @property
    def _serwer(self) -> str:
        """Adres silnika audio.cpp (`VOXCPM_SERWER`) — wspolny z voxem."""
        return os.environ.get("VOXCPM_SERWER", "").strip()
