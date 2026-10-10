"""Supertonic 3 (ONNX/CPU) — backend W PROCESIE, model ladowany na zadanie.

⭐ Dlaczego nie kontener (decyzja grruwiego 15.09): Voicebox i tak trzyma torcha,
a onnxruntime dokladany do tego samego procesu kosztuje 14 MB. Kontener
duplikowal interpreter i biblioteki — 554 MB na cos, co tutaj wychodzi 15.
Zmierzone w procesie z torchem:

    import torch                894 MB
    + onnxruntime               908 MB   (+14)
    + import supertonic         909 MB   (+1)
    + ZALADOWANY MODEL         1427 MB   (+518)
    po generacji               1427 MB   (zero narostu)

Stad podzial: biblioteki siedza zawsze, **model wchodzi dopiero w `load_model()`
i schodzi w `unload_model()`** — tak jak u voxa. Offloader ewiktuje go razem z
kazdym innym TTS; kosztuje to RAM, nie VRAM, wiec zdejmowanie jest tanie.

Model nie klonuje glosu: ma dziesiec wbudowanych stylow (M1-M5, F1-F5).
Generator wlasnych stylow (Voice Builder) Supertone zamknal 31.08.2026, a
enkodera glosu nie ma w wydanych wagach — wlasnej barwy tu nie zrobimy.

⚠️ `synthesize` oddaje krotke (audio, dlugosci) o ksztalcie (1, N).
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

import numpy as np

from .base import (
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)

logger = logging.getLogger(__name__)

SUPERTONIC_SAMPLE_RATE = 44100
SUPERTONIC_DEFAULT_VOICE = "M1"

# Wagi na ADACIE, poza cache HuggingFace — stad wlasna galaz statusu
# w `routes/models.py`.
KATALOG_WAG = Path("/klocki/apczan-kutasu/voicebox/voice-models/supertonic/modele/supertonic3")

SUPERTONIC_REQUIRED = (
    "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/duration_predictor.onnx",
    "onnx/vocoder.onnx",
)

# (voice_id, display_name, gender, language)
# Nazwy sa identyfikatorami Supertone — producent nie opublikowal innych.
# Kazdy styl mowi wszystkimi 31 jezykami, wiec pole jezyka zostaje uniwersalne.
SUPERTONIC_VOICES = [
    ("M1", "M1", "male", "multi"),
    ("M2", "M2", "male", "multi"),
    ("M3", "M3", "male", "multi"),
    ("M4", "M4", "male", "multi"),
    ("M5", "M5", "male", "multi"),
    ("F1", "F1", "female", "multi"),
    ("F2", "F2", "female", "multi"),
    ("F3", "F3", "female", "multi"),
    ("F4", "F4", "female", "multi"),
    ("F5", "F5", "female", "multi"),
]

SUPERTONIC_LANGUAGES = [
    "pl", "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el",
    "es", "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt",
    "lv", "nl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
]


def _cfg(nazwa: str, domyslnie: str) -> str:
    return os.environ.get(nazwa, domyslnie).strip()


class SupertonicTTSBackend:
    """Supertonic 3 w procesie Voiceboxa. Model wchodzi na zadanie."""

    def __init__(self):
        self._tts = None
        self._style_cache: dict = {}
        # ⚠️ „default", nie wlasna nazwa wariantu — inaczej offloader nie
        # rozpoznaje wlasnego configu i zdejmuje model, ktory wlasnie stawia.
        self.model_size = "default"

    @property
    def device(self) -> str:
        return "cpu"

    def is_loaded(self) -> bool:
        return self._tts is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return str(KATALOG_WAG)

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return all((KATALOG_WAG / nazwa).is_file() for nazwa in SUPERTONIC_REQUIRED)

    # -------------------------------------------------------------- zycie

    async def load_model(self, model_size: str = "default") -> None:
        if self._tts is not None:
            return
        if not self._is_model_cached():
            raise RuntimeError(f"brak wag Supertonica w {KATALOG_WAG}")
        await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self):
        with model_load_progress("supertonic-3", self._is_model_cached()):
            from supertonic import TTS

            # ⛔ PUSTE = ONNX Runtime dobiera liczbe watkow sam. Nie wpisujemy
            # tu zadnej liczby za niego.
            intra = _cfg("SUPERTONIC_INTRA_OP_THREADS", "")
            inter = _cfg("SUPERTONIC_INTER_OP_THREADS", "")

            self._tts = TTS(
                model_dir=str(KATALOG_WAG),
                auto_download=False,
                intra_op_num_threads=int(intra) if intra else None,
                inter_op_num_threads=int(inter) if inter else None,
            )

        logger.info("Supertonic 3 zaladowany (ONNX, CPU) — +518 MB RAM, VRAM nietkniety")

    def unload_model(self) -> None:
        """Oddaje ~518 MB RAM. Karty nie dotykal, wiec nie ma czego zwalniac."""
        if self._tts is None:
            return
        self._tts = None
        self._style_cache.clear()

        import gc

        gc.collect()
        logger.info("Supertonic 3 odladowany")

    # -------------------------------------------------------------- glosy

    def _styl(self, nazwa: str):
        """Styl to 292 kB JSON-a — wczytany raz zostaje do odladowania."""
        if nazwa not in self._style_cache:
            self._style_cache[nazwa] = self._tts.get_voice_style(voice_name=nazwa)
        return self._style_cache[nazwa]

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """Supertonic nie klonuje — profil klonowany spada na glos domyslny."""
        return {
            "voice_type": "preset",
            "preset_engine": "supertonic",
            "preset_voice_id": SUPERTONIC_DEFAULT_VOICE,
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        return await _combine_voice_prompts(
            audio_paths, reference_texts, sample_rate=SUPERTONIC_SAMPLE_RATE
        )

    # -------------------------------------------------------------- pokretla

    # ⭐ POKRETLA W `ustawienia.env`. ⛔ PUSTE = POLA NIE PRZEKAZUJEMY —
    # biblioteka bierze wtedy swoja domyslna (`total_steps=8`, `speed=1.05`).
    _POKRETLA: ClassVar[tuple] = (
        ("total_steps", "SUPERTONIC_TOTAL_STEPS", int),
        ("speed", "SUPERTONIC_SPEED", float),
    )

    def _parametry(self) -> dict:
        pola = {}
        for nazwa, zmienna, typ in self._POKRETLA:
            wartosc = _cfg(zmienna, "")
            if not wartosc:
                continue
            try:
                pola[nazwa] = typ(wartosc)
            except ValueError:
                logger.warning("%s=%r nie jest liczba — pomijam", zmienna, wartosc)
        return pola

    # -------------------------------------------------------------- mowa

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "pl",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        `seed` przyjmowany dla zgodnosci interfejsu i ignorowany — model jest
        deterministyczny, ten sam tekst daje ten sam przebieg. `instruct` nie ma
        odpowiednika; ekspresja idzie tagami w tekscie.
        """
        await self.load_model()

        glos = (
            voice_prompt.get("preset_voice_id")
            or voice_prompt.get("supertonic_voice")
            or SUPERTONIC_DEFAULT_VOICE
        )
        jezyk = language if language in SUPERTONIC_LANGUAGES else "en"

        def _generuj():
            wynik = self._tts.synthesize(
                text, voice_style=self._styl(glos), lang=jezyk, **self._parametry()
            )
            # (1, N) -> plaski wektor, ktorego oczekuje reszta stacku.
            audio = np.squeeze(np.asarray(wynik[0], dtype=np.float32))
            return audio, SUPERTONIC_SAMPLE_RATE

        return await asyncio.to_thread(_generuj)
