"""
STT przez faster-whisper (CTranslate2) — wagi w procesie Voicebota, na karcie.

Po co, skoro obok stal whisper.cpp: bo to byl OSOBNY serwer, osobny port i
osobne `whisper up`. Tutaj model jest zwyklym modelem Voicebota — widac go w
panelu, zrzuca sie go z karty tak samo jak silnik TTS, i nie ma czego
podnosic po reboocie.

⚠️ `device="cuda"` to NIE pomylka — CTranslate2 zbudowany z `WITH_HIP=ON`
trzyma nazwe "cuda" dla warstwy GPU. Karta gfx1100 wchodzi tedy.

⚠️ `float32` na tej sciezce konczy sie SIGSEGV (zmierzone 01.09). Wagi na
dysku sa int8 i tak maja zostac.
"""

import asyncio
import gc
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

KATALOG_MODELI = Path(
    os.environ.get(
        "FASTER_WHISPER_DIR",
        Path.home() / ".local" / "share" / "whisper-models",
    )
)

# Nazwy po lewej to te, ktore Voicebox waliduje w trasie transkrypcji
# (WHISPER_HF_REPOS). Po prawej katalog z wagami CTranslate2.
KATALOGI = {
    "large": "ct2-large-v3-int8",
}


class FasterWhisperSTTBackend:
    """Whisper na CTranslate2/HIP. Jeden model na raz, zwalnialny."""

    def __init__(self, model_size: str = "large"):
        self.model_size = model_size
        self._model = None
        self._zaladowany_rozmiar: Optional[str] = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def _get_model_path(self, model_size: Optional[str] = None) -> str:
        rozmiar = model_size or self.model_size
        katalog = KATALOGI.get(rozmiar)
        if katalog is None:
            raise RuntimeError(
                f"nie ma wag CTranslate2 dla '{rozmiar}'. Mam: {', '.join(KATALOGI)}."
            )
        return str(KATALOG_MODELI / katalog)

    def _is_model_cached(self, model_size: str = "large") -> bool:
        katalog = KATALOGI.get(model_size)
        if katalog is None:
            return False
        return (KATALOG_MODELI / katalog / "model.bin").exists()

    async def load_model(self, model_size: Optional[str] = None) -> None:
        rozmiar = model_size or self.model_size
        if self._model is not None and self._zaladowany_rozmiar == rozmiar:
            return
        await asyncio.to_thread(self._zaladuj, rozmiar)

    def _zaladuj(self, rozmiar: str) -> None:
        from faster_whisper import WhisperModel

        sciezka = self._get_model_path(rozmiar)
        if not Path(sciezka, "model.bin").exists():
            raise RuntimeError(f"brak wag pod {sciezka}")

        if self._model is not None:
            self._zwolnij()

        logger.info("faster-whisper: laduje %s (int8, GPU)", sciezka)
        self._model = WhisperModel(sciezka, device="cuda", compute_type="int8")
        self._zaladowany_rozmiar = rozmiar
        self.model_size = rozmiar

    def _zwolnij(self) -> None:
        self._model = None
        self._zaladowany_rozmiar = None
        gc.collect()

    def unload_model(self) -> None:
        if self._model is None:
            return
        logger.info("faster-whisper: zrzucam wagi z karty")
        self._zwolnij()

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        await self.load_model(model_size)
        return await asyncio.to_thread(self._transkrybuj, audio_path, language, prompt)

    def _transkrybuj(
        self, audio_path: str, language: Optional[str], prompt: Optional[str]
    ) -> str:
        if not Path(audio_path).exists():
            raise RuntimeError(f"nie ma pliku: {audio_path}")

        # Wszystko poza jezykiem i podpowiedzia zostaje domyslne faster-whispera.
        opcje = {}
        if language:
            opcje["language"] = language
        if prompt:
            opcje["initial_prompt"] = prompt

        segmenty, _info = self._model.transcribe(audio_path, **opcje)
        # `transcribe` zwraca generator — tekst powstaje dopiero przy iteracji.
        return " ".join(s.text.strip() for s in segmenty).strip()
