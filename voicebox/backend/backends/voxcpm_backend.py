"""
VoxCPM2 TTS backend implementation.

Wraps OpenBMB's VoxCPM2 — tokenizer-free multilingual TTS with native
48 kHz output and two distinct cloning paths:

  * ``reference_wav_path``  — timbre isolated from content (VoxCPM2 only),
    works without a transcript.
  * ``prompt_wav_path`` + ``prompt_text`` — continuation mode, needs the
    transcript but tracks the original delivery more closely.

Both are passed together by default: the reference carries the timbre,
the prompt carries the manner.

⚠️ The ``voxcpm`` package does NOT live in the backend venv (which is
mounted read-only). It ships separately on PYTHONPATH via /app/extra —
see docker-compose.slim.yml. The import therefore fails with a clear
message rather than an ImportError traceback.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

import numpy as np

from .base import (
    is_model_cached,
    get_torch_device,
    empty_device_cache,
    manual_seed,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)

logger = logging.getLogger(__name__)

VOXCPM_HF_REPO = "openbmb/VoxCPM2"

# Voice design / style control: VoxCPM2 czyta opis glosu z NAWIASU na samym
# poczatku tekstu — "(spokojny, lekko szybszy ton)Reszta zdania...". Zadnego
# osobnego pola, zadnych sztywnych tagow: to naturalny jezyk, upstream wymienia
# plec, wiek, ton, emocje i tempo.
#
# ⚠️ Upstream ZABRANIA laczyc control z `prompt_text` (continuation) — to dwie
# sprzeczne instrukcje naraz. Z samym `reference_wav_path` wolno i to jest cala
# sol: barwa mowcy zostaje, opis steruje ekspresja ("controllable cloning").
# Dlatego nawias na wejsciu PRZELACZA tryb zamiast wywalac blad.
_CONTROL_PREFIX = re.compile(r"^\s*\([^)]{1,200}\)")

# Native output rate. VoxCPM2 is 48 kHz end-to-end — no resampling anywhere.
VOXCPM_SAMPLE_RATE = 48000


class VoxCPM2TTSBackend:
    """VoxCPM2 backend for zero-shot voice cloning at 48 kHz."""

    def __init__(self):
        self.model = None
        self.model_size = "default"
        self._device = None
        self._model_load_lock = asyncio.Lock()

    def _get_device(self) -> str:
        return get_torch_device(force_cpu_on_mac=True)

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return VOXCPM_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return is_model_cached(VOXCPM_HF_REPO, required_files=["model.safetensors", "audiovae.pth"])

    async def load_model(self, model_size: str = "default") -> None:
        """Load the VoxCPM2 model."""
        if self.model is not None:
            return
        async with self._model_load_lock:
            if self.model is not None:
                return
            await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self):
        """Synchronous model loading."""
        model_name = "voxcpm2"
        is_cached = self._is_model_cached()

        with model_load_progress(model_name, is_cached):
            try:
                from voxcpm import VoxCPM
            except ImportError as e:
                raise RuntimeError(
                    "Pakiet `voxcpm` nie jest na PYTHONPATH. Zyje poza venv "
                    "(bind-mount /app/extra) — sprawdz docker-compose.slim.yml."
                ) from e

            device = self._get_device()
            self._device = device
            logger.info(f"Loading VoxCPM2 on {device}...")

            # ⛔ NIE WLACZAC `torch.set_float32_matmul_precision("high")`.
            # PyTorch prosi o to w logu, ale na tym modelu PSUJE DZWIEK:
            # zysku zero (zmierzone), a obcieta mantysa kumuluje blad wzdluz
            # sekwencji — glos zaczyna gruby i konczy jak "papierowy komar"
            # (grruwi, 2026-08-11). Ostrzezenie w logu zostaje i tak ma byc.

            # load_denoiser pulls in modelscope, which is not installed —
            # the denoiser only cleans up noisy reference audio anyway.
            # optimize=True: ZMIERZONE 2026-08-11 — torch.compile dziala (po
            # doinstalowaniu gcc), ale nie daje NIC (RTF 0,73 = baseline),
            # a pierwszy przebieg rosnie do 76 s. Zostaje wylaczony.
            self.model = VoxCPM.from_pretrained(
                VOXCPM_HF_REPO,
                load_denoiser=False,
                optimize=False,
                device=device,
            )

        logger.info("VoxCPM2 loaded successfully")

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            device = self._device
            del self.model
            self.model = None
            self._device = None
            empty_device_cache(device)
            logger.info("VoxCPM2 unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        VoxCPM2 reads the reference at generation time, so the prompt only
        carries the path plus its transcript.
        """
        voice_prompt = {
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }
        return voice_prompt, False

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts)

    # Upstream defaults are cfg_value 2.0 / 10 timesteps; measured RTF on
    # gfx1201 was ~0.8-0.95 there, i.e. faster than real time.
    #
    # ⭐ Strojone uchem grruwiego 15.08, stanelo na 2.5 / 8 krokow.
    # ⚠️ TE LICZBY ZALEZA OD PROBKI, nie sa uniwersalne — jego slowa:
    # "rozne glosy sie roznie zachowuja i to zalezy od probki". Na 19 s
    # Kleksa cfg 3.0 i 2.5 przegrywaly z upstreamowym 2.0 (szum, sztywnosc);
    # na czystym glosie z Voice Design cfg 3.0 "zaczynal pierdziec", a 2.5
    # przy zaledwie 8 krokach wyszlo bez zarzutu. Probka z filmu znosi
    # MNIEJ niz probka zsyntezowana. To jest jeden default na wszystkie
    # profile — per-profil trzeba by dopiero dorobic.
    # Measured RTF on gfx1201: ~0.8-0.95, szybciej niz czas rzeczywisty.
    _DEFAULTS: ClassVar[dict] = {
        # Strojone uchem grruwiego. Droga: 2.0/10 -> 2.0/30 (probki) -> 3.0/10
        # -> 2.5/10 -> 1.8/30 (probki Gieni) -> 2.0/10 (16.08, stan biezacy).
        # cfg = jak sztywno model trzyma sie nawiasu: wyzej dokladniej ale
        # plasko, nizej zywiej ale z wiekszym rozrzutem. 30 krokow tylko na
        # czas robienia probek referencyjnych — do gadania wystarcza 10.
        # ⛔ ROBISZ PROBKI REFERENCYJNE → PODNIES NA 30 ZANIM ODPALISZ GENERACJE.
        # grruwi o tym zapomina i prosi, zeby pilnowac za niego. Do gadania 10.
        "cfg_value": 1.5,
        "inference_timesteps": 30,
    }

    # ⛔ BEZ domyslnego seeda. Probowane DWA RAZY (15.08 i 16.08) i dwa razy
    # zdjete: staly seed zabija upstreamowe "generate 1~3 times" — ta sama
    # linijka daje ten sam plik co do bajta (zweryfikowane MD5), wiec powtorka
    # po nieudanej generacji jest bezuzyteczna. Zadanie z jawnym `seed` dziala.

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio using VoxCPM2.

        Args:
            text: Text to synthesize
            voice_prompt: Dict with ref_audio path and ref_text transcript
            language: Ignored — VoxCPM2 infers the language from the text
            seed: Random seed for reproducibility
            instruct: Unused (protocol compatibility)

        Returns:
            Tuple of (audio_array, 48000)
        """
        await self.load_model()

        ref_audio = voice_prompt.get("ref_audio")
        ref_text = voice_prompt.get("ref_text")

        if ref_audio and not Path(ref_audio).exists():
            logger.warning(f"Reference audio not found: {ref_audio}")
            ref_audio = None

        def _generate_sync():
            if seed is not None:
                manual_seed(seed, self._device)

            kwargs = dict(self._DEFAULTS)
            styled = bool(_CONTROL_PREFIX.match(text))

            if ref_audio:
                # Timbre, no transcript needed.
                kwargs["reference_wav_path"] = ref_audio
                # Manner — only valid as a pair, upstream raises otherwise.
                # Dropped when the text carries a style prefix: the reference
                # alone keeps the timbre while the prefix drives delivery.
                if ref_text and not styled:
                    kwargs["prompt_wav_path"] = ref_audio
                    kwargs["prompt_text"] = ref_text

            logger.info(
                f"[VoxCPM2] Generating: ref={'yes' if ref_audio else 'no'}, "
                f"tryb={'styl' if styled else 'kontynuacja'}"
            )

            # normalize=True would import wetext (pynini/openfst), which is
            # not installed. VoxCPM2 handles raw text fine without it.
            wav = self.model.generate(text=text, normalize=False, **kwargs)

            return np.asarray(wav, dtype=np.float32), VOXCPM_SAMPLE_RATE

        return await asyncio.to_thread(_generate_sync)
