"""
OmniVoice (k2-fsa, 600+ languages) on audio.cpp — a backend that LOADS NO WEIGHTS.

The same engine as Vox (`audiocpp_server` in the `audiocpp` container,
`services/audiocpp_host.py`), another model family. ⭐ ONE MODEL AT A TIME (grruwi's call
10.10): loading OmniVoice = restarting the engine with it alone, Vox goes — and the other
way round. Voicebox's Offloader unloads the other engine hosts before loading; the
container's manager enforces one model a second time, at the process level.

Weights: upstream's ready GGUFs (HF `audio-cpp/audio.cpp-gguf`, `OmniVoice-GGUF/`)
in `<models>/audiocpp/weights/omnivoice/`, the variant from the panel (`omni_weights`: q8 | f16).

Cloning: the profile sample (`voice_ref`, a path) + its transcript (`reference_text`).
No sample: `instruct` = voice design ("female, young adult, moderate pitch"),
neither = an automatic voice.

⭐ ITS OWN SLIDERS (the OmniVoice tab in settings, 11.10): `omni_cfg` → `guidance_scale`,
`omni_steps` → `num_inference_steps`, `omni_seed` → `seed`. Empty = upstream's default
(guidance 2.0, 32 steps, a random seed). `omni_flash` = `omnivoice.perf_mode=flash_attention`
(a session option, a change = an engine restart at the next load) — without it attention
computes the full T×T matrix (T = sample + text + speech, x2 for CFG) and VRAM grows with
the longest utterance. The other knobs (duration_sec, shift, denoise, temperatures, ...) =
upstream's defaults. OmniVoice is a non-causal model: it computes the whole utterance at
once and guesses the length up front (from the sample's pace), so the stream = one chunk at
the end, unless `text_chunk_size` is set (upstream's pseudo-streaming).
"""

import asyncio
import io
import logging
import os
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

import numpy as np

from ..services import audiocpp_host
from .base import combine_voice_prompts as _combine_voice_prompts

logger = logging.getLogger(__name__)

OMNIVOICE_SAMPLE_RATE = 24000

# Pseudo-stream (the OmniVoice tab, 11.10 — the counterpart of omnivoice.cpp's `--stream-by-line`):
# the text is cut into chunks, each computed and sent separately, so the first sound comes
# after the first chunk instead of after the whole thing. "none" = nothing sent (upstream
# itself only cuts above 30 s of speech). 160 characters = the value from upstream's stream examples.
_CHUNK_MODES = {"lines": "endline", "sentences": "tag_aware"}
_CHUNK_CHARS = 160


class OmniVoiceBackend:
    """OmniVoice on audio.cpp. The weights live in the engine (the `audiocpp` container), not here."""

    # The Offloader unloads every OTHER engine host before this one loads.
    ENGINE_HOST: ClassVar[str] = audiocpp_host.HOST

    def __init__(self):
        self.model_size = "default"

    # ─────────────────────────────────────────────── weights and engine

    @staticmethod
    def _variant() -> str:
        from ..services import audiocpp_settings

        return audiocpp_settings.selected_omni()

    def _file(self) -> Path:
        from ..services import audiocpp_settings

        return audiocpp_host.omni_weights(audiocpp_settings.OMNI_TYPES[self._variant()])

    def is_loaded(self) -> bool:
        return audiocpp_host.state().get("omni") is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return str(self._file())

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return self._file().is_file()

    async def load_model(self, model_size: str = "default") -> None:
        """Loads OmniVoice into the engine (the panel's variant), unless the same one sits there."""
        from ..services import audiocpp_settings

        if not self._engine:
            raise RuntimeError("OmniVoice needs the audio.cpp engine — AUDIOCPP_ENGINE is empty")
        await asyncio.to_thread(audiocpp_host.load_omni, self._variant(), audiocpp_settings.selected_omni_flash())

    def unload_model(self) -> None:
        """Gives VRAM and RAM back — the engine process stops, the container stays."""
        if self.is_loaded() and audiocpp_host.unload():
            logger.info("OmniVoice unloaded from the engine — VRAM given back")

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
        # Joining samples is numpy on the CPU — works the same, the engine does not matter.
        return await _combine_voice_prompts(audio_paths, reference_texts)

    # ─────────────────────────────────────────────── request

    @staticmethod
    def _sliders() -> dict:
        """cfg / steps / seed from the OmniVoice tab. No value = not sent (upstream's default)."""
        result: dict = {}
        try:
            # The MODULE, not the name — `SessionLocal` is only filled in by `init_db()`.
            from ..database import GenerationSettings
            from ..database import session as db_session

            s = db_session.SessionLocal()
            try:
                row = s.query(GenerationSettings).first()
                if row is not None:
                    if getattr(row, "omni_cfg", None) is not None:
                        result["guidance_scale"] = float(row.omni_cfg)
                    if getattr(row, "omni_steps", None) is not None:
                        result["num_inference_steps"] = int(row.omni_steps)
                    if getattr(row, "omni_seed", None) is not None:
                        result["seed"] = int(row.omni_seed)
                    if getattr(row, "omni_speed", None) is not None:
                        result["speed"] = float(row.omni_speed)
                    mode = getattr(row, "omni_chunk_mode", None) or "none"
                    if mode in _CHUNK_MODES:
                        result["text_chunk_mode"] = _CHUNK_MODES[mode]
                        result["text_chunk_size"] = int(getattr(row, "omni_chunk_chars", None) or _CHUNK_CHARS)
            finally:
                s.close()
        except Exception as e:
            logger.debug("OmniVoice sliders not from the database (%s) — upstream's defaults", e)
        return result

    def _request(self, text: str, voice_prompt: dict, language: Optional[str],
                 seed: Optional[int], instruct: Optional[str]) -> dict:
        """The `/v1/audio/speech` body shared by the stream and the whole take."""
        request = {
            "model": "omnivoice",
            "input": text,
            # float32 samples (our patch in `app/server/runtime.cpp`, shared by the families)
            "sample_format": "f32",
            "options": {},
        }
        ref_audio = voice_prompt.get("ref_audio")
        if ref_audio and not Path(ref_audio).exists():
            logger.warning("no reference sample: %s", ref_audio)
            ref_audio = None
        if ref_audio:
            # An ABSOLUTE path — the engine sees /app/data under the same path (the `audiocpp` container).
            request["voice_ref"] = {"type": "path", "path": str(Path(ref_audio).resolve())}
            if voice_prompt.get("ref_text"):
                request["reference_text"] = voice_prompt["ref_text"]
        if instruct:
            request["options"]["instruction"] = instruct
        if language:
            request["options"]["language"] = language
        sliders = self._sliders()
        if "guidance_scale" in sliders:
            request["guidance_scale"] = sliders["guidance_scale"]
        if "num_inference_steps" in sliders:
            request["num_inference_steps"] = sliders["num_inference_steps"]
        if "speed" in sliders:
            # speed = a multiplier when estimating the speech length (the model guesses it from the sample's pace)
            request["options"]["speed"] = sliders["speed"]
        if "text_chunk_mode" in sliders:
            request["options"]["text_chunk_mode"] = sliders["text_chunk_mode"]
            request["options"]["text_chunk_size"] = sliders["text_chunk_size"]
        # Seed: from the call, then from the panel. Empty = the engine keeps drawing from its
        # session RNG (different on every generation).
        chosen = seed if seed is not None else sliders.get("seed")
        if chosen is not None:
            request["seed"] = int(chosen)
        logger.info("[OmniVoice/audio.cpp] weights=%s cfg=%s steps=%s speed=%s chunks=%s/%s seed=%s language=%s | ref=%s instruct=%s",
                    self._variant(), request.get("guidance_scale", "-"), request.get("num_inference_steps", "-"),
                    request["options"].get("speed", "-"), request["options"].get("text_chunk_mode", "-"),
                    request["options"].get("text_chunk_size", "-"),
                    request.get("seed", "-"), language or "auto",
                    "yes" if ref_audio else "no", "yes" if instruct else "no")
        return request

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        await self.load_model()
        request = self._request(text, voice_prompt, language, seed, instruct)
        request["response_format"] = "wav"
        return await asyncio.to_thread(self._generate, request)

    def _generate(self, request: dict) -> Tuple[np.ndarray, int]:
        import requests
        import soundfile as sf

        response = requests.post(self._engine.rstrip("/") + "/v1/audio/speech", json=request, timeout=900)
        if response.status_code != 200 or not response.content.startswith(b"RIFF"):
            raise RuntimeError(f"audio.cpp: HTTP {response.status_code} {response.content[:300]!r}")
        # `soundfile`, NOT `wave` — `wave` only knows PCM and fails on a float32 WAV.
        audio, sr = sf.read(io.BytesIO(response.content), dtype="float32")
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
        """Audio chunks as `np.float32` (mono 24 kHz) — like Vox:
        `response_format=pcm` + `stream_format=audio`, float32 LE with the
        `X-Sample-Format: f32` header, s16le otherwise."""
        await self.load_model()
        engine = self._engine
        request = self._request(text, voice_prompt, language, seed, instruct)
        request["response_format"] = "pcm"
        request["stream_format"] = "audio"

        # NO LIMIT — like Vox (a full queue lost the END and hung the generation).
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        END = object()

        def fetch():
            import requests

            rest = b""
            try:
                with requests.post(engine.rstrip("/") + "/v1/audio/speech",
                                   json=request, stream=True, timeout=900) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f"audio.cpp: HTTP {response.status_code} {response.text[:300]}")
                    f32 = response.headers.get("X-Sample-Format") == "f32"
                    width = 4 if f32 else 2
                    for block in response.iter_content(chunk_size=None):
                        if not block:
                            continue
                        # a sample must not be split between HTTP blocks
                        block = rest + block
                        cut = len(block) % width
                        rest = block[len(block) - cut:] if cut else b""
                        if cut:
                            block = block[:len(block) - cut]
                        if not block:
                            continue
                        if f32:
                            samples = np.frombuffer(block, dtype="<f4").copy()
                        else:
                            samples = np.frombuffer(block, dtype="<i2").astype(np.float32) / 32768.0
                        loop.call_soon_threadsafe(queue.put_nowait, samples)
            except Exception as error:
                loop.call_soon_threadsafe(queue.put_nowait, error)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, END)

        thread_task = asyncio.create_task(asyncio.to_thread(fetch))
        try:
            while True:
                item = await queue.get()
                if item is END:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            await thread_task

    @property
    def _engine(self) -> str:
        """The audio.cpp engine address (`AUDIOCPP_ENGINE`) — shared with Vox."""
        return os.environ.get("AUDIOCPP_ENGINE", "http://audiocpp:17496").strip()
