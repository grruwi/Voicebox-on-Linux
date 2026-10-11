"""
VoxCPM2 on audio.cpp (ggml, Vulkan/HIP) — a backend that LOADS NO WEIGHTS.

The weights live in the `audiocpp` container (`services/audiocpp_host.py`, the
`audiocpp_server` process); only HTTP requests go from here: `/v1/audio/speech` (a raw
PCM stream or a whole WAV). The engine sits empty from Voicebox's start, Vox loads into
it with the first utterance or from the Models screen.

Since 10.10 instead of `llama-tts-server` from llama.cpp-omni. Measured 10.10 on a
9070 XT, Kleks 29 s, 21 steps, CFG 1.8, F16: RTF 0.40, first byte 0.22–0.25 s
(the old Vox: RTF 0.43, 0.12–0.32 s).

⚠️ Panel fields with no counterpart in audio.cpp (the engine does not know them, not sent):
temperature, `stop_on_predictor`, `append_audio_start`, `target_sr`,
stream granularity (`vox_stream_min/every`).
"""

import asyncio
import io
import logging
import os
import random
import re
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

import numpy as np

from ..services import audiocpp_host
from .base import combine_voice_prompts as _combine_voice_prompts

logger = logging.getLogger(__name__)

VOXCPM_SAMPLE_RATE = 48000

# A parenthesis at the start SWITCHES the mode
# from continuation to style control. The 200-character limit is the model's, not ours.
_CONTROL_PREFIX = re.compile(r"^\s*\([^)]{1,200}\)")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


class VoxCPMGGUFBackend:
    """VoxCPM2 on audio.cpp. The weights live in the engine container, not here."""

    # The Offloader unloads every OTHER engine host before this one loads.
    ENGINE_HOST: ClassVar[str] = audiocpp_host.HOST

    def __init__(self):
        # The name stays for the model registry (`backends/__init__.py`).
        self.model_size = "gguf-q8"
        self._announced = False

    # ─────────────────────────────────────────────── weights and engine

    @staticmethod
    def _precision() -> dict:
        """Weight types (`session_options`) from the panel — ONE source:
        `services/audiocpp_settings.weight_options(selected_precision())`, shared with the Reload button."""
        from ..services import audiocpp_settings

        return audiocpp_settings.weight_options(*audiocpp_settings.selected_precision())

    def _load_into_engine(self) -> bool:
        """Loads Vox (precision from the panel), unless the same one already sits there."""
        if not self._engine:
            return False
        audiocpp_host.ensure_running()
        options = self._precision()
        if audiocpp_host.has_vox(options):
            return False
        audiocpp_host.load_vox(options)
        logger.info("Vox loaded into the engine (%s)", options)
        return True

    # ------------------------------------------------------------- protocol

    def is_loaded(self) -> bool:
        # The weights live in the engine, not in this process — ask it.
        return audiocpp_host.state().get("vox") is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return str(audiocpp_host.weights())

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return any(audiocpp_host.weights().glob("*.gguf"))

    async def load_model(self, model_size: str = "default") -> None:
        """Checks there is something to speak with, and loads Vox into the engine.

        ⛔ NOT `and not self.is_loaded()`. A loaded Vox may have ANOTHER precision than
        the panel's — only `_load_into_engine` knows, it compares `session_options` and
        reloads on a mismatch.
        """
        # the engine container's manager checks the binary — only the weights here
        if not audiocpp_host.weights().exists():
            raise RuntimeError(f"no VoxCPM2 weights: {audiocpp_host.weights()}")
        if not self._announced:
            self._announced = True
            logger.info("VoxCPM2/audio.cpp ready: %s", audiocpp_host.weights())
        if self._engine:
            await asyncio.to_thread(self._load_into_engine)

    def unload_model(self) -> None:
        """Gives VRAM back — the engine process stops (also a lazy one), the container stays."""
        self._announced = False
        if audiocpp_host.unload():
            logger.info("audio.cpp engine stopped — VRAM given back")

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

    # ⭐ KNOBS: panel (database) → the env file → fallback.
    #   cfg       — how strictly the model follows the parenthesis. Higher = more exact
    #               but flat, lower = livelier but with more spread.
    #   timesteps — diffusion steps. 30 for REFERENCE SAMPLES, 10 for talking.
    _FALLBACK: ClassVar[dict] = {"cfg": "2.0", "timesteps": "30"}

    def _knobs(self) -> dict:
        """Order: the DATABASE (GUI sliders) → env variables → fallback.

        The GUI wins, because a slider change has to work from the next generation,
        without a backend restart. With no table (an old database) we quietly fall back.
        """
        from_db: dict = {}
        try:
            # ⚠️ The MODULE, not the name. `SessionLocal` is a global filled in only by
            # `init_db()` — `from ..database import SessionLocal` would catch the `None`
            # from import time and fail on the call.
            from ..database import GenerationSettings
            from ..database import session as db_session

            s = db_session.SessionLocal()
            try:
                row = s.query(GenerationSettings).first()
                if row is not None:
                    if row.vox_cfg is not None:
                        from_db["cfg"] = str(row.vox_cfg)
                    if row.vox_timesteps is not None:
                        from_db["timesteps"] = str(row.vox_timesteps)
                    if getattr(row, "vox_seed", None) is not None:
                        from_db["seed"] = str(row.vox_seed)
            finally:
                s.close()
        except Exception as e:  # no column, no row, anything
            logger.debug("knobs not from the database (%s) — using the environment", e)

        return {
            "cfg": from_db.get("cfg") or _env("VOXCPM_CFG", self._FALLBACK["cfg"]),
            "timesteps": from_db.get("timesteps") or _env("VOXCPM_TIMESTEPS", self._FALLBACK["timesteps"]),
            # An empty string = a random seed (see `_request`).
            "seed": from_db.get("seed") or _env("VOXCPM_SEED", ""),
            # ⭐ No panel field — only the env file. EMPTY = the engine uses its own.
            #   max_steps → `max_tokens`: the step (patch) ceiling. Engine: the model's max_length.
            #   min_steps → `min_tokens`: how many steps before it may stop. Engine: 2.
            "max_steps": _env("VOXCPM_MAX_STEPS", ""),
            "min_steps": _env("VOXCPM_MIN_STEPS", ""),
        }

    def _request(self, text: str, ref_audio: Optional[str], ref_text: Optional[str],
                 seed: Optional[int]) -> dict:
        """The `/v1/audio/speech` body shared by the stream and the whole take."""
        styled = bool(_CONTROL_PREFIX.match(text))
        k = self._knobs()
        # ⛔ AN EMPTY FIELD MUST GIVE A RANDOM SEED, NOT NO SEED — a fixed seed kills
        # "generate again, maybe it comes out better".
        chosen = seed if seed is not None else k["seed"]
        if chosen == "":
            chosen = random.randrange(1, 2**31)
        request = {
            "model": "voxcpm2",
            "input": text,
            "guidance_scale": float(k["cfg"]),
            "num_inference_steps": int(k["timesteps"]),
            "seed": int(chosen),
            # The model is in `streaming` mode — audio.cpp then rejects EVERY request
            # without `retry_badcase=false`, also the whole-WAV one (retrying a bad case
            # exists only in offline mode).
            "options": {"retry_badcase": False},
            # ⭐ float32 samples straight from AudioVAE, no 16-bit quantization (our
            # patch in `app/server/parallel_runtime.cpp`, 10.10). The server confirms the
            # format with the `X-Sample-Format` header.
            "sample_format": "f32",
        }
        if k["max_steps"]:
            request["max_tokens"] = int(k["max_steps"])
        if k["min_steps"]:
            request["options"]["min_tokens"] = int(k["min_steps"])
        if ref_audio:
            # A path, not base64: the engine container mounts /app/data (base64 has a
            # 5 MiB limit). ⚠️ ABSOLUTE — a relative one the server resolves from its
            # config directory (/tmp), not from the backend's CWD.
            request["voice_ref"] = {"type": "path", "path": str(Path(ref_audio).resolve())}
            # ⭐ Transcript = CONTINUATION mode (the same sample as the prompt; our patch
            # in `voxcpm2/session.cpp`). Only WITHOUT a parenthesis — upstream forbids
            # combining a style instruction with prompt_text.
            if ref_text and not styled:
                request["reference_text"] = ref_text
        logger.info(
            "[VoxCPM2/audio.cpp] cfg=%s steps=%s seed=%s | ref=%s mode=%s",
            request["guidance_scale"], request["num_inference_steps"], request["seed"],
            "yes" if ref_audio else "no",
            "style" if styled else ("continuation" if "reference_text" in request else "timbre"),
        )
        return request

    @staticmethod
    def _sample(voice_prompt: dict) -> Tuple[Optional[str], Optional[str]]:
        ref_audio = voice_prompt.get("ref_audio")
        if ref_audio and not Path(ref_audio).exists():
            logger.warning("no reference sample: %s", ref_audio)
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
        ref_audio, ref_text = self._sample(voice_prompt)
        request = self._request(text, ref_audio, ref_text, seed)
        request["response_format"] = "wav"
        return await asyncio.to_thread(self._generate, request)

    def _generate(self, request: dict) -> Tuple[np.ndarray, int]:
        import requests

        if not self._engine:
            raise RuntimeError("Vox needs the audio.cpp engine — AUDIOCPP_ENGINE is empty")
        response = requests.post(self._engine.rstrip("/") + "/v1/audio/speech", json=request, timeout=900)
        if response.status_code != 200 or not response.content.startswith(b"RIFF"):
            raise RuntimeError(f"audio.cpp: HTTP {response.status_code} {response.content[:300]!r}")
        # ⚠️ `soundfile`, NOT `wave` — `wave` only knows PCM and fails on a float32 WAV
        # (WAVE_FORMAT_IEEE_FLOAT).
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(response.content), dtype="float32")
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
        """Audio chunks as they are made, as `np.float32`.

        `response_format=pcm` + `stream_format=audio` = raw mono 48 kHz samples over
        chunked encoding, no WAV header: float32 LE with `sample_format=f32` (header
        `X-Sample-Format: f32`), s16le otherwise.
        """
        await self.load_model()

        engine = self._engine
        if not engine:
            raise RuntimeError("streaming needs the audio.cpp engine — AUDIOCPP_ENGINE is empty")

        ref_audio, ref_text = self._sample(voice_prompt)
        request = self._request(text, ref_audio, ref_text, seed)
        request["response_format"] = "pcm"
        request["stream_format"] = "audio"

        # ⛔ NO LIMIT (30.09). The thread puts via `call_soon_threadsafe(put_nowait)`, so a
        # full queue does NOT slow the thread down, it raises QueueFull in the loop — a chunk
        # or the END gets lost and the generation hangs forever, with the whole job queue
        # behind it. The whole utterance in RAM is a few MB.
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        END = object()

        def fetch():
            """Blocking read of the HTTP stream in a separate thread."""
            import requests

            rest = b""
            try:
                with requests.post(engine.rstrip("/") + "/v1/audio/speech",
                                   json=request, stream=True, timeout=900) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f"audio.cpp: HTTP {response.status_code} {response.text[:300]}")
                    # No header = a binary from before the f32 patch → s16le, like upstream.
                    f32 = response.headers.get("X-Sample-Format") == "f32"
                    width = 4 if f32 else 2
                    for block in response.iter_content(chunk_size=None):
                        if not block:
                            continue
                        # A sample must not be split between HTTP blocks;
                        # an incomplete tail waits to be joined to the next one.
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

    # ── engine ────────────────────────────────────────────────────────────

    @property
    def _engine(self) -> str:
        """The audio.cpp engine address (`AUDIOCPP_ENGINE`)."""
        return _env("AUDIOCPP_ENGINE", "http://audiocpp:17496")
