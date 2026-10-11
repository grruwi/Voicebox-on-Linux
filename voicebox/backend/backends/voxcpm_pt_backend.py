"""VoxCPM2 on PyTorch/ROCm in the `pytorch` container — the second Vox, next to the ggml one.

⚠️ THIS IS THE SAME MODEL AS `voxcpm`, ONLY ANOTHER ENGINE. Kept apart on purpose, so
they can be switched in the GUI and the difference heard:

    voxcpm     — audio.cpp (Vulkan/HIP), the `audiocpp` container
    voxcpm_pt  — PyTorch/ROCm, the `voxcpm2` worker of the `pytorch` container

PyTorch is also the road to LoRA and fine-tuning; ggml cannot do that.

The worker is OUR file (`pytorch/workers/voxcpm2/server.py`), not upstream: the VoxCPM2
repo on HuggingFace ships weights only, no server code.
⛔ The worker has no default values — a field that was not sent is not passed to `generate()`.
"""

import asyncio
import base64
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

import numpy as np

from ..config import VOICE_MODELS
from ..services import pytorch_host
from .base import combine_voice_prompts as _combine_voice_prompts

logger = logging.getLogger(__name__)

VOXCPM_SAMPLE_RATE = 48000
MODEL = "voxcpm2"
WEIGHTS_DIR = Path(os.environ.get("VOXCPM_PT_WEIGHTS") or VOICE_MODELS / "voxcpm-pt" / "weights")

# A parenthesis at the start of the text = a style description. Same rule as the ggml engine.
_STYLE_PREFIX = re.compile(r"^\s*\([^)]{1,200}\)")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


class VoxCPMPyTorchBackend:
    """VoxCPM2 in the `pytorch` container. The weights live there, not here."""

    # The Offloader unloads every OTHER engine host before this one loads.
    ENGINE_HOST: ClassVar[str] = pytorch_host.HOST

    def __init__(self):
        self.model_size = "default"

    # -------------------------------------------------------------- files

    @property
    def _model(self) -> Path:
        return WEIGHTS_DIR / "model.safetensors"

    @property
    def _audiovae(self) -> Path:
        return WEIGHTS_DIR / "audiovae.pth"

    def _get_model_path(self, model_size: str = "default") -> str:
        return str(WEIGHTS_DIR)

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return self._model.exists() and self._audiovae.exists()

    # -------------------------------------------------------------- knobs

    # ⭐ THE KNOBS LIVE IN THE ENV FILE. ⛔ EMPTY = THE FIELD IS NOT SENT.
    # Separate from `VOXCPM_*` (ggml) on purpose: another engine may need another
    # operating point with the same weights.
    _KNOBS: ClassVar[tuple] = (
        ("cfg_value", "VOXCPM_PT_CFG"),
        ("inference_timesteps", "VOXCPM_PT_TIMESTEPS"),
        ("normalize", "VOXCPM_PT_NORMALIZE"),
        ("retry_badcase", "VOXCPM_PT_RETRY_BADCASE"),
    )

    @staticmethod
    def _to_value(text: str):
        if text.lower() in ("true", "yes", "1"):
            return True
        if text.lower() in ("false", "no", "0"):
            return False
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return text

    @staticmethod
    def _settings_row():
        """⚠️ The MODULE, not the name: `SessionLocal` is only filled in by `init_db()`."""
        from ..database import GenerationSettings
        from ..database import session as db_session

        s = db_session.SessionLocal()
        try:
            return s.query(GenerationSettings).first()
        finally:
            s.close()

    def _from_database(self) -> dict:
        """⭐ SLIDERS FROM THE GUI. The same columns as the ggml engine (`vox_cfg`,
        `vox_timesteps`) — the same model, the same meaning, so one slider drives both."""
        fields: dict = {}
        try:
            row = self._settings_row()
            if row is not None:
                if row.vox_cfg is not None:
                    fields["cfg_value"] = float(row.vox_cfg)
                if row.vox_timesteps is not None:
                    fields["inference_timesteps"] = int(row.vox_timesteps)
                # The seed from the request (API) wins — `_generate` overrides it after `_params()`.
                if row.vox_seed is not None:
                    fields["seed"] = int(row.vox_seed)
        except Exception as error:
            logger.debug("knobs not from the database (%s) — using the environment", error)
        return fields

    def _params(self) -> dict:
        """Order: the env file → the DATABASE (GUI sliders). The slider wins, it has to
        work from the next generation, without a backend restart — like the ggml engine."""
        fields = {}
        for name, variable in self._KNOBS:
            value = _env(variable)
            if not value:
                continue
            # cfg and timesteps are numbers — "1" has to stay a one, not True.
            if name in ("cfg_value", "inference_timesteps"):
                fields[name] = float(value) if name == "cfg_value" else int(value)
            else:
                fields[name] = self._to_value(value)
        fields.update(self._from_database())
        return fields

    def _precision(self) -> dict:
        """The precision set from the panel (`vox_base_lm` / `vox_acoustic`) = the worker's options."""
        base_lm, acoustic = "q8", "f16"
        try:
            row = self._settings_row()
            if row is not None:
                base_lm = (row.vox_base_lm or base_lm).lower()
                acoustic = (row.vox_acoustic or acoustic).lower()
        except Exception as error:
            logger.debug("precision not from the database (%s) — q8/f16", error)
        # BaseLM f32 was cut from the panel (11.10, grruwi's call); the worker knows q8 | f16.
        # ⛔ CUDA graphs OFF (11.10, grruwi's call, "for now"): measured on q8-f16 and
        # f16-f16 at 30 steps — same RTF, same first sound, same load, ~200 MB less VRAM.
        # `"compile": False` would skip torch.compile entirely: load 11 → 6.8 s, but short
        # and medium texts ~15–20% slower (RTF) and the first sound +50 ms (measured 11.10).
        return {"base_lm": base_lm if base_lm in ("q8", "f16") else "q8", "acoustic": acoustic,
                "cuda_graphs": False}

    # -------------------------------------------------------------- life

    def is_loaded(self) -> bool:
        return pytorch_host.has(MODEL)

    async def load_model(self, model_size: str = "default") -> None:
        missing = [str(p) for p in (self._model, self._audiovae) if not p.exists()]
        if missing:
            raise RuntimeError("no VoxCPM2 (PyTorch) weights: " + ", ".join(missing))
        options = self._precision()
        if not pytorch_host.has(MODEL, options):
            # The panel changed the set or nothing is loaded → a new worker (blocking: load + compile).
            await asyncio.to_thread(pytorch_host.load, MODEL, options)

    def reload(self, base_lm: Optional[str] = None, acoustic: Optional[str] = None) -> dict:
        """The "Reload" button on the Vox tab while THIS engine runs. Empty = the panel.
        ⚠️ Blocking (load + compile) — call from a thread pool, not the event loop."""
        options = self._precision()
        options.update({k: v for k, v in (("base_lm", base_lm), ("acoustic", acoustic)) if v})
        return pytorch_host.load(MODEL, options, force=True)

    def unload_model(self) -> None:
        """Gives VRAM back — the worker process goes, the container stays."""
        if pytorch_host.has(MODEL):
            pytorch_host.unload()

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

    # -------------------------------------------------------------- speech

    def _request(self, text: str, voice_prompt: dict, seed: Optional[int]) -> Tuple[dict, bool]:
        ref_audio = voice_prompt.get("ref_audio")
        ref_text = voice_prompt.get("ref_text")
        if ref_audio and not Path(ref_audio).exists():
            logger.warning("no reference sample: %s", ref_audio)
            ref_audio = None

        request = {"text": text}
        request.update(self._params())
        if seed is not None:
            request["seed"] = seed

        if ref_audio:
            with open(ref_audio, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")

            # Timbre — works without a transcript.
            request["reference_wav"] = encoded

            # ⚠️ The way of speaking — valid ONLY together with the transcript, upstream
            # raises on `prompt_wav_path` alone. And it goes away when the text carries
            # a style parenthesis: upstream forbids both instructions at once. The same
            # rule as the ggml engine — see `voxcpm_gguf_backend.py`.
            if ref_text and not _STYLE_PREFIX.match(text):
                request["prompt_wav"] = encoded
                request["prompt_text"] = ref_text
        return request, bool(ref_audio)

    @staticmethod
    def _log(kind: str, request: dict, has_ref: bool, text: str) -> None:
        logger.info(
            "[VoxCPM2/pt] %s %s | ref=%s | mode=%s", kind,
            {k: v for k, v in request.items() if k not in ("text", "reference_wav", "prompt_wav", "prompt_text")},
            "yes" if has_ref else "no",
            "style" if _STYLE_PREFIX.match(text) else "continuation",
        )

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        await self.load_model()
        request, has_ref = self._request(text, voice_prompt, seed)
        self._log("WHOLE", request, has_ref, text)
        return await asyncio.to_thread(self._generate, request)

    def _generate(self, request: dict) -> Tuple[np.ndarray, int]:
        import requests

        response = requests.post(pytorch_host.worker_url() + "/generate", json=request, timeout=900)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(response.content)
            path = Path(f.name)
        try:
            return self._read_wav(path)
        finally:
            path.unlink(missing_ok=True)

    # -------------------------------------------------------------- stream

    # ⭐ The consumer recognizes this engine by `hasattr(backend, "generate_stream")`.
    # An engine without this method takes the old road — no other backend needs changes.
    SAMPLE_RATE: ClassVar[int] = VOXCPM_SAMPLE_RATE

    async def generate_stream(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ):
        """Audio chunks as they are made, as `np.float32` @ 48 kHz.

        ⚠️ THE TOTAL TIME IS THE SAME as `generate()` — the gain is the FIRST SOUND
        after ~0.25 s instead of after the whole thing. It matters where someone listens live.
        ⛔ Effects and level normalization do NOT work on the stream — both need the
        whole take. The consumer has to know it.
        """
        await self.load_model()
        request, has_ref = self._request(text, voice_prompt, seed)
        self._log("STREAM", request, has_ref, text)

        # ⛔ NO LIMIT (30.09). The thread puts via `call_soon_threadsafe(put_nowait)`, so a
        # full queue does NOT slow the thread down, it raises QueueFull in the loop — a chunk
        # or the END gets lost and the generation hangs forever, with the whole job queue
        # behind it. The engine computes ~3x faster than ffplay plays; a long utterance went
        # past 64 chunks. The whole utterance in RAM is a few MB.
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        END = object()

        def fetch():
            """Blocking read of the HTTP stream in a separate thread."""
            import requests

            rest = b""
            try:
                with requests.post(pytorch_host.worker_url() + "/generate/stream",
                                   json=request, stream=True, timeout=900) as response:
                    response.raise_for_status()
                    for block in response.iter_content(chunk_size=None):
                        # A float32 must not be split between HTTP blocks;
                        # an incomplete tail waits to be joined to the next one.
                        block = rest + block
                        cut = len(block) % 4
                        rest = block[len(block) - cut:] if cut else b""
                        if cut:
                            block = block[:len(block) - cut]
                        if block:
                            loop.call_soon_threadsafe(
                                queue.put_nowait, np.frombuffer(block, dtype="<f4").copy()
                            )
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

    @staticmethod
    def _read_wav(path: Path) -> Tuple[np.ndarray, int]:
        import soundfile as sf

        audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), rate
