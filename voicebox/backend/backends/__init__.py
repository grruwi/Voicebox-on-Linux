"""
Backend abstraction layer for TTS, STT and LLM.

This fork runs exactly three TTS engines, all outside this process:

    voxcpm      VoxCPM2 on audio.cpp     — the `audiocpp` engine host
    omnivoice   OmniVoice on audio.cpp   — the `audiocpp` engine host
    voxcpm_pt   VoxCPM2 on PyTorch       — the `pytorch` engine host

plus whisper.cpp (the `whisper` container) for STT and LM Studio over HTTP for
transcript refinement. Upstream's in-process engines (Qwen, LuxTTS, Chatterbox,
TADA, Kokoro, MLX) are not wired in: the registry and the factories below do not
know them, so nothing can load or download them — not even through the API.
"""

# Install HF compatibility patches before any backend imports transformers /
# huggingface_hub. The module runs ``patch_transformers_mistral_regex`` at
# import time, which wraps transformers' tokenizer load against the
# unconditional HuggingFace metadata call that otherwise raises on
# HF_HUB_OFFLINE=1 and on network failures.
from ..utils import hf_offline_patch  # noqa: F401

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Protocol, Optional, Tuple, List
from typing_extensions import runtime_checkable
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LLM_MAX_TOKENS = 512
DEFAULT_LLM_TEMPERATURE = 0.7

LANGUAGE_CODE_TO_NAME = {
    "zh": "chinese",
    "en": "english",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "ru": "russian",
    "pt": "portuguese",
    "es": "spanish",
    "it": "italian",
}

# ⛔ BASE AND SMALL REMOVED ON PURPOSE (16.08, grruwi's call). They cannot handle
# Polish — their transcripts are garbage. This list is ALSO the validation:
# `routes/transcription.py` rejects a size outside it, so removing them really
# blocks loading, not just hides them from the list.
WHISPER_HF_REPOS = {
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
    "turbo": "openai/whisper-large-v3-turbo",
}


@dataclass
class ModelConfig:
    """Declarative config for a downloadable model variant."""

    model_name: str  # e.g. "voxcpm2", "whisper-large"
    display_name: str  # e.g. "VoxCPM2 (Multilingual, 48 kHz)"
    engine: str  # e.g. "voxcpm", "whisper"
    hf_repo_id: str  # e.g. "openbmb/VoxCPM2"
    model_size: str = "default"
    size_mb: int = 0
    needs_trim: bool = False
    retries_runaway: bool = False
    supports_instruct: bool = False
    languages: list[str] = field(default_factory=lambda: ["en"])


@runtime_checkable
class TTSBackend(Protocol):
    """Protocol for TTS backend implementations."""

    # Each backend class should define MODEL_CONFIGS as a class variable:
    # MODEL_CONFIGS: list[ModelConfig]

    async def load_model(self, model_size: str) -> None:
        """Load TTS model."""
        ...

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        Returns:
            Tuple of (voice_prompt_dict, was_cached)
        """
        ...

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        """
        Combine multiple voice prompts.

        Returns:
            Tuple of (combined_audio_array, combined_text)
        """
        ...

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text.

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        ...

    def unload_model(self) -> None:
        """Unload model to free memory."""
        ...

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...

    def _get_model_path(self, model_size: str) -> str:
        """
        Get model path for a given size.

        Returns:
            Model path or HuggingFace Hub ID
        """
        ...


@runtime_checkable
class STTBackend(Protocol):
    """Protocol for STT (Speech-to-Text) backend implementations."""

    async def load_model(self, model_size: str) -> None:
        """Load STT model."""
        ...

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio to text.

        Returns:
            Transcribed text
        """
        ...

    def unload_model(self) -> None:
        """Unload model to free memory."""
        ...

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for local LLM (chat/completion) backend implementations."""

    async def load_model(self, model_size: str) -> None:
        """Load LLM weights and tokenizer."""
        ...

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        model_size: Optional[str] = None,
        examples: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """Run a single-turn chat completion and return the assistant reply.

        ``examples`` is an optional list of ``(user, assistant)`` pairs
        prepended to the conversation as proper chat turns — small models
        pattern-match on inline system-prompt examples (echoing them
        verbatim for unrelated inputs), but treat structured turns as
        data and generalize instead. Used by the refinement service.
        """
        ...

    def unload_model(self) -> None:
        ...

    def is_loaded(self) -> bool:
        ...


# Global backend instances
_tts_backends: dict[str, TTSBackend] = {}
_tts_backends_lock = threading.Lock()
_stt_backend: Optional[STTBackend] = None
_llm_backend: Optional[LLMBackend] = None

DEFAULT_TTS_ENGINE = "voxcpm"

# The only TTS engines this build knows. Display names for the GUI.
TTS_ENGINES = {
    "voxcpm": "VoxCPM2",
    "omnivoice": "OmniVoice",
    "voxcpm_pt": "VoxCPM2 PyTorch",
}

LLM_ENGINES = {
    # Refinement over HTTP to LM Studio. The weights stay THERE, so Voicebox gives
    # away not a single megabyte of the VRAM that belongs to the voice engines.
    "lmstudio_llm": "LM Studio (HTTP)",
}

_COMMON_LANGUAGES = [
    "pl", "en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es",
    "it", "ar", "nl", "tr", "vi", "id", "th", "cs", "hu", "ro",
]


def _get_tts_configs() -> list[ModelConfig]:
    """The TTS models. Their weights live on disk next to the engine containers
    (`download-models.sh`), not in the HuggingFace cache — `hf_repo_id` is only
    where they come from."""
    return [
        ModelConfig(
            model_name="voxcpm2",
            display_name="VoxCPM2 (Multilingual, 48 kHz)",
            engine="voxcpm",
            hf_repo_id="openbmb/VoxCPM2",
            size_mb=4700,
            languages=_COMMON_LANGUAGES,
        ),
        # OmniVoice (k2-fsa) — THE SAME audio.cpp engine as Vox (the `audiocpp` container),
        # one model at a time. Upstream's ready GGUFs in <models>/audiocpp/weights/omnivoice
        # (the q8|f16 variant from the panel).
        ModelConfig(
            model_name="omnivoice",
            display_name="OmniVoice (600+ languages, 24 kHz)",
            engine="omnivoice",
            hf_repo_id="audio-cpp/audio.cpp-gguf",
            size_mb=1288,
            languages=_COMMON_LANGUAGES,
        ),
        # VoxCPM2 on PyTorch — THE SAME WEIGHTS as `voxcpm2` above, another engine.
        # A separate entry, so it can be switched in the GUI and the difference heard.
        ModelConfig(
            model_name="voxcpm2-pytorch",
            display_name="VoxCPM2 PyTorch (Multilingual, 48 kHz)",
            engine="voxcpm_pt",
            hf_repo_id="openbmb/VoxCPM2",
            size_mb=4728,
            languages=_COMMON_LANGUAGES,
        ),
    ]


def _get_whisper_configs() -> list[ModelConfig]:
    """Whisper models for whisper.cpp (`services/whisper_host.py`). One entry per model —
    the quantization (q5/q8) and the device (GPU/CPU) are picked in the panel.

    ⛔ base and small were removed on purpose (16.08, grruwi's call): they cannot handle
    Polish, their transcripts are garbage."""
    return [
        ModelConfig(
            model_name="whisper-medium",
            display_name="Whisper Medium (ggml)",
            engine="whisper",
            hf_repo_id="ggerganov/whisper.cpp",
            model_size="medium",
        ),
        ModelConfig(
            model_name="whisper-large",
            display_name="Whisper Large v3 (ggml)",
            engine="whisper",
            hf_repo_id="ggerganov/whisper.cpp",
            model_size="large",
        ),
        ModelConfig(
            model_name="whisper-turbo",
            display_name="Whisper Large v3 Turbo (ggml)",
            engine="whisper",
            hf_repo_id="ggerganov/whisper.cpp",
            model_size="turbo",
        ),
    ]


def get_all_model_configs() -> list[ModelConfig]:
    """Return the full list of model configs (TTS + STT). LM Studio has no model of ours."""
    return _get_tts_configs() + _get_whisper_configs()


def get_tts_model_configs() -> list[ModelConfig]:
    """Return only TTS model configs."""
    return _get_tts_configs()


def get_llm_model_configs() -> list[ModelConfig]:
    """LLM model configs — none: refinement uses whatever LM Studio has loaded."""
    return []


def get_stt_model_configs() -> list[ModelConfig]:
    """Return only STT (Whisper) model configs."""
    return _get_whisper_configs()


# Lookup helpers — these replace the if/elif chains in main.py


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Look up a model config by model_name."""
    for cfg in get_all_model_configs():
        if cfg.model_name == model_name:
            return cfg
    return None


def engine_needs_trim(engine: str) -> bool:
    """Whether this engine's output should be run through trim_tts_output."""
    for cfg in get_tts_model_configs():
        if cfg.engine == engine:
            return cfg.needs_trim
    return False


def engine_retries_runaway(engine: str) -> bool:
    """Whether unstable output should be retried in smaller chunks."""
    for cfg in get_tts_model_configs():
        if cfg.engine == engine:
            return cfg.retries_runaway
    return False


def engine_has_model_sizes(engine: str) -> bool:
    """Whether this engine supports multiple model sizes (none of ours does)."""
    configs = [c for c in get_tts_model_configs() if c.engine == engine]
    return len(configs) > 1


class WeightsMissingError(RuntimeError):
    """The weights are not on disk. Nothing is ever downloaded on the fly — the
    message says how to get them."""


def _weights_hint(engine: str) -> str:
    display = TTS_ENGINES.get(engine, engine)
    return f"{display} weights are not on disk — run download-models.sh (nothing is downloaded on the fly)"


async def load_engine_model(engine: str, model_size: str = "default") -> None:
    """Load a model for the given engine."""
    backend = get_tts_backend_for_engine(engine)
    if not backend._is_model_cached(model_size):
        raise WeightsMissingError(_weights_hint(engine))

    # Make room BEFORE loading, not after. Every TTS engine has its own instance
    # in _tts_backends, so without this switching voices leaves the previous model
    # in VRAM until someone cuts it. A no-op when nothing else sits in memory.
    #
    # On purpose WITHOUT try/except: when freeing fails, EvictionError has to stop
    # the load. The machine is used remotely — a failed generation is survivable,
    # an OOM cuts off access to the whole computer.
    evict_other_models(engine, model_size)
    await backend.load_model(model_size)


async def ensure_model_cached_or_raise(engine: str, model_size: str = "default") -> None:
    """Raise HTTP 400 when the engine's weights are not on disk. Used by the streaming endpoint."""
    from fastapi import HTTPException

    if not get_tts_backend_for_engine(engine)._is_model_cached(model_size):
        raise HTTPException(status_code=400, detail=_weights_hint(engine))


def unload_model_by_config(config: ModelConfig) -> bool:
    """Unload a model given its config. Returns True if it was loaded, False otherwise."""
    from ..services import transcribe

    if config.engine == "whisper":
        whisper_model = transcribe.get_whisper_model()
        if whisper_model.is_loaded() and whisper_model.model_size == config.model_size:
            transcribe.unload_whisper_model()
            return True
        return False

    # The weights live in an engine host, so unloading means stopping that model's
    # process — `unload_model()` does it.
    backend = get_tts_backend_for_engine(config.engine)
    if not backend.is_loaded():
        return False
    backend.unload_model()
    return True


# ─── Offloader: one model at a time ────────────────────────────────────────
# The backend loads models but never unloads any on its own. Switching engines would
# leave the previous one in VRAM, so memory would grow until someone from outside
# cut it. The rules:
#
#     a TTS model unloads every engine host except its own (whatever it holds)
#     whisper unloads whispers
#
# Whisper stays out of the hosts: speaking does not touch the dictation ear.
# The variant being asked for is spared, so reusing the same model reloads nothing.

_STT_ENGINES = frozenset({"whisper"})
_LLM_ENGINES = frozenset({"lmstudio_llm"})


def engine_category(engine: str) -> str:
    """The engine category: 'stt', 'llm' or 'tts' (the default)."""
    if engine in _STT_ENGINES:
        return "stt"
    if engine in _LLM_ENGINES:
        return "llm"
    return "tts"


class EvictionError(RuntimeError):
    """Memory was not freed. Blocks loading the next model."""


def _same_size(a: str | None, b: str | None) -> bool:
    """Whether two `model_size` values point at the same model variant.

    Engines without variants have `"default"` in the registry, but the generation route
    passes `None` for them (`generations.py`: `... if engine_has_model_sizes(engine) else None`).
    A bare `==` calls the two different, so the Offloader does not recognize its OWN model
    and unloads it right before use — then loads it back 2 ms later.

    The cost of that one inequality, measured 2026-08-02: 21 cycles of
    "unloaded chatterbox-tts (room for chatterbox/None)" in one session,
    ~7.4 s of reloading glued to every utterance.
    """
    return (a or "default") == (b or "default")


def evict_other_models(engine: str, model_size: str = "default") -> list[str]:
    """
    Make room for `engine` before it loads. Returns what actually went down.

    Nothing here knows any particular engine:
      - TTS: every engine host (`services/engine_host.py`) other than the one `engine`
        lives in is unloaded and checked, whatever it holds. Inside its own host, the
        host's `/load` swaps the model itself. A new model in an existing host, or a new
        host in `VOICEBOX_ENGINE_HOSTS`, is covered with no change here.
      - STT: every other loaded whisper goes.

    ⚠️ THIS IS A CRITICAL PATH, NOT COSMETICS. The machine is used REMOTELY: an OOM or a
    hang is not a broken generation, it is losing access to the whole computer until
    someone is physically back at it. So every failure is hard: if room was not made,
    the next model does NOT load next to the old one. A loud refusal is cheap, silently
    piling up VRAM costs the machine.

    After every unload we VERIFY the model is really gone — an unload may report success
    and still leave the weights in memory.
    """
    from ..services import engine_host

    category = engine_category(engine)
    freed: list[str] = []
    failed: list[str] = []

    if category == "tts":
        keep = getattr(get_tts_backend_for_engine(engine), "ENGINE_HOST", None)
        try:
            freed += engine_host.offload_except(keep)
        except engine_host.EngineHostError as e:
            failed.append(str(e))

    if category == "stt":
        for config in _get_whisper_configs():
            if _same_size(config.model_size, model_size):
                continue
            try:
                if not unload_model_by_config(config):
                    continue  # it was not loaded, nothing to talk about
                # Trust, but verify — only absence from memory counts as success.
                if check_model_loaded(config):
                    logger.error("Offloader: %s reported unloaded but still sits in memory", config.model_name)
                    failed.append(config.model_name)
                else:
                    freed.append(config.model_name)
            except Exception:
                logger.exception("Offloader: unloading %s failed", config.model_name)
                failed.append(config.model_name)

    if freed:
        logger.info("Offloader: unloaded %s (room for %s/%s)", ", ".join(freed), engine, model_size)

    if failed:
        raise EvictionError(
            "Memory not freed by: " + ", ".join(failed)
            + f" — holding off loading {engine}/{model_size}, so VRAM does not pile up"
        )

    return freed


def check_model_loaded(config: ModelConfig) -> bool:
    """Check if a model is currently loaded."""
    from ..services import transcribe

    try:
        if config.engine == "whisper":
            whisper_model = transcribe.get_whisper_model()
            return whisper_model.is_loaded() and getattr(whisper_model, "model_size", None) == config.model_size
        # The engine host knows what it holds (Vox and OmniVoice share one engine, so it
        # is asked about the model, not about a port).
        return get_tts_backend_for_engine(config.engine).is_loaded()
    except Exception:
        return False


def get_model_load_func(config: ModelConfig):
    """Return a callable that loads the model (from disk — nothing is downloaded)."""
    from ..services import transcribe

    if config.engine == "whisper":
        return lambda: transcribe.get_whisper_model().load_model(config.model_size)

    async def load():
        await load_engine_model(config.engine, config.model_size)

    return load


def get_tts_backend() -> TTSBackend:
    """The default TTS backend (VoxCPM2 on audio.cpp)."""
    return get_tts_backend_for_engine(DEFAULT_TTS_ENGINE)


def get_tts_backend_for_engine(engine: str) -> TTSBackend:
    """
    Get or create the TTS backend for the given engine.

    Args:
        engine: one of TTS_ENGINES

    Raises:
        ValueError: for any other engine — upstream's engines are not wired in.
    """
    if engine in _tts_backends:
        return _tts_backends[engine]

    with _tts_backends_lock:
        if engine in _tts_backends:
            return _tts_backends[engine]

        if engine == "voxcpm":
            from .voxcpm_gguf_backend import VoxCPMGGUFBackend

            backend = VoxCPMGGUFBackend()
        elif engine == "omnivoice":
            from .omnivoice_backend import OmniVoiceBackend

            backend = OmniVoiceBackend()
        elif engine == "voxcpm_pt":
            from .voxcpm_pt_backend import VoxCPMPyTorchBackend

            backend = VoxCPMPyTorchBackend()
        else:
            raise ValueError(f"Unknown TTS engine: {engine}. Supported: {list(TTS_ENGINES.keys())}")

        _tts_backends[engine] = backend
        return backend


def get_stt_backend() -> STTBackend:
    """The STT backend: whisper.cpp in the `whisper` container."""
    global _stt_backend

    if _stt_backend is None:
        from .whispercpp_backend import WhisperCppSTTBackend

        _stt_backend = WhisperCppSTTBackend()
    return _stt_backend


def get_llm_backend() -> LLMBackend:
    """The refinement backend: LM Studio over HTTP."""
    global _llm_backend

    if _llm_backend is None:
        from .lmstudio_llm_backend import LMStudioLLMBackend

        _llm_backend = LMStudioLLMBackend()
    return _llm_backend


def get_llm_backend_for_engine(engine: str) -> LLMBackend:
    if engine not in LLM_ENGINES:
        raise ValueError(f"Unknown LLM engine: {engine}. Supported: {list(LLM_ENGINES.keys())}")
    return get_llm_backend()


def reset_backends():
    """Reset backend instances (useful for testing)."""
    global _stt_backend, _llm_backend
    _tts_backends.clear()
    _stt_backend = None
    _llm_backend = None
