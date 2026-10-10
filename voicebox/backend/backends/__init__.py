"""
Backend abstraction layer for TTS and STT.

Provides a unified interface for MLX and PyTorch backends,
and a model config registry that eliminates per-engine dispatch maps.
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

from ..utils.platform_detect import get_backend_type

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

# ⛔ BASE I SMALL USUNIETE SWIADOMIE (16.08, polecenie grruwiego). Nie radza
# sobie z polskim — transkrypt z nich nadaje sie do wyrzucenia. Ta lista jest
# JEDNOCZESNIE walidacja: `routes/transcription.py` odrzuca rozmiar spoza niej,
# wiec ich usuniecie realnie blokuje zaladowanie, nie tylko chowa z listy.
# Przywrocenie: backends/__init__.py.bak-whisper-20260816
WHISPER_HF_REPOS = {
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
    "turbo": "openai/whisper-large-v3-turbo",
}


@dataclass
class ModelConfig:
    """Declarative config for a downloadable model variant."""

    model_name: str  # e.g. "luxtts", "chatterbox-tts"
    display_name: str  # e.g. "LuxTTS (Fast, CPU-friendly)"
    engine: str  # e.g. "luxtts", "chatterbox"
    hf_repo_id: str  # e.g. "YatharthS/LuxTTS"
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
_tts_backend: Optional[TTSBackend] = None
_tts_backends: dict[str, TTSBackend] = {}
_tts_backends_lock = threading.Lock()
_stt_backend: Optional[STTBackend] = None
_llm_backends: dict[str, LLMBackend] = {}
_llm_backends_lock = threading.Lock()

# Supported TTS engines — keyed by engine name, value is the backend class import path.
# The factory function uses this for the if/elif chain; the model configs live on the backend classes.
TTS_ENGINES = {
    "qwen": "Qwen TTS",
    "qwen_custom_voice": "Qwen CustomVoice",
    "luxtts": "LuxTTS",
    "chatterbox": "Chatterbox TTS",
    "chatterbox_turbo": "Chatterbox Turbo",
    "tada": "TADA",
    "kokoro": "Kokoro",
    "voxcpm": "VoxCPM2",
    "omnivoice": "OmniVoice",
    "fish_s2": "Fish S2 Pro",
    "fish_s1": "Fish S1-mini",
    "voxcpm_pt": "VoxCPM2 PyTorch",
    "fish_s2_pt": "Fish S2 Pro PyTorch",
    "supertonic": "Supertonic 3",
}

LLM_ENGINES = {
    "qwen_llm": "Qwen3 LLM",
    # Refine przez HTTP do LM Studio. Wagi zostaja u NIEGO, wiec Voicebox
    # nie oddaje ani megabajta VRAM-u, ktory nalezy do VoxCPM.
    "lmstudio_llm": "LM Studio (HTTP)",
}


def _get_non_qwen_tts_configs() -> list[ModelConfig]:
    """Return model configs for non-Qwen TTS engines.

    These are static — no backend-type branching needed.

    ⛔ 30.08 — obce silniki TTS wyciete (Qwen TTS x2, Qwen CustomVoice x2,
    LuxTTS, Chatterbox x2, TADA x2, Kokoro). Nie sa zainstalowane i nie maja
    byc; pokazywanie ich w GUI bylo dziura. Kopia sprzed ciecia:
    ~/Dokumenty/backupy-klodzio/backends__init__.py.przed-wycieciem-silnikow-2026-08-30
    """
    return [
        ModelConfig(
            model_name="voxcpm2",
            display_name="VoxCPM2 (Multilingual, 48 kHz)",
            engine="voxcpm",
            hf_repo_id="openbmb/VoxCPM2",
            size_mb=4700,
            languages=[
                "pl",
                "en",
                "zh",
                "ja",
                "ko",
                "de",
                "fr",
                "ru",
                "pt",
                "es",
                "it",
                "ar",
                "nl",
                "tr",
                "vi",
                "id",
                "th",
                "cs",
                "hu",
                "ro",
            ],
        ),
        # OmniVoice (k2-fsa) — TEN SAM silnik audio.cpp co vox (kontener `audiocpp`),
        # jeden model naraz. Gotowe GGUF-y z upstreamu w voice-models/audiocpp/wagi/omnivoice
        # (wariant q8|f16 z panelu), poza cache HF — status liczy `routes/models.py`.
        ModelConfig(
            model_name="omnivoice",
            display_name="OmniVoice (600+ jezykow, 24 kHz)",
            engine="omnivoice",
            hf_repo_id="audio-cpp/audio.cpp-gguf",
            size_mb=1288,
            languages=[
                "pl",
                "en",
                "zh",
                "ja",
                "ko",
                "de",
                "fr",
                "ru",
                "pt",
                "es",
                "it",
                "ar",
                "nl",
                "tr",
                "vi",
                "id",
                "th",
                "cs",
                "hu",
                "ro",
            ],
        ),
        # VoxCPM2 na PyTorchu — TE SAME WAGI co `voxcpm2` wyzej, inny silnik.
        # Osobny wpis, zeby dalo sie przelaczyc w GUI i uslyszec roznice.
        ModelConfig(
            model_name="voxcpm2-pytorch",
            display_name="VoxCPM2 PyTorch (Multilingual, 48 kHz, Docker)",
            engine="voxcpm_pt",
            hf_repo_id="openbmb/VoxCPM2",
            size_mb=4728,
            languages=[
                "pl",
                "en",
                "zh",
                "ja",
                "ko",
                "de",
                "fr",
                "ru",
                "pt",
                "es",
                "it",
                "ar",
                "nl",
                "tr",
                "vi",
                "id",
                "th",
                "cs",
                "hu",
                "ro",
            ],
        ),
    ]


def _get_whisper_configs() -> list[ModelConfig]:
    """Return Whisper STT model configs."""
    # ⛔ Wpisy whisper-base i whisper-small usuniete 16.08 — to jest lista, z
    # ktorej GUI oferuje POBRANIE modelu. Zostawione byly jedyna droga, zeby
    # base wrocil na dysk po tym, jak zniknal z WHISPER_HF_REPOS.
    # ⭐ Od 30.09 to pliki ggml dla whisper.cpp (`services/whisper_serwer.py`),
    # nie PyTorch. Jeden wpis na model — kwantyzacje (q5/q8) i urzadzenie
    # (karta/procesor) wybiera panel, jak wagi voxa.
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


def _get_qwen_llm_configs() -> list[ModelConfig]:
    """Return Qwen3 LLM configs with backend-aware HF repo IDs.

    MLX path uses 4-bit community quantizations for Apple Silicon; PyTorch path
    uses the upstream instruct weights.
    """
    backend_type = get_backend_type()
    if backend_type == "mlx":
        repo_0_6 = "mlx-community/Qwen3-0.6B-4bit"
        repo_1_7 = "mlx-community/Qwen3-1.7B-4bit"
        repo_4 = "mlx-community/Qwen3-4B-4bit"
    else:
        repo_0_6 = "Qwen/Qwen3-0.6B"
        repo_1_7 = "Qwen/Qwen3-1.7B"
        repo_4 = "Qwen/Qwen3-4B"

    common_languages = [
        "en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it",
    ]

    return [
        ModelConfig(
            model_name="qwen3-0.6b",
            display_name="Qwen3 0.6B",
            engine="qwen_llm",
            hf_repo_id=repo_0_6,
            model_size="0.6B",
            size_mb=400 if backend_type == "mlx" else 1400,
            languages=common_languages,
        ),
        ModelConfig(
            model_name="qwen3-1.7b",
            display_name="Qwen3 1.7B",
            engine="qwen_llm",
            hf_repo_id=repo_1_7,
            model_size="1.7B",
            size_mb=1100 if backend_type == "mlx" else 3500,
            languages=common_languages,
        ),
        ModelConfig(
            model_name="qwen3-4b",
            display_name="Qwen3 4B",
            engine="qwen_llm",
            hf_repo_id=repo_4,
            model_size="4B",
            size_mb=2500 if backend_type == "mlx" else 8000,
            languages=common_languages,
        ),
    ]


def get_all_model_configs() -> list[ModelConfig]:
    """Return the full list of model configs (TTS + STT + LLM)."""
    return (
        _get_non_qwen_tts_configs()
        + _get_whisper_configs()
        + _get_qwen_llm_configs()
    )


def get_tts_model_configs() -> list[ModelConfig]:
    """Return only TTS model configs."""
    return _get_non_qwen_tts_configs()


def get_llm_model_configs() -> list[ModelConfig]:
    """Return only LLM model configs."""
    return _get_qwen_llm_configs()


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
    """Whether this engine supports multiple model sizes (only Qwen currently)."""
    configs = [c for c in get_tts_model_configs() if c.engine == engine]
    return len(configs) > 1


async def load_engine_model(engine: str, model_size: str = "default") -> None:
    """Load a model for the given engine, handling engines with multiple model sizes."""
    backend = get_tts_backend_for_engine(engine)

    # Zrob miejsce PRZED zaladowaniem, nie po. Kazdy silnik TTS ma wlasna
    # instancje w _tts_backends, wiec bez tego przelaczenie glosu zostawia
    # poprzedni model w VRAM az do zamkniecia aplikacji.
    # No-op gdy nic innego nie siedzi w pamieci.
    #
    # Celowo BEZ try/except: gdy zwolnienie zawiedzie, EvictionError ma przerwac
    # ladowanie. Maszyna pracuje zdalnie — nieudana generacja jest do przezycia,
    # a OOM odcina dostep do calego komputera.
    evict_other_models(engine, model_size)

    if engine in ("qwen", "qwen_custom_voice"):
        await backend.load_model_async(model_size)
    elif engine == "tada":
        await backend.load_model(model_size)
    else:
        # Reszta silnikow ma po jednym wariancie; model_size to zawsze "default".
        await backend.load_model(model_size)


async def ensure_model_cached_or_raise(engine: str, model_size: str = "default") -> None:
    """Check if a model is cached, raise HTTPException if not. Used by streaming endpoint."""
    from fastapi import HTTPException

    backend = get_tts_backend_for_engine(engine)
    cfg = None
    for c in get_tts_model_configs():
        # Ta sama pułapka co w evict_other_models: `None` z trasy generacji kontra
        # `"default"` w rejestrze. Bez zrównania `cfg` zostaje None i sprawdzenie
        # cache'u po cichu nie odbywa się wcale.
        if c.engine == engine and _ten_sam_rozmiar(c.model_size, model_size):
            cfg = c
            break

    if engine in ("qwen", "qwen_custom_voice", "tada"):
        if not backend._is_model_cached(model_size):
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_size} is not downloaded yet. Use /generate to trigger a download.",
            )
    else:
        if not backend._is_model_cached():
            display = cfg.display_name if cfg else engine
            raise HTTPException(
                status_code=400,
                detail=f"{display} model is not downloaded yet. Use /generate to trigger a download.",
            )


def unload_model_by_config(config: ModelConfig) -> bool:
    """Unload a model given its config. Returns True if it was loaded, False otherwise."""
    from . import get_tts_backend_for_engine
    from ..services import tts, transcribe, llm as llm_service

    if config.engine == "whisper":
        whisper_model = transcribe.get_whisper_model()
        if whisper_model.is_loaded() and whisper_model.model_size == config.model_size:
            transcribe.unload_whisper_model()
            return True
        return False

    if config.engine == "qwen_llm":
        backend = llm_service.get_llm_model()
        loaded_size = getattr(backend, "_current_model_size", None) or getattr(backend, "model_size", None)
        if backend.is_loaded() and loaded_size == config.model_size:
            backend.unload_model()
            return True
        return False

    if config.engine == "qwen":
        tts_model = tts.get_tts_model()
        loaded_size = getattr(tts_model, "_current_model_size", None) or getattr(tts_model, "model_size", None)
        if tts_model.is_loaded() and loaded_size == config.model_size:
            tts.unload_tts_model()
            return True
        return False

    if config.engine in ("voxcpm", "fish_s2", "fish_s1", "voxcpm_pt", "fish_s2_pt", "omnivoice"):
        # Wagi leza w osobnym procesie (`llama-tts-server` / `s2 --server`)
        # albo w kontenerze (`fish-s1`, `voxcpm-pt`),
        # wiec odladowanie znaczy ubicie tego procesu — robi to `unload_model()`.
        # Fallback nizej porownuje model_size ("gguf-q8" kontra domyslne
        # "default") i nigdy tu nie trafial: przycisk w GUI nie oddawal nic.
        backend = get_tts_backend_for_engine(config.engine)
        # SUMA, nie wybor. `_serwer_stoi()` puka w ADRES_SERWERA — dla silnikow
        # w SASIEDNICH kontenerach to `127.0.0.1`, czyli localhost TEGO
        # kontenera, gdzie nic nie stoi; miec fisha na karcie i uslyszec
        # „nie stoi" znaczylo zostawic go tam obok nowego modelu.
        # `is_loaded()` pyta `docker ps` i widzi kontener niezaleznie od adresu.
        stoi = backend.is_loaded()
        if not stoi and hasattr(backend, "_serwer_stoi"):
            stoi = backend._serwer_stoi()
        if not stoi:
            return False
        backend.unload_model()
        return True

    if config.engine == "qwen_custom_voice":
        backend = get_tts_backend_for_engine(config.engine)
        loaded_size = getattr(backend, "_current_model_size", None) or getattr(backend, "model_size", None)
        if backend.is_loaded() and loaded_size == config.model_size:
            backend.unload_model()
            return True
        return False

    # All other TTS engines.
    # Compare model_size like every branch above, otherwise asking to unload
    # tada-3b-ml would happily unload a loaded tada-1b instead (they share
    # engine="tada"). Single-variant engines report "default", which is
    # ModelConfig's own default, so they keep working unchanged.
    backend = get_tts_backend_for_engine(config.engine)
    if not backend.is_loaded():
        return False
    loaded_size = getattr(backend, "_current_model_size", None) or getattr(backend, "model_size", None)
    if loaded_size is not None and loaded_size != config.model_size:
        return False
    backend.unload_model()
    return True


# ─── Offloader: jeden model na kategorie ───────────────────────────────────
# Backend laduje modele, ale sam nigdy zadnego nie zdejmuje poza zamknieciem
# aplikacji. Przelaczenie silnika zostawia poprzedni w VRAM, wiec pamiec rosnie
# az ktos z zewnatrz utnie. Regula jest prosta i celowo tepa:
#
#     TTS wysadza wszystkie pozostale TTS
#     whisper wysadza whispery
#     LLM wysadza LLM-y
#
# Kategorie sie nie mieszaja: mowienie nie rusza whispera, transkrypcja nie
# rusza glosu. Wariant o ktory wlasnie prosimy jest oszczedzany, wiec ponowne
# uzycie tego samego modelu niczego nie przeladowuje.

_STT_ENGINES = frozenset({"whisper"})
_LLM_ENGINES = frozenset({"qwen_llm", "lmstudio_llm"})


def engine_category(engine: str) -> str:
    """Kategoria silnika: 'stt', 'llm' albo 'tts' (domyslna)."""
    if engine in _STT_ENGINES:
        return "stt"
    if engine in _LLM_ENGINES:
        return "llm"
    return "tts"


class EvictionError(RuntimeError):
    """Nie udalo sie zwolnic pamieci. Blokuje zaladowanie kolejnego modelu."""


def _ten_sam_rozmiar(a: str | None, b: str | None) -> bool:
    """Czy dwa `model_size` wskazują ten sam wariant modelu.

    Silniki bez wariantów mają w rejestrze `"default"`, ale trasa generacji podaje
    dla nich `None` (`generations.py`: `... if engine_has_model_sizes(engine) else None`).
    Gołe `==` uznaje te dwa za różne, więc offloader nie rozpoznaje WŁASNEGO modelu
    i zdejmuje go tuż przed użyciem — po czym ładuje z powrotem 2 ms później.

    Koszt tej jednej nierówności, zmierzony 2026-08-02: 21 cykli
    „zdjeto chatterbox-tts (miejsce dla chatterbox/None)" w jednej sesji,
    ~7,4 s przeładowania doklejone do każdej wypowiedzi.
    """
    return (a or "default") == (b or "default")


def evict_other_models(engine: str, model_size: str = "default") -> list[str]:
    """
    Zdejmij z pamieci kazdy zaladowany model z tej samej kategorii co podany,
    poza nim samym. Zwraca nazwy tego, co faktycznie zeszlo.

    ⚠️ TO JEST SCIEZKA KRYTYCZNA, NIE KOSMETYKA. Ta maszyna pracuje ZDALNIE:
    OOM albo zwiechа to nie zepsuta generacja, tylko utrata dostepu do calego
    komputera az do fizycznego powrotu do niego. Dlatego kazdy blad jest twardy:
    jesli nie udalo sie zrobic miejsca, NIE ladujemy kolejnego modelu obok
    starego. Glosna odmowa jest tania, cicha kumulacja VRAM kosztuje maszyne.

    Po kazdym zdjeciu WERYFIKUJEMY, ze model faktycznie zszedl — `unload_model`
    moze zwrocic sukces, a mimo to zostawic wagi w cache alokatora.
    """
    keep_category = engine_category(engine)
    freed: list[str] = []
    failed: list[str] = []

    for config in get_all_model_configs():
        if engine_category(config.engine) != keep_category:
            continue
        if config.engine == engine and _ten_sam_rozmiar(config.model_size, model_size):
            continue

        try:
            if not unload_model_by_config(config):
                continue  # nie bylo zaladowane, nie ma o czym mowic
        except Exception:
            logger.exception("Offloader: wyjatek przy zdejmowaniu %s", config.model_name)
            failed.append(config.model_name)
            continue

        # Zaufaj, ale sprawdz — dopiero brak w pamieci liczy sie jako sukces.
        try:
            if check_model_loaded(config):
                logger.error("Offloader: %s zglosil zdjecie, a nadal siedzi w pamieci", config.model_name)
                failed.append(config.model_name)
            else:
                freed.append(config.model_name)
        except Exception:
            logger.exception("Offloader: nie da sie zweryfikowac stanu %s", config.model_name)
            failed.append(config.model_name)

    if freed:
        logger.info("Offloader: zdjeto %s (miejsce dla %s/%s)", ", ".join(freed), engine, model_size)

    if failed:
        raise EvictionError(
            "Nie zwolniono pamieci po: " + ", ".join(failed)
            + f" — wstrzymuje ladowanie {engine}/{model_size}, zeby nie kumulowac VRAM"
        )

    return freed


def check_model_loaded(config: ModelConfig) -> bool:
    """Check if a model is currently loaded."""
    from . import get_tts_backend_for_engine
    from ..services import tts, transcribe, llm as llm_service

    try:
        if config.engine == "whisper":
            whisper_model = transcribe.get_whisper_model()
            return whisper_model.is_loaded() and getattr(whisper_model, "model_size", None) == config.model_size

        if config.engine == "qwen_llm":
            backend = llm_service.get_llm_model()
            loaded_size = getattr(backend, "_current_model_size", None) or getattr(backend, "model_size", None)
            return backend.is_loaded() and loaded_size == config.model_size

        if config.engine == "qwen":
            tts_model = tts.get_tts_model()
            loaded_size = getattr(tts_model, "_current_model_size", None) or getattr(tts_model, "model_size", None)
            return tts_model.is_loaded() and loaded_size == config.model_size

        if config.engine == "qwen_custom_voice":
            backend = get_tts_backend_for_engine(config.engine)
            loaded_size = getattr(backend, "_current_model_size", None) or getattr(backend, "model_size", None)
            return backend.is_loaded() and loaded_size == config.model_size

        # ⛔ voxcpm ma WLASNA galaz z dwoch powodow (zgloszone 30.08):
        # 1. backend ggml deklaruje model_size "gguf-q8", a wpis w rejestrze
        #    zostaje na domyslnym "default" — fallback nizej porownuje te dwie
        #    wartosci i zawsze wychodzilo False;
        # 2. wagi leza w OSOBNYM procesie (`llama-tts-server`), o ktory Python
        #    nigdy nie pytal — `is_loaded()` zwraca tylko flage "sprawdzilem
        #    pliki", ustawiana w `load_model()`, ktorego przy starcie nikt nie
        #    wola. Zrodlem prawdy jest to, czy serwer TTS przyjmuje polaczenie.
        # Bez serwera (tryb CLI) model faktycznie NIE lezy miedzy generacjami,
        # wiec wracamy do starej odpowiedzi.
        # JEDNA RURA: port gospodarza ggml stoi zawsze — o voxa pytamy jego.
        if config.engine == "voxcpm":
            return get_tts_backend_for_engine("voxcpm").is_loaded()

        # OmniVoice siedzi w tym samym silniku co vox — pytamy silnik o NIEGO (`/v1/models`),
        # nie o port, bo port stoi takze wtedy, gdy siedzi vox.
        if config.engine == "omnivoice":
            return get_tts_backend_for_engine("omnivoice").is_loaded()

        if config.engine in ("voxcpm", "fish_s2", "fish_s1", "voxcpm_pt", "fish_s2_pt"):
            backend = get_tts_backend_for_engine(config.engine)
            adres = getattr(backend, "_serwer", "")
            if adres:
                import socket
                from urllib.parse import urlparse

                cel = urlparse(adres)
                if cel.hostname and cel.port:
                    try:
                        with socket.create_connection((cel.hostname, cel.port), timeout=0.3):
                            return True
                    except OSError:
                        pass
            return backend.is_loaded()

        # Compare model_size like every branch above. Without it this fallback
        # answered "is this ENGINE holding anything", not "is THIS VARIANT loaded".
        # Since tada-1b and tada-3b-ml share engine="tada", loading either one
        # reported both as loaded.
        # Safe for single-variant engines: chatterbox/chatterbox_turbo/luxtts/kokoro
        # all set self.model_size = "default", which is ModelConfig's own default.
        backend = get_tts_backend_for_engine(config.engine)
        if not backend.is_loaded():
            return False
        loaded_size = getattr(backend, "_current_model_size", None) or getattr(backend, "model_size", None)
        return loaded_size is None or loaded_size == config.model_size
    except Exception:
        return False


def get_model_load_func(config: ModelConfig):
    """Return a callable that loads/downloads the model."""
    from . import get_tts_backend_for_engine
    from ..services import tts, transcribe, llm as llm_service

    if config.engine == "whisper":
        return lambda: transcribe.get_whisper_model().load_model(config.model_size)

    if config.engine == "qwen":
        return lambda: tts.get_tts_model().load_model(config.model_size)

    if config.engine == "qwen_custom_voice":
        return lambda: get_tts_backend_for_engine(config.engine).load_model(config.model_size)

    if config.engine == "qwen_llm":
        return lambda: llm_service.get_llm_model().load_model(config.model_size)

    # Pass config.model_size like every branch above. Without it this fallback
    # called load_model() bare, so the backend fell back to its own default and
    # loaded the wrong variant: requesting tada-3b-ml downloaded 3B to disk but
    # loaded tada-1B into VRAM (hume_backend.load_model defaults to "1B").
    # It also defeated that backend's own eviction — load_model() only unloads
    # when model_size changes, and with the argument dropped it never did, so
    # variants piled up in VRAM instead of replacing each other.
    # Safe for every engine reaching this line: ModelConfig.model_size defaults
    # to "default", which is exactly what chatterbox/luxtts/kokoro expect.
    return lambda: get_tts_backend_for_engine(config.engine).load_model(config.model_size)


def get_tts_backend() -> TTSBackend:
    """
    Get or create the default (Qwen) TTS backend instance based on platform.

    Returns:
        TTS backend instance (MLX or PyTorch)
    """
    return get_tts_backend_for_engine("qwen")


# ⭐ BIALA LISTA KAGANCA — tu sie dopisuje silnik, ktory ma miec prawo wstac.
# Wszystko spoza tej listy dostaje `_AtrapaTTS` i nie tknie karty. Kaganiec
# zostaje wlaczony na stale jako zabezpieczenie przed niespodziewanymi ruchami
# Voiceboxa (decyzja grruwiego 31.08) — nie zdejmujemy go, tylko dopisujemy.
# 11.10 (decyzja grruwiego): na liscie zostaja TYLKO vox (ggml), OmniVoice i vox PyTorch —
# Fish S2/S1/S2-int8 i Supertonic wyciete z rejestru i z kaganca (dostaja atrape).
_SILNIKI_PRZEPUSZCZANE = {"voxcpm", "voxcpm_pt", "omnivoice"}


class _AtrapaTTS:
    """Silnik, ktorego nie ma. Stoi w miejscu obcych backendow, gdy kaganiec
    jest zalozony (`VOICEBOX_TYLKO_VOXCPM`).

    Musi UDAWAC DZIALAJACY, a nie rzucac — bo pyta o niego offloader przed
    kazda generacja i wyjatek zablokowalby nasz wlasny silnik.
    """

    def __init__(self, engine: str):
        self.engine = engine
        self.model_size = "default"

    def is_loaded(self) -> bool:
        return False

    def unload_model(self) -> None:
        return None

    def _get_model_path(self, model_size: str = "default") -> str:
        return f"<{self.engine} zablokowany>"

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return False

    async def load_model(self, model_size: str = "default") -> None:
        raise RuntimeError(
            f"silnik '{self.engine}' jest wylaczony — Voicebox nie laduje wlasnych wag. "
            "Uzyj 'voxcpm' albo zdejmij kaganiec: VOICEBOX_TYLKO_VOXCPM=0"
        )

    async def generate(self, *a, **k):
        await self.load_model()

    async def create_voice_prompt(self, *a, **k):
        await self.load_model()

    async def combine_voice_prompts(self, audio_paths, reference_texts):
        # ⚠️ SKLEJANIE PROBEK ZOSTAJE. To numpy na CPU, zero wag — a to jest
        # jedyna rzecz, po ktora grruwi tu wchodzi (robienie profili).
        from .base import combine_voice_prompts as _sklej

        return await _sklej(audio_paths, reference_texts)


def get_tts_backend_for_engine(engine: str) -> TTSBackend:
    """
    Get or create a TTS backend for the given engine.

    Args:
        engine: Engine name (e.g. "qwen", "luxtts", "chatterbox", "chatterbox_turbo")

    Returns:
        TTS backend instance
    """
    global _tts_backends

    # ⛔ KAGANIEC (16.08, polecenie grruwiego: „wszystko wytnij, żeby nic nie
    # miało się nigdy prawa załadować, oprócz naszego backendu").
    # Voicebox wozi osiem silnikow TTS i KAZDY laduje wagi na te sama karte —
    # jedno klkniecie w GUI potrafi sciagnac i wrzucic kilkanascie GB. Tutaj
    # wszystko poza `voxcpm` (czyli poza ggml przez `voxcpm2-cli`/serwer)
    # dostaje odmowe ZANIM cokolwiek sie zaimportuje.
    # Zdjecie kaganca: `VOICEBOX_TYLKO_VOXCPM=0` w `ustawienia.env`.
    # ⛔ NIE RZUCAC WYJATKU. Offloader pyta fabryke o KAZDY silnik, zeby przed
    # generacja sprzatnac po innych — wyjatek wywala sprzatanie, a to z kolei
    # blokuje ladowanie NASZEGO silnika (`EvictionError: nie zwolniono pamieci`).
    # Wiec zamiast odmowy oddajemy atrape: melduje „nic tu nie lezy", pozwala
    # sie „odladowac", a przy probie generacji mowi wprost, co jest grane.
    if os.environ.get("VOICEBOX_TYLKO_VOXCPM", "1").strip() not in ("0", "false", "no"):
        if engine not in _SILNIKI_PRZEPUSZCZANE:
            with _tts_backends_lock:
                if engine not in _tts_backends:
                    _tts_backends[engine] = _AtrapaTTS(engine)
                return _tts_backends[engine]

    # Fast path: check without lock
    if engine in _tts_backends:
        return _tts_backends[engine]

    # Slow path: create with lock to avoid duplicate instantiation
    with _tts_backends_lock:
        # Double-check after acquiring lock
        if engine in _tts_backends:
            return _tts_backends[engine]

        if engine == "qwen":
            backend_type = get_backend_type()
            if backend_type == "mlx":
                from .mlx_backend import MLXTTSBackend

                backend = MLXTTSBackend()
            else:
                from .pytorch_backend import PyTorchTTSBackend

                backend = PyTorchTTSBackend()
        elif engine == "luxtts":
            from .luxtts_backend import LuxTTSBackend

            backend = LuxTTSBackend()
        elif engine == "chatterbox":
            from .chatterbox_backend import ChatterboxTTSBackend

            backend = ChatterboxTTSBackend()
        elif engine == "chatterbox_turbo":
            from .chatterbox_turbo_backend import ChatterboxTurboTTSBackend

            backend = ChatterboxTurboTTSBackend()
        elif engine == "tada":
            from .hume_backend import HumeTadaBackend

            backend = HumeTadaBackend()
        elif engine == "supertonic":
            from .supertonic_backend import SupertonicTTSBackend

            backend = SupertonicTTSBackend()
        elif engine == "kokoro":
            from .kokoro_backend import KokoroTTSBackend

            backend = KokoroTTSBackend()
        elif engine == "voxcpm":
            # ⚠️ `VOXCPM_GGUF=1` PODMIENIA SILNIK POD TA SAMA NAZWA. Profile,
            # historia i GUI nie widza roznicy, a wagi przestaja wchodzic do
            # tego procesu — inferencja idzie do `voxcpm2-cli` (ggml/Vulkan),
            # ktory konczy sie po kazdej wypowiedzi. Zmierzone 16.08: 7,00 s
            # -> 3,33 s i 5,6 GB rezydentnie -> zero miedzy wypowiedziami.
            if os.environ.get("VOXCPM_GGUF", "").strip() in ("1", "true", "yes"):
                from .voxcpm_gguf_backend import VoxCPMGGUFBackend

                backend = VoxCPMGGUFBackend()
            else:
                from .voxcpm_backend import VoxCPM2TTSBackend

                backend = VoxCPM2TTSBackend()
        elif engine == "omnivoice":
            # OmniVoice — ten sam silnik audio.cpp co ggml-owy voxcpm, jeden model naraz.
            from .omnivoice_backend import OmniVoiceBackend

            backend = OmniVoiceBackend()
        elif engine == "fish_s2":
            # Fish S2 Pro przez s2.cpp — ta sama zasada co ggml-owy voxcpm:
            # wagi leza w osobnym procesie, nie tutaj. ⛔ Sciezka PyTorchowa
            # odrzucona po pomiarze (31.08): 2,69 s dzwieku w 49,5 s i 16,5 GB
            # RAM, wobec 7,29 s dzwieku w 11,4 s na s2.cpp/Vulkan.
            from .fish_s2_backend import FishS2Backend

            backend = FishS2Backend()
        elif engine == "fish_s1":
            # Fish S1-mini — PyTorch/ROCm w KONTENERZE. Wagi leza w Dockerze,
            # wiec „odladowanie" to `docker stop`, nie zwolnienie obiektu.
            # Polecenie `docker run` zyje w `fish-s1-up.sh`, nie tutaj.
            from .fish_s1_backend import FishS1Backend

            backend = FishS1Backend()
        elif engine == "fish_s2_pt":
            # ⚠️ TE SAME WAGI ZRODLOWE CO `fish_s2`, inny silnik i inny format:
            # tam ggml/Vulkan, tu PyTorch na wagach int8. Kontener stoi na
            # :17499, polecenie `docker run` zyje w `fish-s2-int8-up.sh`.
            from .fish_s2_pt_backend import FishS2PyTorchBackend

            backend = FishS2PyTorchBackend()
        elif engine == "voxcpm_pt":
            # ⚠️ TE SAME WAGI CO `voxcpm`, inny silnik. ggml jest szybszy
            # (RTF 0,54-0,59 kontra 0,69-0,72), ale tylko PyTorch prowadzi do
            # LoRA. Rozdzielone, zeby dalo sie przelaczac i sluchac roznicy.
            from .voxcpm_pt_backend import VoxCPMPyTorchBackend

            backend = VoxCPMPyTorchBackend()
        elif engine == "qwen_custom_voice":
            from .qwen_custom_voice_backend import QwenCustomVoiceBackend

            backend = QwenCustomVoiceBackend()
        else:
            raise ValueError(f"Unknown TTS engine: {engine}. Supported: {list(TTS_ENGINES.keys())}")

        _tts_backends[engine] = backend
        return backend


def get_stt_backend() -> STTBackend:
    """
    Get or create STT backend instance based on platform.

    Returns:
        STT backend instance (MLX or PyTorch)
    """
    global _stt_backend

    if _stt_backend is None:
        # ⭐ `VOICEBOX_STT_ENGINE=whispercpp` — transkrypcja idzie do
        # whispera w gospodarzu ggml (jeden proces z voxem, `services/gospodarz.py`,
        # `services/whisper_serwer.py`). Bez tej zmiennej wszystko zostaje po
        # staremu (PyTorch, 1,5 GB na karte przy pierwszym klknieciu w transkrypcje).
        # Przy okazji znika blad przycinania wejscia do 30 sekund.
        if os.environ.get("VOICEBOX_STT_ENGINE", "").strip() == "whispercpp":
            from .whispercpp_backend import WhisperCppSTTBackend

            _stt_backend = WhisperCppSTTBackend()
            return _stt_backend

        backend_type = get_backend_type()

        if backend_type == "mlx":
            from .mlx_backend import MLXSTTBackend

            _stt_backend = MLXSTTBackend()
        else:
            from .pytorch_backend import PyTorchSTTBackend

            _stt_backend = PyTorchSTTBackend()

    return _stt_backend


def get_llm_backend() -> LLMBackend:
    """Domyslny backend refine. `VOICEBOX_LLM_ENGINE=lmstudio_llm` przelacza
    na LM Studio po HTTP; bez tej zmiennej nic sie nie zmienia i dalej laduje
    sie Qwen lokalnie, jak przedtem."""
    return get_llm_backend_for_engine(
        os.environ.get("VOICEBOX_LLM_ENGINE", "qwen_llm").strip() or "qwen_llm"
    )


def get_llm_backend_for_engine(engine: str) -> LLMBackend:
    """Get or create an LLM backend for the given engine."""
    global _llm_backends

    if engine in _llm_backends:
        return _llm_backends[engine]

    with _llm_backends_lock:
        if engine in _llm_backends:
            return _llm_backends[engine]

        if engine == "qwen_llm":
            backend_type = get_backend_type()
            if backend_type == "mlx":
                from .qwen_llm_backend import MLXQwenLLMBackend

                backend = MLXQwenLLMBackend()
            else:
                from .qwen_llm_backend import PyTorchQwenLLMBackend

                backend = PyTorchQwenLLMBackend()
        elif engine == "lmstudio_llm":
            from .lmstudio_llm_backend import LMStudioLLMBackend

            backend = LMStudioLLMBackend()
        else:
            raise ValueError(f"Unknown LLM engine: {engine}. Supported: {list(LLM_ENGINES.keys())}")

        _llm_backends[engine] = backend
        return backend


def reset_backends():
    """Reset backend instances (useful for testing)."""
    global _tts_backend, _tts_backends, _stt_backend, _llm_backends
    _tts_backend = None
    _tts_backends.clear()
    _stt_backend = None
    _llm_backends.clear()
