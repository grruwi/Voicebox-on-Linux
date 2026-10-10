"""Monkey-patch huggingface_hub to force offline mode with cached models.

Prevents mlx_audio / transformers from making network requests when models
are already downloaded. Must be imported BEFORE mlx_audio.
"""

import logging
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)


# huggingface_hub reads ``HF_HUB_OFFLINE`` once at import time into
# ``huggingface_hub.constants.HF_HUB_OFFLINE``; transformers mirrors that into
# ``transformers.utils.hub._is_offline_mode`` at *its* import time. Toggling
# ``os.environ`` after either module is imported does not flip those cached
# bools, and the hot paths (``_http._default_backend_factory``,
# ``transformers.utils.hub.is_offline_mode``) read the bools — not the env.
# We mutate the cached constants directly, guarded by a refcount so
# concurrent inference threads share a single offline window safely.

_offline_lock = threading.RLock()
_offline_refcount = 0


def _tf_hub():
    """``transformers.utils.hub``, ale TYLKO jesli juz siedzi w procesie.

    ⭐ Import wciagnalby tu 900 MB torcha przy KAZDYM ladowaniu modelu — takze
    przy silnikach ggml/ONNX, ktore transformers w ogole nie uzywaja. Gdy modul
    nie jest zaladowany, nie ma czego przestawiac: transformers przy swoim
    imporcie odczyta `HF_HUB_OFFLINE` ze srodowiska, ktore ustawiamy obok.
    """
    return sys.modules.get("transformers.utils.hub")
_saved_env: Optional[str] = None
_saved_hf_const: Optional[bool] = None
_saved_transformers_const: Optional[bool] = None


@contextmanager
def force_offline_if_cached(is_cached: bool, model_label: str = ""):
    """Force offline mode for the duration of a cached-model operation.

    Flips ``HF_HUB_OFFLINE`` in the process env **and** in the cached bools
    inside ``huggingface_hub.constants`` / ``transformers.utils.hub`` so HTTP
    adapters and offline-mode checks actually see the change. Uses a refcount
    so multiple concurrent inference threads share a single offline window
    and the last one to exit restores state.

    If *is_cached* is ``False`` the block runs normally (network allowed).

    Args:
        is_cached: Whether the model weights are already on disk.
        model_label: Human-readable name used in log messages.
    """
    if not is_cached:
        yield
        return

    global _offline_refcount, _saved_env, _saved_hf_const, _saved_transformers_const

    with _offline_lock:
        if _offline_refcount == 0:
            # Snapshot prior state, apply new state, roll back on *any*
            # failure. Catching only ImportError here would let a partially
            # broken install (RuntimeError, AttributeError from a half-init
            # module, etc.) leave the cached HF constants mutated without
            # bumping the refcount — a persistent offline leak that outlives
            # the process and is miserable to debug.
            prev_env = os.environ.get("HF_HUB_OFFLINE")
            prev_hf: Optional[bool] = None
            prev_tf: Optional[bool] = None
            try:
                try:
                    import huggingface_hub.constants as hf_const

                    prev_hf = hf_const.HF_HUB_OFFLINE
                    hf_const.HF_HUB_OFFLINE = True
                except ImportError:
                    prev_hf = None

                tf_hub = _tf_hub()
                if tf_hub is not None:
                    prev_tf = tf_hub._is_offline_mode
                    tf_hub._is_offline_mode = True
                else:
                    prev_tf = None

                os.environ["HF_HUB_OFFLINE"] = "1"
            except BaseException:
                # Roll back whatever we already changed, then re-raise so
                # the caller sees the real failure.
                if prev_hf is not None:
                    try:
                        import huggingface_hub.constants as hf_const

                        hf_const.HF_HUB_OFFLINE = prev_hf
                    except ImportError:
                        pass
                if prev_tf is not None:
                    tf_hub = _tf_hub()
                    if tf_hub is not None:
                        tf_hub._is_offline_mode = prev_tf
                if prev_env is not None:
                    os.environ["HF_HUB_OFFLINE"] = prev_env
                else:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                raise

            _saved_env = prev_env
            _saved_hf_const = prev_hf
            _saved_transformers_const = prev_tf
            logger.info(
                "[offline-guard] %s is cached — forcing offline mode",
                model_label or "model",
            )
        _offline_refcount += 1

    try:
        yield
    finally:
        with _offline_lock:
            _offline_refcount -= 1
            if _offline_refcount == 0:
                if _saved_env is not None:
                    os.environ["HF_HUB_OFFLINE"] = _saved_env
                else:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                if _saved_hf_const is not None:
                    try:
                        import huggingface_hub.constants as hf_const

                        hf_const.HF_HUB_OFFLINE = _saved_hf_const
                    except ImportError:
                        pass
                if _saved_transformers_const is not None:
                    tf_hub = _tf_hub()
                    if tf_hub is not None:
                        tf_hub._is_offline_mode = _saved_transformers_const
                _saved_env = None
                _saved_hf_const = None
                _saved_transformers_const = None


_mistral_regex_patched = False


def patch_transformers_mistral_regex():
    """Make transformers' tokenizer load robust to HuggingFace metadata failures.

    transformers 4.57.x added ``PreTrainedTokenizerBase._patch_mistral_regex``
    which unconditionally calls ``huggingface_hub.model_info(repo_id)`` during
    every non-local tokenizer load to check whether the model is a Mistral
    variant. That call raises on ``HF_HUB_OFFLINE=1`` and on plain network
    failures, killing unrelated loads (Qwen TTS, TADA, etc.).

    Voicebox never loads Mistral models, so the rewrite the function would
    apply is a no-op for us anyway. Wrap the method so any exception from the
    metadata lookup returns the tokenizer unchanged — matching the success-path
    behavior for non-Mistral repos (transformers 4.57.3,
    ``tokenization_utils_base.py:2503``).
    """
    global _mistral_regex_patched
    if _mistral_regex_patched:
        return

    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except ImportError:
        logger.debug("transformers not available, skipping mistral-regex patch")
        return

    original = getattr(PreTrainedTokenizerBase, "_patch_mistral_regex", None)
    if original is None:
        logger.debug(
            "transformers has no _patch_mistral_regex attribute, skipping patch",
        )
        return

    def safe_patch_mistral_regex(cls, tokenizer, pretrained_model_name_or_path, *args, **kwargs):
        try:
            return original(tokenizer, pretrained_model_name_or_path, *args, **kwargs)
        except Exception as exc:
            logger.debug(
                "[mistral-regex-patch] suppressed %s for %r, returning tokenizer as-is",
                type(exc).__name__,
                pretrained_model_name_or_path,
            )
            return tokenizer

    PreTrainedTokenizerBase._patch_mistral_regex = classmethod(safe_patch_mistral_regex)
    _mistral_regex_patched = True
    logger.debug("installed _patch_mistral_regex wrapper")


def patch_huggingface_hub_offline():
    """Monkey-patch huggingface_hub to force offline mode."""
    try:
        import huggingface_hub  # noqa: F401 -- need the package loaded
        from huggingface_hub import constants as hf_constants
        from huggingface_hub.file_download import _try_to_load_from_cache

        original_try_load = _try_to_load_from_cache

        def _patched_try_to_load_from_cache(
            repo_id: str,
            filename: str,
            cache_dir: Union[str, Path, None] = None,
            revision: Optional[str] = None,
            repo_type: Optional[str] = None,
        ):
            result = original_try_load(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                revision=revision,
                repo_type=repo_type,
            )

            if result is None:
                cache_path = Path(hf_constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
                logger.debug("file not cached: %s/%s (expected at %s)", repo_id, filename, cache_path)
            else:
                logger.debug("cache hit: %s/%s", repo_id, filename)

            return result

        import huggingface_hub.file_download as fd

        fd._try_to_load_from_cache = _patched_try_to_load_from_cache
        logger.debug("huggingface_hub patched for offline mode")

    except ImportError:
        logger.debug("huggingface_hub not available, skipping offline patch")
    except Exception:
        logger.exception("failed to patch huggingface_hub for offline mode")


def ensure_original_qwen_config_cached():
    """Symlink the original Qwen repo cache to the MLX community version.

    mlx_audio may try to fetch config from the original Qwen repo. If only
    the MLX community variant is cached, create a symlink so the cache lookup
    succeeds without a network request.
    """
    try:
        from huggingface_hub import constants as hf_constants
    except ImportError:
        return

    original_repo = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    mlx_repo = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

    cache_dir = Path(hf_constants.HF_HUB_CACHE)
    original_path = cache_dir / f"models--{original_repo.replace('/', '--')}"
    mlx_path = cache_dir / f"models--{mlx_repo.replace('/', '--')}"

    if not original_path.exists() and mlx_path.exists():
        try:
            original_path.parent.mkdir(parents=True, exist_ok=True)
            original_path.symlink_to(mlx_path, target_is_directory=True)
            logger.info("created cache symlink: %s -> %s", original_repo, mlx_repo)
        except Exception:
            logger.warning("could not create cache symlink for %s", original_repo, exc_info=True)


# --------------------------------------------------------------------------
# Latka na transformers — ZAWIESZONA, nie odpalana od razu
# --------------------------------------------------------------------------
#
# ⭐ `patch_transformers_mistral_regex()` musiala kiedys importowac
# transformers, zeby cokolwiek zalatac — a transformers ciagnie torcha, czyli
# ~900 MB RAM przy samym starcie Voiceboxa. Latka jest potrzebna PRZED
# pierwszym tokenizerem, nie przed pierwszym requestem, wiec wpinamy sie w
# maszynerie importu: dopiero gdy ktos naprawde zaimportuje
# `transformers.tokenization_utils_base`, haczyk odpala latke i sam sie
# wypina. Silniki ggml/ONNX nie dotykaja transformers i nigdy jej nie obudza.


class _LoaderProxy:
    """Prawdziwy loader + jedno wywolanie zwrotne po wykonaniu modulu."""

    def __init__(self, loader, callback: Callable[[], None]):
        self._loader = loader
        self._callback = callback

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        self._callback()

    def __getattr__(self, nazwa):
        return getattr(self._loader, nazwa)


class _LatkaPoImporcie:
    """Finder, ktory nie laduje niczego sam — tylko doszywa callback do modulu."""

    def __init__(self, cel: str, callback: Callable[[], None]):
        self._cel = cel
        self._callback = callback

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._cel:
            return None
        # Pytamy reszte meta_path o prawdziwy spec — bez siebie, zeby nie
        # wpasc w nieskonczona petle.
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            znajdz = getattr(finder, "find_spec", None)
            if znajdz is None:
                continue
            spec = znajdz(fullname, path, target)
            if spec is not None:
                break
        else:
            return None

        if spec.loader is None:
            return spec
        spec.loader = _LoaderProxy(spec.loader, self._odpal)
        return spec

    def _odpal(self):
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        self._callback()


def schedule_transformers_mistral_regex_patch() -> None:
    """Zawiesza latke na transformers zamiast importowac je na starcie."""
    cel = "transformers.tokenization_utils_base"
    if cel in sys.modules:
        patch_transformers_mistral_regex()
        return
    if any(isinstance(f, _LatkaPoImporcie) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _LatkaPoImporcie(cel, patch_transformers_mistral_regex))
    logger.debug("latka mistral-regex zawieszona do importu %s", cel)


if os.environ.get("VOICEBOX_OFFLINE_PATCH", "1") != "0":
    patch_huggingface_hub_offline()
    schedule_transformers_mistral_regex_patch()
    ensure_original_qwen_config_cached()
