"""memory.py — give memory back AFTER EVERY generation, without unloading a model.

Upstream calls `torch.cuda.empty_cache()` in ONE place: when a model is unloaded. After
an utterance it gives nothing back, so the allocator cache grows with every generation.

Why this is a critical path, not cosmetics: this machine is used REMOTELY. A full VRAM
on a 16 GiB card is not broken sound, it is a GPU reset and a lost graphical session
until someone is physically back at the computer. Measured 2026-08-03: past 15 GiB, the
generation cut off halfway, the voice "farts" and only says the ending.

Voicebox runs no model of its own now (the engines live in containers), so this mostly
returns the heap: torch is only touched when something already imported it.

The order matters:
  1. gc.collect()    — frees tensors with no live references (while someone holds
                       them, the allocator does not consider the blocks free),
  2. empty_cache()   — gives the driver the blocks torch kept in reserve,
  3. malloc_trim(0)  — gives the kernel free heap pages; this CANNOT be called from
                       outside the process, hence the code.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import logging
import os
import sys

logger = logging.getLogger(__name__)

_libc = None
_libc_checked = False


def _malloc_trim() -> bool:
    """Gives free heap pages back to the kernel. True when it worked."""
    global _libc, _libc_checked
    if not _libc_checked:
        _libc_checked = True
        try:
            name = ctypes.util.find_library("c") or "libc.so.6"
            candidate = ctypes.CDLL(name)
            # malloc_trim is a glibc extension — musl does not have it.
            candidate.malloc_trim.argtypes = [ctypes.c_size_t]
            candidate.malloc_trim.restype = ctypes.c_int
            _libc = candidate
        except (OSError, AttributeError):
            logger.debug("malloc_trim unavailable (not glibc?) — skipping")
            _libc = None
    if _libc is None:
        return False
    try:
        _libc.malloc_trim(0)
        return True
    except Exception:
        return False


def _rss_mib() -> float:
    """The process's own RSS. The one measurement that works without torch."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * 4096 / 1048576
    except (OSError, ValueError, IndexError):
        return 0.0


def release_memory(label: str = "") -> None:
    """Give back the GPU cache and the free heap. Never raises — a helper path.

    Called in the `finally` of both generation routes AND after unloading a model, so it
    has to survive an interrupted and a failed utterance too. An exception here would
    blow up a generation that already succeeded.
    """
    if os.environ.get("VOICEBOX_MEMORY_RELEASE_OFF"):
        return
    try:
        gc.collect()

        before = after = None
        # ⭐ Only when torch ALREADY sits in the process. This runs in the `finally` of both
        # generation routes, so `import torch` would pull in all of PyTorch after every
        # utterance. Without torch there is no torch cache to give back either.
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    before = torch.cuda.memory_reserved()
                    torch.cuda.empty_cache()
                    after = torch.cuda.memory_reserved()
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.empty_cache()
            except Exception:
                logger.debug("empty_cache skipped", exc_info=True)

        rss_before = _rss_mib()
        trimmed = _malloc_trim()
        rss_after = _rss_mib()

        # ⛔ The log MUST NOT hang on the torch branch — without torch `before`/`after` are
        # empty and the function would work in silence, with no way to see it ran at all.
        parts = [f"RSS {rss_before:.0f} -> {rss_after:.0f} MiB"]
        if before is not None and after is not None:
            parts.insert(0, f"GPU {before / 1048576:.0f} -> {after / 1048576:.0f} MiB")
        logger.info(
            "memory[%s]: %s, malloc_trim=%s",
            label or "?",
            ", ".join(parts),
            "yes" if trimmed else "no",
        )
    except Exception:
        logger.exception("release_memory[%s] failed", label or "?")
