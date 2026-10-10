"""pamiec.py — oddanie pamieci PO KAZDEJ generacji, bez zdejmowania modelu.

Upstream wola `torch.cuda.empty_cache()` w JEDNYM miejscu: przy zdejmowaniu
modelu (`backends/base.py`, `qwen_custom_voice_backend.py`, `kokoro_backend.py`).
Po samej wypowiedzi nie oddaje nic. Cache alokatora rosnie wiec z kazda
generacja, a jedyna droga do jego odzyskania jest wyrzucenie wag — czyli
przeladowanie modelu przy nastepnym zdaniu (~7 s ciszy).

Dlaczego to sciezka krytyczna, a nie kosmetyka: ta maszyna pracuje ZDALNIE.
Pelny VRAM na karcie 16 GiB to nie zepsuty dzwiek, tylko reset GPU i utrata
sesji graficznej az do fizycznego powrotu do komputera. Zmierzone 2026-08-03:
przekroczenie 15 GiB, generacja urwana w polowie, glos "pierdzi" i dopowiada
sama koncowke.

Model ZOSTAJE zaladowany — oddajemy cache alokatora, nie wagi.

Kolejnosc ma znaczenie:
  1. gc.collect()    — zwalnia tensory bez zywych referencji (dopoki ktos je
                       trzyma, alokator nie uzna blokow za wolne),
  2. empty_cache()   — oddaje sterownikowi bloki, ktore torch trzymal na zapas,
  3. malloc_trim(0)  — oddaje jadru wolne strony sterty; tego NIE da sie
                       wywolac z zewnatrz procesu, stad latka w kodzie.
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
_libc_sprawdzone = False


def _malloc_trim() -> bool:
    """Oddaje jadru wolne strony sterty. True gdy sie udalo."""
    global _libc, _libc_sprawdzone
    if not _libc_sprawdzone:
        _libc_sprawdzone = True
        try:
            nazwa = ctypes.util.find_library("c") or "libc.so.6"
            kandydat = ctypes.CDLL(nazwa)
            # malloc_trim to rozszerzenie glibc — na musl go nie ma.
            kandydat.malloc_trim.argtypes = [ctypes.c_size_t]
            kandydat.malloc_trim.restype = ctypes.c_int
            _libc = kandydat
        except (OSError, AttributeError):
            logger.debug("malloc_trim niedostepny (nie-glibc?) — pomijam")
            _libc = None
    if _libc is None:
        return False
    try:
        _libc.malloc_trim(0)
        return True
    except Exception:
        return False


def _rss_mib() -> float:
    """RSS wlasnego procesu. Jedyny pomiar, ktory dziala bez torcha."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * 4096 / 1048576
    except (OSError, ValueError, IndexError):
        return 0.0


def zwolnij_pamiec(etykieta: str = "") -> None:
    """Oddaj cache GPU i wolna sterte. Nigdy nie rzuca — to sciezka pomocnicza.

    Wywolywane w `finally` obu tras generacji ORAZ po odladowaniu modelu,
    wiec musi przezyc rowniez przerwana i nieudana wypowiedz. Wyjatek stad
    wysadzilby generacje, ktora juz sie powiodla.
    """
    if os.environ.get("PAMIEC_OFF"):
        return
    try:
        gc.collect()

        przed = po = None
        # ⭐ Tylko gdy torch JUZ siedzi w procesie. Ta funkcja leci w `finally`
        # obu tras generacji, wiec `import torch` sciagalby caly PyTorch po
        # kazdej wypowiedzi — takze tej z ggml czy ONNX-a, ktore o torchu nic
        # nie wiedza. Bez torcha nie ma tez jego cache do oddania.
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    przed = torch.cuda.memory_reserved()
                    torch.cuda.empty_cache()
                    po = torch.cuda.memory_reserved()
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.empty_cache()
            except Exception:
                logger.debug("empty_cache pominiete", exc_info=True)

        rss_przed = _rss_mib()
        przyciete = _malloc_trim()
        rss_po = _rss_mib()

        # ⛔ Log NIE MOZE wisiec na galezi torcha. Vox i Supertonic chodza bez
        # PyTorcha, wiec `przed`/`po` sa puste i funkcja pracowala w ciszy —
        # nie bylo jak zobaczyc, czy w ogole sie odpalila.
        czesci = [f"RSS {rss_przed:.0f} -> {rss_po:.0f} MiB"]
        if przed is not None and po is not None:
            czesci.insert(0, f"GPU {przed / 1048576:.0f} -> {po / 1048576:.0f} MiB")
        logger.info(
            "pamiec[%s]: %s, malloc_trim=%s",
            etykieta or "?",
            ", ".join(czesci),
            "tak" if przyciete else "nie",
        )
    except Exception:
        logger.exception("zwolnij_pamiec[%s] wywalilo sie", etykieta or "?")
