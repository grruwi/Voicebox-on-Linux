"""Precyzja wag voxa w gospodarzu audio.cpp — jedyne miejsce, ktore o niej decyduje.

Od 10.10 silnik to `audiocpp_server` (`services/gospodarz.py`). Wagi leza w JEDNYM
GGUF-ie (BF16 + AudioVAE F32), a precyzje wybiera sie przy ladowaniu modelu
(`session_options`). Gdy dla wybranej precyzji generatora lezy GOTOWY plik
(`wagi-gotowe/<typ>/`, `skrypty/zrob-wagi-audiocpp.sh`), silnik nic nie przepisuje
(~2 s, z zimnego dysku ~4 s); bez niego przepisuje oryginal (~7 s / ~23 s).
Zmiana w panelu = restart gospodarza z nowymi `session_options`.
Do 10.10: osobne pliki GGUF na kazdy wariant + `llama-tts-server`.
"""

import logging
import os
import shlex
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ⭐ OD 10.10 (decyzja grruwiego, punkt 4 listy): dwa pola panelu, trzy czesci modelu.
#   BaseLM   (`vox_base_lm`)  → `voxcpm2.base_lm_weight_type` — SAM BaseLM (nasza latka 11):
#            tekst → plan wypowiedzi (slowa, akcent, intonacja); q8 = -1,3 GiB VRAM.
#   akustyka (`vox_akustyka`) → `voxcpm2.weight_type` — reszta generatora: ResidualLM,
#            LocEnc, DiT, projekcje (jak „Acoustic" w starym voxie).
#   glowica  = AudioVAE → ZAWSZE f32 (decyzja grruwiego „F32 do konca"), bez pola.
# Gotowe pliki na kazde polaczenie: `wagi-gotowe/<base_lm>-<akustyka>/`
# (`skrypty/zrob-wagi-audiocpp.sh`); bez pliku silnik przepisuje oryginal przy ladowaniu.
WARIANTY = {
    "q8": "q8_0",
    "f16": "f16",
    "f32": "f32",
}

# ⛔ Wariantu q8 dla akustyki NIE MA (dyfuzja DiT).
AKUSTYKA_WARIANTY = {
    "f16": "f16",
    "f32": "f32",
}

GLOWICA = "f32"

# Backend silnika (`vox_backend`, 10.10): osobne binarki, zero narzutu na nieuzywany
# (jedna binarka z oboma rejestrowalaby oba i budzila runtime ROCm takze pod Vulkanem).
# Gniazda robi `skrypty/buduj-audiocpp.sh [hip|vulkan]`. Zmiana = restart gospodarza.
BACKENDY = {
    "hip": "bin-docker",
    "vulkan": "bin-docker-vulkan",
}


def opcje_wag(base_lm: str, akustyka: str) -> dict:
    """Klucze panelu → `session_options` silnika. Jedyne miejsce tego mapowania."""
    return {
        "voxcpm2.base_lm_weight_type": WARIANTY.get(base_lm, WARIANTY["q8"]),
        "voxcpm2.weight_type": AKUSTYKA_WARIANTY.get(akustyka, AKUSTYKA_WARIANTY["f16"]),
        "voxcpm2.audiovae_weight_type": GLOWICA,
    }


def _env(nazwa: str, domyslnie: str = "") -> str:
    return os.environ.get(nazwa, domyslnie).strip()


def wybrane() -> tuple:
    """(base_lm, akustyka) z BAZY — panel wygrywa. Zapas: env, potem domyslne.

    Jedno zrodlo prawdy dla obu drog do serwera (backend i `przeladuj`).
    """
    base_lm, akustyka = None, None
    try:
        from ..database import session as db_session
        from ..database import GenerationSettings

        s = db_session.SessionLocal()
        try:
            w = s.query(GenerationSettings).first()
            if w is not None:
                base_lm = (getattr(w, "vox_base_lm", None) or "").lower() or None
                akustyka = (getattr(w, "vox_akustyka", None) or "").lower() or None
        finally:
            s.close()
    except Exception as e:
        logger.debug("wagi nie z bazy (%s) — biore ze srodowiska", e)
    return (
        base_lm or _env("VOXCPM_GGUF_WARIANT", "q8").lower(),
        akustyka or "f16",
    )


def wybrany_backend() -> str:
    """hip | vulkan z BAZY — panel wygrywa; domyslnie vulkan (jedyny w publicznym obrazie)."""
    backend = None
    try:
        from ..database import session as db_session
        from ..database import GenerationSettings

        s = db_session.SessionLocal()
        try:
            w = s.query(GenerationSettings).first()
            if w is not None:
                backend = (getattr(w, "vox_backend", None) or "").lower() or None
        finally:
            s.close()
    except Exception as e:
        logger.debug("backend nie z bazy (%s) — biore vulkan", e)
    return backend if backend in BACKENDY else "vulkan"


# OmniVoice (`omni_wagi`, 10.10): gotowe GGUF-y z upstreamu (HF `audio-cpp/audio.cpp-gguf`,
# `OmniVoice-GGUF/`) w `wagi/omnivoice/`. q8 = domyslny upstreamu (`recommended_package`).
OMNI_WARIANTY = {
    "q8": "q8_0",
    "f16": "f16",
}


def wybrany_omni() -> str:
    """q8 | f16 z BAZY — panel wygrywa; domyslnie q8 (jak upstream)."""
    wariant = None
    try:
        from ..database import session as db_session
        from ..database import GenerationSettings

        s = db_session.SessionLocal()
        try:
            w = s.query(GenerationSettings).first()
            if w is not None:
                wariant = (getattr(w, "omni_wagi", None) or "").lower() or None
        finally:
            s.close()
    except Exception as e:
        logger.debug("wagi OmniVoice nie z bazy (%s) — biore q8", e)
    return wariant if wariant in OMNI_WARIANTY else "q8"


def wybrany_omni_flash() -> bool:
    """Flash attention OmniVoice z BAZY — domyslnie TAK (decyzja grruwiego „wszystko FA")."""
    try:
        from ..database import session as db_session
        from ..database import GenerationSettings

        s = db_session.SessionLocal()
        try:
            w = s.query(GenerationSettings).first()
            if w is not None and getattr(w, "omni_flash", None) is not None:
                return bool(w.omni_flash)
        finally:
            s.close()
    except Exception as e:
        logger.debug("flash OmniVoice nie z bazy (%s) — biore tak", e)
    return True


# Ziarnistosc strumienia (`vox_stream_min` / `vox_stream_co`) — pola w panelu
# zostaja dla zgodnosci, silnik ich nie zna (audio.cpp sypie kawalek co patch).
_STRUMIEN_MIN = 12
_STRUMIEN_CO = 16


def strumien_wybrany() -> tuple:
    """(min, co) z BAZY — tylko do pokazania w panelu."""
    minimum, co = None, None
    try:
        from ..database import session as db_session
        from ..database import GenerationSettings

        s = db_session.SessionLocal()
        try:
            w = s.query(GenerationSettings).first()
            if w is not None:
                minimum = getattr(w, "vox_stream_min", None) or None
                co = getattr(w, "vox_stream_co", None) or None
        finally:
            s.close()
    except Exception as e:
        logger.debug("ziarnistosc nie z bazy (%s) — biore ze srodowiska", e)
    def _liczba(wartosc, zmienna: str, domyslnie: int) -> int:
        if wartosc:
            return int(wartosc)
        z_env = _env(zmienna)
        return int(z_env) if z_env.isdigit() and int(z_env) > 0 else domyslnie
    return (
        _liczba(minimum, "VOXCPM_STREAM_MIN", _STRUMIEN_MIN),
        _liczba(co, "VOXCPM_STREAM_CO", _STRUMIEN_CO),
    )


def flagi_dodatkowe() -> list:
    """Dowolne flagi dopisywane do `audiocpp_server` z `VOXCPM_SERWER_FLAGI`.

    Wolna dziurka: cokolwiek rozumie `audiocpp_server --help` idzie tu jednym
    napisem. `shlex` zna cudzyslowy. Puste = nic nie dopisujemy.
    """
    surowe = _env("VOXCPM_SERWER_FLAGI")
    if not surowe:
        return []
    try:
        rozbite = shlex.split(surowe)
    except ValueError as e:
        logger.warning("VOXCPM_SERWER_FLAGI nie da sie rozebrac (%s) — pomijam", e)
        return []
    logger.info("serwer TTS dostaje flagi z ustawien: %s", " ".join(rozbite))
    return rozbite


def _host_port() -> Optional[tuple]:
    """Adres z `VOXCPM_SERWER`. None = serwer nie jest uzywany."""
    adres = _env("VOXCPM_SERWER")
    if not adres:
        return None
    u = urlparse(adres)
    return (u.hostname or "127.0.0.1", u.port or 17496)


def zaladowane() -> tuple:
    """(base_lm, akustyka), z ktorymi vox FAKTYCZNIE siedzi w gospodarzu — z jego
    `/v1/models`, nie z panelu. (None, None) = nie siedzi."""
    from . import gospodarz

    v = gospodarz.stan().get("vox")
    if v is None:
        return (None, None)
    opcje = v.get("session_options", {})
    typ_base = opcje.get("voxcpm2.base_lm_weight_type", opcje.get("voxcpm2.weight_type"))
    base = next((k for k, t in WARIANTY.items() if typ_base == t), None)
    ak = next((k for k, t in AKUSTYKA_WARIANTY.items() if opcje.get("voxcpm2.weight_type") == t), None)
    return (base, ak)


def stan() -> dict:
    from . import gospodarz

    adres = _host_port()
    chodzi = gospodarz.stan().get("vox") is not None
    zal_base, zal_ak = zaladowane() if chodzi else (None, None)
    wyb_base, wyb_ak = wybrane()
    wyb_min, wyb_co = strumien_wybrany()
    omni = gospodarz.stan().get("omni")
    zal_backend = gospodarz.backend_procesu() if (chodzi or omni) else None
    return {
        "dostepny": adres is not None,
        "chodzi": chodzi,
        # `wariant` zostaje dla zgodnosci ze starym panelem
        "wariant": zal_base,
        "zaladowany_base_lm": zal_base,
        "zaladowany_akustyka": zal_ak,
        "wybrany_base_lm": wyb_base,
        "wybrany_akustyka": wyb_ak,
        "zaladowany_backend": zal_backend,
        "wybrany_backend": wybrany_backend(),
        # OmniVoice siedzi w tym samym silniku — jeden model naraz, wiec albo on, albo vox
        "omni_chodzi": omni is not None,
        "zaladowany_omni": (omni or {}).get("wariant"),
        "zaladowany_omni_flash": (omni or {}).get("flash"),
        "wybrany_omni": wybrany_omni(),
        "omni": [{
            "klucz": k,
            "plik": gospodarz.wagi_omni(t).name,
            "jest_na_dysku": gospodarz.wagi_omni(t).is_file(),
            "rozmiar_mib": int(gospodarz.wagi_omni(t).stat().st_size / 1048576)
            if gospodarz.wagi_omni(t).is_file() else 0,
        } for k, t in OMNI_WARIANTY.items()],
        "base_lm": _opisz(WARIANTY),
        "akustyka": _opisz(AKUSTYKA_WARIANTY),
        "zaladowany_stream_min": None,
        "zaladowany_stream_co": None,
        "wybrany_stream_min": wyb_min,
        "wybrany_stream_co": wyb_co,
    }


def _opisz(mapa: dict) -> list:
    """Kazdy wariant da sie zaladowac z oryginalu (silnik przepisze); gotowe pliki
    sa per POLACZENIE pol, wiec tu tylko typ i oryginal."""
    from . import gospodarz

    pliki = sorted(gospodarz.wagi().glob("*.gguf"))
    plik = pliki[0] if pliki else None
    return [{
        "klucz": klucz,
        "plik": f"{plik.name} → {typ}" if plik else f"(brak GGUF) → {typ}",
        "jest_na_dysku": plik is not None,
        "rozmiar_mib": int(plik.stat().st_size / 1048576) if plik else 0,
    } for klucz, typ in mapa.items()]


def przeladuj(wariant: Optional[str] = None, akustyka_wariant: Optional[str] = None,
              backend: Optional[str] = None) -> dict:
    """Laduje voxa w gospodarzu z wybrana precyzja. Puste = to, co stoi w panelu."""
    import time

    dom_base, dom_ak = wybrane()
    wariant = (wariant or dom_base).lower()
    akustyka_wariant = (akustyka_wariant or dom_ak).lower()
    if wariant not in WARIANTY:
        raise ValueError(f"nieznany wariant base_lm: {wariant}")
    if akustyka_wariant not in AKUSTYKA_WARIANTY:
        raise ValueError(f"nieznany wariant akustyki: {akustyka_wariant}")
    backend = (backend or wybrany_backend()).lower()
    if backend not in BACKENDY:
        raise ValueError(f"nieznany backend: {backend}")
    if _host_port() is None:
        raise RuntimeError(
            "serwer TTS nie jest uzywany (puste VOXCPM_SERWER) — nie ma czego przeladowac"
        )

    from . import gospodarz

    start = time.monotonic()
    gospodarz.zaladuj_voxa(opcje_wag(wariant, akustyka_wariant), backend)
    ms = int((time.monotonic() - start) * 1000)
    logger.info("vox przeladowany w gospodarzu: base_lm=%s akustyka=%s backend=%s w %d ms",
                wariant, akustyka_wariant, backend, ms)
    return {**stan(), "ms": ms}


def przeladuj_omni() -> dict:
    """Guzik „Przeladuj" w zakladce OmniVoice: restart silnika z OmniVoice z panelu
    (wagi, flash attention, backend) — takze przy tym samym zestawie. Vox schodzi."""
    import time

    if _host_port() is None:
        raise RuntimeError("silnik audio.cpp nie jest uzywany (puste VOXCPM_SERWER)")
    from . import gospodarz

    start = time.monotonic()
    gospodarz.zaladuj_omni(wybrany_omni(), wybrany_omni_flash(), wybrany_backend(), wymus=True)
    ms = int((time.monotonic() - start) * 1000)
    logger.info("OmniVoice przeladowany: wagi=%s flash=%s backend=%s w %d ms",
                wybrany_omni(), wybrany_omni_flash(), wybrany_backend(), ms)
    return {**stan(), "ms": ms}


def zdejmij_omni() -> dict:
    """Guzik „Rozladuj" w zakladce OmniVoice — VRAM wraca, proces silnika zostaje."""
    from . import gospodarz

    if gospodarz.stoi():
        gospodarz.zdejmij_omni()
    return stan()
