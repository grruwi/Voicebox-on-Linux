"""Gospodarz audio.cpp — klient kontenera `audiocpp` (silnik glosu poza Voiceboxem).

⭐ OD 10.10 (decyzja grruwiego): `audiocpp_server` chodzi w OSOBNYM KONTENERZE
`audiocpp` (`/klocki/apczan-kutasu/voicebox/audiocpp/`), zeby Voicebox i modele
przenosily sie osobno. Tu zostaje MOZG: wpis modelu skladany z panelu. Proces
podnosi i zdejmuje zarzadca kontenera (`AUDIOCPP_ZARZADCA`, `/uruchom`, `/zatrzymaj`),
a z samym silnikiem (`VOXCPM_SERWER`) gadamy wprost: `/v1/audio/speech`,
`/v1/models`, `/v1/tasks/unload_models`.

⭐ JEDEN MODEL NARAZ: proces ma zawsze dokladnie jeden wpis w `models` (pilnuje
zarzadca), wiec zaladowanie innego = restart procesu i VRAM na pewno wraca.

Silnik wstaje z Voiceboxem z voxem LENIWYM (`lazy`, VRAM pusty). Precyzja wag
z panelu siedzi w `session_options`, a te sa w configu procesu — wiec „zaladuj"
albo zmiana precyzji = restart procesu z `lazy: false` (model laduje sie PRZED
otwarciem portu). Zdjecie = `/v1/tasks/unload_models`, proces zostaje.
⛔ `ui_management` (`/v1/models/load` w locie) wymaga builda z
`AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON` (BoringSSL + instalator z sieci) — nie mamy.

Do 10.10: proces w kontenerze Voiceboxa (kopia: gałąź `audiocpp`, commit cb64f1f),
wczesniej `llama-tts-server` z llama.cpp-omni (tag `przed-audiocpp` w `native/`).
Whisper NIE tutaj — osobny kontener `whisper` (`services/whisper_serwer.py`).

Binarki: `skrypty/buduj-audiocpp.sh [hip|vulkan]` → `voice-models/audiocpp/bin-docker*/`
(montowane w kontener silnika). Ich obecnosc sprawdza ZARZADCA, nie Voicebox — kontener
Voiceboxa nie musi ich widziec (publiczne repo: binarka siedzi tylko w obrazie silnika).
"""

import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_blokada = threading.Lock()

VOX_ID = "voxcpm2"
OMNI_ID = "omnivoice"

# ⭐ Probki profili maja do ~30 s (10.10: najdluzsza 29,4 s). Domyslna pojemnosc
# enkodera AudioVAE konczy sie na ~15 s i request pada z
# „AudioVAE encoder sample capacity exceeded". 512000 probek 16 kHz = 32 s,
# graf enkodera dostaje pod to 2048 MB. Zmierzone na Kleksie (29 s).
_POJEMNOSC_ENKODERA = {
    "voxcpm2.audiovae_encoder_sample_capacity": 512000,
    "voxcpm2.audiovae_encoder_graph_context_mb": 2048,
}


# ⭐ Cache probki na DYSKU (nasza latka w audio.cpp, `prompt_disk_cache.h`):
# cechy AudioVAE + embeddingi LocEnc, ~2,4 s liczenia przy kazdej zmianie
# profilu (RAM trzyma 1 profil i znika z procesem). Klucz to tresc probki,
# transkrypt i typ wag; katalog rozdziela pliki wag (nazwa, rozmiar, mtime).
_CACHE_PROBEK = Path("/app/data/cache/audiocpp-probki")


def _katalog_cache() -> str:
    znacznik = "-".join(
        f"{p.stem}-{p.stat().st_size}-{int(p.stat().st_mtime)}"
        for p in sorted(wagi().glob("*.gguf"))
    ) or "brak-wag"
    return str(_CACHE_PROBEK / znacznik)


def _env(nazwa: str, domyslnie: str = "") -> str:
    return os.environ.get(nazwa, domyslnie).strip()


def adres() -> tuple:
    u = urlparse(_env("VOXCPM_SERWER", "http://audiocpp:17496"))
    return (u.hostname or "audiocpp", u.port or 17496)


def _zarzadca(sciezka: str, dane: Optional[dict] = None, timeout: float = 30) -> dict:
    """GET/POST do zarzadcy kontenera `audiocpp`. Kontener wstaje razem z Voiceboxem,
    wiec odmowa polaczenia chwile po starcie = czekamy (do ~20 s), nie padamy."""
    url = _env("AUDIOCPP_ZARZADCA", "http://audiocpp:17495").rstrip("/") + sciezka
    req = urllib.request.Request(
        url,
        data=None if dane is None else json.dumps(dane).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if dane is None else "POST",
    )
    for proba in range(40):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            tresc = e.read().decode("utf-8", "replace")
            try:
                tresc = json.loads(tresc).get("blad", tresc)
            except ValueError:
                pass
            raise RuntimeError(f"zarzadca audiocpp {sciezka}: {tresc}") from e
        except (urllib.error.URLError, ConnectionError) as e:
            if proba == 39:
                raise RuntimeError(f"zarzadca audiocpp nie odpowiada ({url}): {e}") from e
            time.sleep(0.5)


def _url(sciezka: str) -> str:
    h, p = adres()
    return f"http://{h}:{p}{sciezka}"


def backend_procesu() -> Optional[str]:
    """hip | vulkan — na ktorej binarce FAKTYCZNIE chodzi silnik (wg zarzadcy)."""
    try:
        return _zarzadca("/stan", timeout=3).get("backend")
    except RuntimeError as e:
        logger.debug("gospodarz: zarzadca /stan nie odpowiada (%s)", e)
        return None


def wagi() -> Path:
    """Katalog z GGUF-em VoxCPM2 — audio.cpp bierze katalog, nie plik."""
    from ..config import MODELE_GLOSOWE

    return Path(_env("AUDIOCPP_WAGI", str(MODELE_GLOSOWE / "audiocpp" / "wagi")))


def wagi_omni(typ: str) -> Path:
    """Gotowy GGUF OmniVoice z upstreamu (`omnivoice-q8_0.gguf` / `omnivoice-f16.gguf`) —
    config, tokenizer tekstu i tokenizer audio siedza w srodku, silnik bierze sam plik
    (sidecary rozpakowuje do /tmp kontenera)."""
    return wagi() / "omnivoice" / f"omnivoice-{typ}.gguf"


def _katalog_wag(opcje: dict) -> Path:
    """GOTOWY plik dla tej precyzji generatora, jak lezy — inaczej oryginal.

    ⭐ `wagi-gotowe/<typ>/` robi `skrypty/zrob-wagi-audiocpp.sh <typ>`: tensory juz
    w typach, ktore silnik i tak by zrobil, wiec ladowanie to sam odczyt + wrzut
    (10.10: 2,2 s zamiast 7,3; z zimnego dysku 4,2 zamiast ~23). Bez gotowego
    pliku silnik przepisuje oryginal BF16 przy kazdym starcie — dziala, tylko wolno.
    Cache probek zostaje kluczowany oryginalem: gotowy plik daje te same wagi bit
    w bit, a klucz wpisu i tak zawiera typ wag.
    """
    nazwa = f'{opcje["voxcpm2.base_lm_weight_type"]}-{opcje["voxcpm2.weight_type"]}'
    gotowy = wagi().parent / "wagi-gotowe" / nazwa
    if (gotowy / "voxcpm2.gguf").is_file():
        return gotowy
    logger.info("gospodarz: brak gotowego pliku %s — silnik przepisze oryginal przy ladowaniu", gotowy)
    return wagi()


def stoi() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(adres()) == 0


def zapytaj(sciezka: str, dane: Optional[dict] = None, timeout: float = 120) -> dict:
    """GET (bez danych) albo POST JSON do gospodarza. Blad serwera → RuntimeError."""
    req = urllib.request.Request(
        _url(sciezka),
        data=None if dane is None else json.dumps(dane).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if dane is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"gospodarz {sciezka}: {e.read().decode('utf-8', 'replace')[:300]}") from e


def _wpis_voxa(opcje: dict, leniwy: bool) -> dict:
    """Wpis voxa do configu silnika — precyzja z panelu, cache probek, pojemnosc enkodera."""
    return {
        "id": VOX_ID,
        "family": "voxcpm2",
        "task": "tts",
        "mode": "streaming",
        "path": str(_katalog_wag(opcje)),
        "lazy": leniwy,
        "session_options": {
            **opcje,
            "voxcpm2.prompt_disk_cache_dir": _katalog_cache(),
            **_POJEMNOSC_ENKODERA,
        },
    }


def _uruchom(wpis: dict, backend: Optional[str] = None, wymus: bool = False) -> None:
    """Zarzadca podnosi silnik z TYM JEDNYM modelem (stary proces schodzi). `lazy: false`
    = wagi laduja sie przed otwarciem portu, wiec czekamy dluzej (gotowy plik vox ~2 s,
    z zimnego dysku ~4 s; oryginal z przepisywaniem ~7 / ~23 s)."""
    from . import tts_serwer

    backend = backend or tts_serwer.wybrany_backend()
    limit = 30 if wpis.get("lazy") else 300
    odp = _zarzadca("/uruchom", {
        "backend": backend, "model": wpis,
        "flagi": tts_serwer.flagi_dodatkowe(), "limit_s": limit, "wymus": wymus,
    }, timeout=limit + 30)
    opcje = wpis.get("session_options", {})
    logger.info("silnik audio.cpp wstal (%s, %s, %s) w %s ms — %s",
                wpis["id"], "lazy" if wpis.get("lazy") else "zaladowany", backend,
                odp.get("ms"), {k: v for k, v in opcje.items() if k.endswith("weight_type")})


def upewnij() -> None:
    """Gospodarz stoi — jak nie, podnosi go z voxem LENIWYM (precyzja z panelu)."""
    with _blokada:
        if stoi():
            return
        from . import tts_serwer

        _uruchom(_wpis_voxa(tts_serwer.opcje_wag(*tts_serwer.wybrane()), leniwy=True))


def stan() -> dict:
    """{'vox': {'session_options'}|None, 'omni': {'wariant'}|None, 'whisper': None} —
    tylko ZALADOWANE. Jeden model naraz, wiec najwyzej jedno z vox/omni nie jest None."""
    from . import tts_serwer

    pusto = {"vox": None, "omni": None, "whisper": None}
    if not stoi():
        return pusto
    try:
        modele = zapytaj("/v1/models?include_session_options=true", timeout=3).get("data", [])
    except Exception as e:
        logger.debug("gospodarz: /v1/models nie odpowiada (%s)", e)
        return pusto
    wynik = dict(pusto)
    for m in modele:
        if not m.get("loaded"):
            continue
        if m.get("id") == VOX_ID:
            wynik["vox"] = {"session_options": m.get("session_options") or {}}
        elif m.get("id") == OMNI_ID:
            # wariant z wpisu, z ktorym proces FAKTYCZNIE wstal (sciezka pliku u zarzadcy)
            wpis = (_zarzadca("/stan", timeout=3).get("konfig") or {}).get("model") or {}
            wariant = next((k for k, t in tts_serwer.OMNI_WARIANTY.items()
                            if wpis.get("path", "") == str(wagi_omni(t))), None)
            flash = (wpis.get("session_options") or {}).get("omnivoice.perf_mode") == "flash_attention"
            wynik["omni"] = {"wariant": wariant, "flash": flash}
    return wynik


def ma_wagi(opcje: dict, backend: Optional[str] = None) -> bool:
    """Vox siedzi DOKLADNIE z tymi typami wag i na tym backendzie (puste = z panelu)."""
    from . import tts_serwer

    if backend_procesu() != (backend or tts_serwer.wybrany_backend()):
        return False
    zal = (stan().get("vox") or {}).get("session_options", {})
    return all(zal.get(k) == v for k, v in opcje.items())


def zaladuj_voxa(opcje: dict, backend: Optional[str] = None) -> None:
    """Vox siedzi z ta precyzja i backendem → nic. Inaczej restart procesu z `lazy: false`."""
    with _blokada:
        if ma_wagi(opcje, backend):
            return
        _uruchom(_wpis_voxa(opcje, leniwy=False), backend)


def zdejmij_voxa() -> bool:
    """Zdejmuje voxa z gospodarza — VRAM wraca, proces zostaje. True = siedzial."""
    if stan().get("vox") is None:
        return False
    zapytaj("/v1/tasks/unload_models", {"model_ids": [VOX_ID]})
    return True


def _wpis_omni(typ: str, flash: bool, leniwy: bool) -> dict:
    """Wpis OmniVoice. Typy wag jak w pliku (defaulty upstreamu); `flash` =
    `omnivoice.perf_mode=flash_attention` — bez niego uwaga trzyma macierz T×T."""
    return {
        "id": OMNI_ID,
        "family": "omnivoice",
        "task": "tts",
        # pseudo-streaming upstreamu: jeden kawalek dzwieku na kawalek tekstu
        # (`text_chunk_size`); bez niego cala wypowiedz w jednym kawalku
        "mode": "streaming",
        "path": str(wagi_omni(typ)),
        "lazy": leniwy,
        "session_options": {"omnivoice.perf_mode": "flash_attention"} if flash else {},
    }


def ma_omni(wariant: str, flash: bool, backend: Optional[str] = None) -> bool:
    """OmniVoice siedzi DOKLADNIE z tym plikiem wag, trybem uwagi i backendem (puste = z panelu)."""
    from . import tts_serwer

    if backend_procesu() != (backend or tts_serwer.wybrany_backend()):
        return False
    o = stan().get("omni") or {}
    return o.get("wariant") == wariant and o.get("flash") == flash


def zaladuj_omni(wariant: str, flash: bool = True, backend: Optional[str] = None,
                 wymus: bool = False) -> None:
    """OmniVoice siedzi z tym plikiem → nic. Inaczej restart silnika z SAMYM nim
    (vox schodzi razem ze starym procesem — jeden model naraz)."""
    from . import tts_serwer

    typ = tts_serwer.OMNI_WARIANTY[wariant]
    if not wagi_omni(typ).is_file():
        raise RuntimeError(f"brak wag OmniVoice: {wagi_omni(typ)}")
    with _blokada:
        if not wymus and ma_omni(wariant, flash, backend):
            return
        _uruchom(_wpis_omni(typ, flash, leniwy=False), backend, wymus=wymus)


def zdejmij_omni() -> bool:
    """Zdejmuje OmniVoice z silnika — VRAM wraca, proces zostaje. True = siedzial."""
    if stan().get("omni") is None:
        return False
    zapytaj("/v1/tasks/unload_models", {"model_ids": [OMNI_ID]})
    return True
