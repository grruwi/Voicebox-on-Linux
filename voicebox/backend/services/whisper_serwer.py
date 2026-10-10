"""Whisper (whisper.cpp) — KLIENT osobnego kontenera `whisper` (od 10.10, decyzja grruwiego).

`whisper-server` nie rodzi sie juz w Voiceboxie. Zyje w kontenerze `whisper`
(`/klocki/apczan-kutasu/voicebox/whisper/`), ktorym rzadzi `zarzadca.py`; ten
modul tylko wysyla mu komendy. Dyktafon gada z kontenerem wprost
(127.0.0.1:17480), wiec Voicebox moze lezec albo mielic glos.

Wybor w panelu (`generation_settings`): model (medium/large/turbo), kwantyzacja
(q5/q8), urzadzenie (gpu/cpu), „przy starcie". PANEL JEST PRAWDA — zarzadca
trzyma tylko jego lustro (`whisper/dane/wybor.json`), zeby wstac z dobrym
modelem bez Voiceboxa. Lustro idzie przy kazdej zmianie w panelu, przy starcie
Voiceboxa i przed kazdym ladowaniem. Beam: `WHISPER_BEAM_SIZE` z env Voiceboxa.

Interfejs (`plik`, `wybrane`, `przy_starcie`, `zaladowany`, `uruchom`,
`zatrzymaj`, `stan`, `URL`) zostal jak przy procesie-dziecku — trasy i backend
STT sie nie zmienily. Stara wersja (proces-dziecko): `whisper_serwer.py.bak-przed-kontenera`.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from ..config import MODELE_GLOSOWE

logger = logging.getLogger(__name__)

KATALOG = MODELE_GLOSOWE / "whisper-cpp"
_WAGI = KATALOG / "wagi"

# (model, kwant) -> plik ggml. large-v3-q8_0 nie ma na HF — kwantyzowany recznie.
PLIKI = {
    ("medium", "q5"): "ggml-medium-q5_0.bin",
    ("medium", "q8"): "ggml-medium-q8_0.bin",
    ("large", "q5"): "ggml-large-v3-q5_0.bin",
    ("large", "q8"): "ggml-large-v3-q8_0.bin",
    ("turbo", "q5"): "ggml-large-v3-turbo-q5_0.bin",
    ("turbo", "q8"): "ggml-large-v3-turbo-q8_0.bin",
}
MODELE = ("medium", "large", "turbo")
KWANTY = ("q5", "q8")
URZADZENIA = ("gpu", "cpu")

# Domyslne = to, na czym stal whisper z hosta (large-v3-q5_0 na karcie).
_DOMYSLNIE = ("large", "q5", "gpu")

# Siec dockera `glos` (wspolna z kontenerem `whisper`). Z hosta: 127.0.0.1:17480.
ZARZADCA = "http://whisper:17480"
SCIEZKA = "/v1/audio/transcriptions"
URL = ZARZADCA + SCIEZKA


def _http(metoda: str, sciezka: str, dane: Optional[dict] = None, timeout: float = 5.0) -> dict:
    cialo = None if dane is None else json.dumps(dane).encode()
    req = urllib.request.Request(
        ZARZADCA + sciezka, data=cialo, method=metoda,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            detal = json.loads(e.read()).get("blad")
        except Exception:
            detal = None
        raise RuntimeError(f"kontener whisper: {detal or e}") from e
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"kontener whisper nie odpowiada ({e}) — `docker start whisper`") from e


def plik(model: str, kwant: str) -> Path:
    return _WAGI / PLIKI[(model, kwant)]


def wybrane() -> tuple:
    """(model, kwant, urzadzenie) z BAZY — panel wygrywa. Puste = domyslne."""
    model, kwant, urzadzenie = _DOMYSLNIE
    try:
        from ..database import session as db_session
        from ..database import GenerationSettings

        s = db_session.SessionLocal()
        try:
            w = s.query(GenerationSettings).first()
            if w is not None:
                model = getattr(w, "whisper_model", None) or model
                kwant = getattr(w, "whisper_kwant", None) or kwant
                urzadzenie = getattr(w, "whisper_urzadzenie", None) or urzadzenie
        finally:
            s.close()
    except Exception as e:
        logger.debug("whisper: wybor nie z bazy (%s) — biore domyslne", e)
    return (model, kwant, urzadzenie)


def przy_starcie() -> bool:
    """Przelacznik z panelu: czy whisper ma wstawac sam (teraz: przy starcie KONTENERA)."""
    try:
        from ..database import session as db_session
        from ..database import GenerationSettings

        s = db_session.SessionLocal()
        try:
            w = s.query(GenerationSettings).first()
            return bool(w is not None and getattr(w, "whisper_przy_starcie", False))
        finally:
            s.close()
    except Exception as e:
        logger.debug("whisper: przelacznik startu nie z bazy (%s)", e)
        return False


def flash() -> bool:
    """Flash attention whisper-servera z panelu — domyslnie TAK (jak whisper.cpp)."""
    try:
        from ..database import session as db_session
        from ..database import GenerationSettings

        s = db_session.SessionLocal()
        try:
            w = s.query(GenerationSettings).first()
            if w is not None and getattr(w, "whisper_fa", None) is not None:
                return bool(w.whisper_fa)
        finally:
            s.close()
    except Exception as e:
        logger.debug("whisper: flash nie z bazy (%s)", e)
    return True


def _zestaw(model: str, kwant: str, urzadzenie: str) -> dict:
    return {
        "model": model,
        "kwant": kwant,
        "urzadzenie": urzadzenie,
        "plik": PLIKI[(model, kwant)],
        # ⚠️ env_file czyta sie TYLKO przy tworzeniu kontenera Voiceboxa
        "bs": os.environ.get("WHISPER_BEAM_SIZE", "").strip(),
        "fa": flash(),
    }


def synchronizuj() -> None:
    """Lustro panelu → zarzadca. Niczego nie laduje."""
    model, kwant, urzadzenie = wybrane()
    _http("PUT", "/wybor", {**_zestaw(model, kwant, urzadzenie), "przy_starcie": przy_starcie()})


def _stan_zarzadcy() -> Optional[dict]:
    try:
        return _http("GET", "/stan", timeout=2.0)
    except RuntimeError as e:
        logger.debug("whisper: %s", e)
        return None


def zaladowany() -> Optional[tuple]:
    """(model, kwant, urzadzenie), ktore proces FAKTYCZNIE trzyma. None = nie chodzi."""
    z = (_stan_zarzadcy() or {}).get("zaladowany")
    if not z:
        return None
    return (z.get("model"), z.get("kwant"), z.get("urzadzenie"))


def zatrzymaj() -> bool:
    """Zdejmuje whispera w kontenerze (VRAM wraca dopiero z koncem procesu)."""
    try:
        return bool(_http("POST", "/zdejmij", timeout=20.0).get("zdjety"))
    except RuntimeError as e:
        logger.warning("whisper: %s", e)
        return False


def uruchom(model: Optional[str] = None, kwant: Optional[str] = None,
            urzadzenie: Optional[str] = None, wymus: bool = False) -> dict:
    """Podnosi whispera w kontenerze. Puste argumenty = to, co w panelu.

    Gdy juz chodzi dokladnie z tym zestawem — zarzadca nic nie robi.
    """
    dom_model, dom_kwant, dom_urz = wybrane()
    model = model or dom_model
    kwant = kwant or dom_kwant
    urzadzenie = urzadzenie or dom_urz
    if (model, kwant) not in PLIKI:
        raise ValueError(f"nieznany whisper: {model}/{kwant}")
    if urzadzenie not in URZADZENIA:
        raise ValueError(f"nieznane urzadzenie: {urzadzenie}")

    synchronizuj()
    s = _http("POST", "/zaladuj", {**_zestaw(model, kwant, urzadzenie), "wymus": wymus}, timeout=70.0)
    if "ms" in s:
        logger.info("whisper: %s/%s na %s wstal w %d ms (kontener)", model, kwant, urzadzenie, s["ms"])
        return {**stan(s), "ms": s["ms"]}
    return stan(s)


def stan(s: Optional[dict] = None) -> dict:
    if s is None:
        s = _stan_zarzadcy()
    z = (s or {}).get("zaladowany")
    # wagi z kontenera (to on je laduje); bez kontenera — z naszego montazu
    wagi = (s or {}).get("wagi")
    model, kwant, urzadzenie = wybrane()

    def _plik(n: str) -> tuple:
        if wagi is not None:
            return (n in wagi, wagi.get(n, 0))
        p = _WAGI / n
        return (p.is_file(), int(p.stat().st_size / 1048576) if p.is_file() else 0)

    return {
        "chodzi": bool(z),
        "kontener": s is not None,
        "zaladowany_model": z.get("model") if z else None,
        "zaladowany_kwant": z.get("kwant") if z else None,
        "zaladowane_urzadzenie": z.get("urzadzenie") if z else None,
        "zaladowany_fa": z.get("fa", True) if z else None,
        "wybrany_fa": flash(),
        "wybrany_model": model,
        "wybrany_kwant": kwant,
        "wybrane_urzadzenie": urzadzenie,
        "pliki": [
            {
                "model": m,
                "kwant": k,
                "plik": n,
                "jest_na_dysku": _plik(n)[0],
                "rozmiar_mib": _plik(n)[1],
            }
            for (m, k), n in PLIKI.items()
        ],
    }
