"""User settings endpoints — capture/refine and generation defaults."""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Literal, Optional

from .. import models
from ..database import get_db
from ..services import karta as karta_service
from ..services import settings as settings_service
from ..services import tts_serwer
from ..services import whisper_serwer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/captures", response_model=models.CaptureSettingsResponse)
async def get_capture_settings_endpoint(db: Session = Depends(get_db)):
    return settings_service.get_capture_settings(db)


@router.put("/captures", response_model=models.CaptureSettingsResponse)
async def update_capture_settings_endpoint(
    patch: models.CaptureSettingsUpdate,
    db: Session = Depends(get_db),
):
    return settings_service.update_capture_settings(db, patch.model_dump(exclude_unset=True))


@router.get("/generation", response_model=models.GenerationSettingsResponse)
async def get_generation_settings_endpoint(db: Session = Depends(get_db)):
    return settings_service.get_generation_settings(db)


@router.put("/generation", response_model=models.GenerationSettingsResponse)
async def update_generation_settings_endpoint(
    patch: models.GenerationSettingsUpdate,
    db: Session = Depends(get_db),
):
    zmiany = patch.model_dump(exclude_unset=True)
    wynik = settings_service.update_generation_settings(db, zmiany)
    # Whisper w osobnym kontenerze trzyma lustro wyboru, zeby wstac bez Voiceboxa.
    if any(k.startswith("whisper_") for k in zmiany):
        try:
            await asyncio.to_thread(whisper_serwer.synchronizuj)
        except RuntimeError as e:
            logger.warning("whisper: lustro wyboru nie poszlo (%s)", e)
    return wynik


# ── wagi VoxCPM2 — DWA niezalezne pliki ──────────────────────────────────────
# base_lm  q8 | f16 | f32   czesc jezykowa
# akustyka f16 | f32        AudioVAE, czyli to, co lepi fale
#
# Wariant wybiera sie przy STARCIE `llama-tts-server`, wiec zmiana w panelu
# wymaga przeladowania serwera — te dwie trasy robia to bez ruszania backendu.
# ⭐ Od 15.09 wybor siedzi W BAZIE (`generation_settings`), nie w `ustawienia.env`,
# i czytaja go OBIE drogi do serwera.


class WariantWag(BaseModel):
    klucz: str
    plik: str
    jest_na_dysku: bool
    rozmiar_mib: int


class StanTTS(BaseModel):
    dostepny: bool          # False = backend chodzi w trybie CLI, nie ma serwera
    chodzi: bool
    wariant: Optional[str]  # base_lm, ktory serwer FAKTYCZNIE trzyma (/proc)
    ms: Optional[int] = None
    # ⭐ Zaznaczone w panelu kontra to, co realnie siedzi na karcie. Rozjezdzaja
    # sie dokladnie do chwili przeladowania i panel ma to pokazac.
    zaladowany_base_lm: Optional[str] = None
    zaladowany_akustyka: Optional[str] = None
    wybrany_base_lm: Optional[str] = None
    wybrany_akustyka: Optional[str] = None
    # hip | vulkan — z ktorej binarki chodzi gospodarz (/proc) kontra panel
    zaladowany_backend: Optional[str] = None
    wybrany_backend: Optional[str] = None
    # OmniVoice — ten sam silnik, jeden model naraz (albo on, albo vox)
    omni_chodzi: bool = False
    zaladowany_omni: Optional[str] = None
    zaladowany_omni_flash: Optional[bool] = None
    wybrany_omni: Optional[str] = None
    omni: list[WariantWag] = []
    base_lm: list[WariantWag] = []
    akustyka: list[WariantWag] = []
    # ⭐ Ziarnistosc strumienia — w KROKACH dekodera (~0,17 s dzwieku kazdy).
    # Ida do silnika SRODOWISKIEM, nie argumentem, wiec „zaladowane" czyta sie
    # z `/proc/<pid>/environ`.
    zaladowany_stream_min: Optional[int] = None
    zaladowany_stream_co: Optional[int] = None
    wybrany_stream_min: Optional[int] = None
    wybrany_stream_co: Optional[int] = None


class ZadanieReloadTTS(BaseModel):
    # Puste = przeladuj tym, co stoi w panelu.
    wariant: Optional[Literal["q8", "f16", "f32"]] = None
    akustyka: Optional[Literal["f16", "f32"]] = None
    backend: Optional[Literal["hip", "vulkan"]] = None


@router.get("/generation/tts-stan", response_model=StanTTS)
async def get_tts_stan_endpoint():
    return tts_serwer.stan()


# ⚠️ Zwykly `def`, nie `async def` — FastAPI odpala go w puli watkow. Jako
# korutyna blokowalby petle zdarzen na czas ladowania wag (f16 ~10 s) i caly
# Voicebox stalby w miejscu.
@router.post("/generation/tts-reload", response_model=StanTTS)
def reload_tts_endpoint(zadanie: ZadanieReloadTTS):
    try:
        return tts_serwer.przeladuj(zadanie.wariant, zadanie.akustyka, zadanie.backend)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))


# OmniVoice — ten sam silnik co vox, jeden model naraz: przeladowanie wysadza voxa.
@router.post("/generation/omni-reload", response_model=StanTTS)
def reload_omni_endpoint():
    try:
        return tts_serwer.przeladuj_omni()
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/generation/omni-rozladuj", response_model=StanTTS)
def rozladuj_omni_endpoint():
    try:
        return tts_serwer.zdejmij_omni()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


# ── whisper (whisper.cpp) — osobny proces, osobny klik od glosu ──────────────
# Wybor (model/kwant/urzadzenie) idzie przez zwykly PUT /generation. Te trasy
# tylko pokazuja stan procesu i go podnosza/zdejmuja.


@router.get("/generation/whisper-stan")
async def get_whisper_stan_endpoint():
    return whisper_serwer.stan()


# ⚠️ Zwykle `def` — start trwa sekundy i nie moze stac na petli zdarzen.
@router.post("/generation/whisper-zaladuj")
def whisper_zaladuj_endpoint():
    # Guzik „Zaladuj/Przeladuj" w zakladce Whisper — restart takze przy tym samym zestawie.
    try:
        return whisper_serwer.uruchom(wymus=True)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/generation/whisper-rozladuj")
def whisper_rozladuj_endpoint():
    whisper_serwer.zatrzymaj()
    return whisper_serwer.stan()


# ── wybor karty graficznej dla modeli ────────────────────────────────────────
# ⛔ Wybor jest TWARDY: niewybrana karta staje sie NIEWIDZIALNA dla ROCm i dla
# Vulkana naraz, wiec nic nie ma jak sie na nia rozlac. Szczegoly i powod —
# `services/karta.py`.
# ⚠️ Wchodzi PO RESTARCIE Voiceboxa: zmienne widocznosci czyta sie przy starcie
# procesu i backend nie przestawi ich sobie w locie. `wymaga_restartu` w
# odpowiedzi mowi wprost, czy plik rozjechal sie z tym, co proces widzi.
# ⭐ `wybrana` to zapis w pliku, `rozwiazana` to karta, ktora silniki DOSTANA —
# roznia sie po wyjeciu karty albo po przenosinach stacku na inna maszyne.


class Karta(BaseModel):
    indeks: int
    pci: str
    nazwa: str
    vram_mib: int


class StanKarty(BaseModel):
    karty: list[Karta]
    wybrana: Optional[int]         # co stoi w pliku
    aktywna: Optional[int]         # co widzi ZYWY proces
    rozwiazana: Optional[int]      # co launcher da silnikom po restarcie
    powod: str                     # skad ten wybor — po ludzku, do panelu
    wymaga_restartu: bool


class WyborKarty(BaseModel):
    indeks: Optional[int]          # None = zdejmij ograniczenie, obie widoczne


@router.get("/karta", response_model=StanKarty)
async def get_karta_endpoint():
    return karta_service.stan()


@router.put("/karta", response_model=StanKarty)
async def set_karta_endpoint(wybor: WyborKarty):
    try:
        return karta_service.ustaw(wybor.indeks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"nie udalo sie zapisac: {e}")
