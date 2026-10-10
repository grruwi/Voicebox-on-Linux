"""Zgodnosc z OpenAI: /v1/audio/{speech,voices,models,transcriptions}.

Po co: Open WebUI i Newelle znaja tylko standard OpenAI — bez tego mikrofon
i glosnik w ich czacie nie maja do czego gadac. Adres bazowy w ich panelu:
`http://kombajn:17494/v1`, klucz API dowolny (nie sprawdzamy).

Mowa idzie ta sama kolejka co GUI i MCP (`generate_speech` → `run_generation`),
wiec jedno zadanie naraz, pokretla z panelu i wpis w historii (zrodlo
"openai"). ⛔ Dzwiek NIE leci na glosniki kombajnu — gra go klient, ktory
pytal (przegladarka moze byc na telefonie). Wylaczenia: `_zrodlo_openai`
w services/generation.py i AGENT_SOURCES we froncie.

`voice` = nazwa profilu (bez rozrozniania wielkosci liter). Puste = profil
domyslny z panelu MCP. `model` ignorowany — silnik bierze sie z profilu.
`instructions` = nawias Voxa: idzie przed tekst jako `(…)`.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .. import config, models
from ..database import Generation as DBGeneration, VoiceProfile as DBVoiceProfile, get_db
from ..mcp_server import events as mcp_events
from ..mcp_server.resolve import resolve_profile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/audio")

# response_format → (Content-Type, argumenty ffmpeg na wyjsciu). `None` = oddaj WAV
# z dysku bez przekodowania. `pcm` jak u OpenAI: surowe s16le mono 24 kHz.
FORMATY = {
    "mp3": ("audio/mpeg", ["-c:a", "libmp3lame", "-q:a", "2", "-f", "mp3"]),
    "opus": ("audio/ogg", ["-c:a", "libopus", "-b:a", "64k", "-f", "ogg"]),
    "aac": ("audio/aac", ["-c:a", "aac", "-b:a", "128k", "-f", "adts"]),
    "flac": ("audio/flac", ["-c:a", "flac", "-f", "flac"]),
    "pcm": ("audio/pcm", ["-ac", "1", "-ar", "24000", "-f", "s16le"]),
    "wav": ("audio/wav", None),
}

# Limit nawiasu Voxa — dluzszy backend nie wykryje i po cichu wraca kontynuacja.
NAWIAS_MAX = 200


class SpeechRequest(BaseModel):
    input: str
    voice: str | None = None
    model: str | None = None
    response_format: str = "mp3"
    instructions: str | None = None
    speed: float | None = None  # Vox nie ma tempa jako liczby — ignorowane


def _ustawienia_panelu(db: Session) -> dict:
    """Normalizacja, ciecie i przenikanie z panelu — tak samo jak MCP.

    Pydantic ma tam wlasne defaulty (True / 800 / 50 ms), ktore bez tego
    przebilyby suwaki w GUI.
    """
    from ..database.models import GenerationSettings

    pola: dict = {}
    try:
        ustawienia = db.query(GenerationSettings).first()
        if ustawienia is not None:
            pola["normalize"] = bool(ustawienia.normalize_audio)
            if ustawienia.max_chunk_chars is not None:
                pola["max_chunk_chars"] = ustawienia.max_chunk_chars
            if ustawienia.crossfade_ms is not None:
                pola["crossfade_ms"] = ustawienia.crossfade_ms
    except Exception:
        pass
    return pola


async def _czekaj_na_koniec(generation_id: str, kolejka, db: Session) -> DBGeneration:
    """Czekaj na `speak-end` z szyny zdarzen; baza co sekunde jako siatka.

    Szyna gubi zdarzenia przy pelnej kolejce (maxsize 64), wiec samo
    czekanie na nia mogloby wisiec w nieskonczonosc.
    """
    while True:
        try:
            zdarzenie = await asyncio.wait_for(kolejka.get(), timeout=1.0)
            if zdarzenie.get("generation_id") != generation_id:
                continue
        except asyncio.TimeoutError:
            pass
        db.expire_all()
        wiersz = db.query(DBGeneration).filter_by(id=generation_id).first()
        if wiersz is None:
            raise HTTPException(status_code=500, detail="Generacja zniknela z bazy.")
        if wiersz.status in ("completed", "failed"):
            return wiersz


async def _przekoduj(sciezka: str, argumenty: list[str]) -> bytes:
    proces = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", sciezka, *argumenty, "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    wyjscie, bledy = await proces.communicate()
    if proces.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg: {bledy.decode(errors='replace').strip()}")
    return wyjscie


@router.post("/speech")
async def speech(data: SpeechRequest, db: Session = Depends(get_db)):
    format_ = (data.response_format or "mp3").lower()
    if format_ not in FORMATY:
        raise HTTPException(
            status_code=400,
            detail=f"Format '{format_}' nieobslugiwany. Dostepne: {', '.join(FORMATY)}",
        )
    tekst = data.input.strip()
    if not tekst:
        raise HTTPException(status_code=400, detail="Pusty `input`.")

    profil = resolve_profile(data.voice or None, None, db)
    if profil is None:
        if data.voice:
            raise HTTPException(
                status_code=404,
                detail=f"Profil '{data.voice}' nie istnieje. Lista: GET /v1/audio/voices",
            )
        raise HTTPException(
            status_code=400,
            detail="Brak `voice` i brak profilu domyslnego (Voicebox → Settings → MCP).",
        )

    instrukcja = (data.instructions or "").strip()
    if instrukcja:
        if len(instrukcja) > NAWIAS_MAX:
            raise HTTPException(
                status_code=400,
                detail=f"`instructions` ma {len(instrukcja)} znakow, Vox czyta nawias do {NAWIAS_MAX}.",
            )
        tekst = f"({instrukcja}){tekst}"

    from .generations import generate_speech

    # Subskrypcja PRZED kolejka — inaczej krotka generacja moglaby skonczyc
    # sie, zanim zaczniemy sluchac.
    kolejka = mcp_events.subscribe()
    try:
        generacja = await generate_speech(
            models.GenerationRequest(
                profile_id=profil.id,
                text=tekst,
                language=profil.language or "en",
                engine=None,
                **_ustawienia_panelu(db),
            ),
            db,
            source="openai",
        )
        wiersz = await _czekaj_na_koniec(generacja.id, kolejka, db)
    finally:
        mcp_events.unsubscribe(kolejka)

    if wiersz.status == "failed":
        raise HTTPException(status_code=500, detail=wiersz.error or "Generacja padla.")
    sciezka = config.resolve_storage_path(wiersz.audio_path)
    if not sciezka or not sciezka.exists():
        raise HTTPException(status_code=500, detail="Brak pliku po generacji.")

    typ, argumenty = FORMATY[format_]
    if argumenty is None:
        return FileResponse(sciezka, media_type=typ)
    return Response(content=await _przekoduj(str(sciezka), argumenty), media_type=typ)


@router.get("/voices")
async def voices(db: Session = Depends(get_db)):
    """Profile Voiceboxa jako glosy. id = nazwa, bo w Newelle wpisuje sie ja recznie."""
    profile = (
        db.query(DBVoiceProfile)
        .filter(DBVoiceProfile.voice_type != "import")
        .order_by(DBVoiceProfile.name)
        .all()
    )
    return {"voices": [{"id": p.name, "name": p.name} for p in profile]}


@router.get("/models")
async def audio_models():
    return {"models": [{"id": "voicebox"}]}


@router.post("/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
):
    """Whisper z panelu. `model` przechodzi tylko, gdy to nasz rozmiar
    (medium/large/turbo) — „whisper-1" od klientow OpenAI = wybor z panelu."""
    from ..backends import WHISPER_HF_REPOS
    from .transcription import transcribe_audio

    wynik = await transcribe_audio(
        file=file,
        language=language or None,
        model=model if model in WHISPER_HF_REPOS else None,
        prompt=prompt,
        carry_initial_prompt=None,
        max_len=None,
    )
    if response_format == "text":
        return PlainTextResponse(wynik.text)
    if response_format == "verbose_json":
        return {"text": wynik.text, "duration": wynik.duration, "language": language}
    return {"text": wynik.text}
