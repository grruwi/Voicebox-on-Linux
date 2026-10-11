"""OpenAI compatibility: /v1/audio/{speech,voices,models,transcriptions}.

Why: Open WebUI, Newelle and the like only know the OpenAI standard — without this
the microphone and the speaker in their chat have nothing to talk to. Base URL in
their settings: `http://<voicebox-host>:17493/v1`, any API key (not checked).

Speech goes through the same queue as the GUI and MCP (`generate_speech` →
`run_generation`), so one job at a time, the panel's knobs and a history entry
(source "openai"). ⛔ The sound does NOT play on the server's speakers — the client
that asked plays it (the browser may be on a phone). Exceptions:
`_source_is_openai` in services/generation.py and AGENT_SOURCES in the frontend.

`voice` = a profile name (case-insensitive). Empty = the default profile from the
MCP panel. `model` is ignored — the engine comes from the profile.
`instructions` = Vox's parentheses (`(…)` in front of the text) — ONLY for Vox
profiles. OmniVoice and the rest get them in the `instruct` field (Omni: a voice
description), never in the text.
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

# response_format → (Content-Type, ffmpeg output arguments). `None` = send the WAV
# from disk without re-encoding. `pcm` as in OpenAI: raw s16le mono 24 kHz.
FORMATS = {
    "mp3": ("audio/mpeg", ["-c:a", "libmp3lame", "-q:a", "2", "-f", "mp3"]),
    "opus": ("audio/ogg", ["-c:a", "libopus", "-b:a", "64k", "-f", "ogg"]),
    "aac": ("audio/aac", ["-c:a", "aac", "-b:a", "128k", "-f", "adts"]),
    "flac": ("audio/flac", ["-c:a", "flac", "-f", "flac"]),
    "pcm": ("audio/pcm", ["-ac", "1", "-ar", "24000", "-f", "s16le"]),
    "wav": ("audio/wav", None),
}

# Vox's parentheses limit — a longer one the backend does not detect, and it silently
# falls back to continuation.
PARENTHESES_MAX = 200
# `GenerationRequest.instruct` (models.py, max_length)
INSTRUCT_MAX = 500


class SpeechRequest(BaseModel):
    input: str
    voice: str | None = None
    model: str | None = None
    response_format: str = "mp3"
    instructions: str | None = None
    speed: float | None = None  # Vox has no speed as a number — ignored
    # ⭐ Real streaming (11.10): `stream: true` or `stream_format` = the sound goes out while
    # it is computed. "audio" = the encoded bytes as they come, "sse" = OpenAI's
    # `speech.audio.delta` events (base64). Nothing set = the whole file at the end, as before.
    stream: bool | None = None
    stream_format: str | None = None


def _panel_settings(db: Session) -> dict:
    """Normalization, chunking and crossfade from the panel — same as MCP.

    Pydantic has its own defaults there (True / 800 / 50 ms), which would otherwise
    override the sliders in the GUI.
    """
    from ..database.models import GenerationSettings

    fields: dict = {}
    try:
        settings = db.query(GenerationSettings).first()
        if settings is not None:
            fields["normalize"] = bool(settings.normalize_audio)
            if settings.max_chunk_chars is not None:
                fields["max_chunk_chars"] = settings.max_chunk_chars
            if settings.crossfade_ms is not None:
                fields["crossfade_ms"] = settings.crossfade_ms
    except Exception:
        pass
    return fields


async def _wait_for_end(generation_id: str, queue, db: Session) -> DBGeneration:
    """Wait for `speak-end` from the event bus; the database every second as a net.

    The bus drops events when its queue is full (maxsize 64), so waiting on it
    alone could hang forever.
    """
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=1.0)
            if event.get("generation_id") != generation_id:
                continue
        except asyncio.TimeoutError:
            pass
        db.expire_all()
        row = db.query(DBGeneration).filter_by(id=generation_id).first()
        if row is None:
            raise HTTPException(status_code=500, detail="The generation vanished from the database.")
        if row.status in ("completed", "failed"):
            return row


async def _transcode(path: str, arguments: list[str]) -> bytes:
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, *arguments, "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    output, errors = await process.communicate()
    if process.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg: {errors.decode(errors='replace').strip()}")
    return output


@router.post("/speech")
async def speech(data: SpeechRequest, db: Session = Depends(get_db)):
    format_ = (data.response_format or "mp3").lower()
    if format_ not in FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Format '{format_}' not supported. Available: {', '.join(FORMATS)}",
        )
    text = data.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty `input`.")

    profile = resolve_profile(data.voice or None, None, db)
    if profile is None:
        if data.voice:
            raise HTTPException(
                status_code=404,
                detail=f"Profile '{data.voice}' does not exist. List: GET /v1/audio/voices",
            )
        raise HTTPException(
            status_code=400,
            detail="No `voice` and no default profile (Voicebox → Settings → MCP).",
        )

    instruction = (data.instructions or "").strip()
    # The engine as in `_resolve_generation_engine` (engine=None below): only Vox
    # understands the parentheses — OmniVoice would read them aloud, so it gets the
    # instruction in a separate field.
    engine = getattr(profile, "default_engine", None) or getattr(profile, "preset_engine", None) or ""
    instruct = None
    if instruction and engine.startswith("voxcpm"):
        if len(instruction) > PARENTHESES_MAX:
            raise HTTPException(
                status_code=400,
                detail=f"`instructions` has {len(instruction)} characters, Vox reads parentheses up to {PARENTHESES_MAX}.",
            )
        text = f"({instruction}){text}"
    elif instruction:
        if len(instruction) > INSTRUCT_MAX:
            raise HTTPException(
                status_code=400,
                detail=f"`instructions` has {len(instruction)} characters, the limit is {INSTRUCT_MAX}.",
            )
        instruct = instruction

    from .generations import generate_speech

    stream_format = (data.stream_format or ("audio" if data.stream else "")).lower()
    if stream_format and stream_format not in ("audio", "sse"):
        raise HTTPException(status_code=400, detail="`stream_format`: audio or sse.")
    if stream_format:
        return await _speech_streamed(profile, text, instruct, format_, stream_format, db)

    # Subscribe BEFORE queueing — otherwise a short generation could finish before
    # we start listening.
    queue = mcp_events.subscribe()
    try:
        generation = await generate_speech(
            models.GenerationRequest(
                profile_id=profile.id,
                text=text,
                language=profile.language or "en",
                engine=None,
                instruct=instruct,
                **_panel_settings(db),
            ),
            db,
            source="openai",
        )
        row = await _wait_for_end(generation.id, queue, db)
    finally:
        mcp_events.unsubscribe(queue)

    if row.status == "failed":
        raise HTTPException(status_code=500, detail=row.error or "The generation failed.")
    path = config.resolve_storage_path(row.audio_path)
    if not path or not path.exists():
        raise HTTPException(status_code=500, detail="No audio file after the generation.")

    media_type, arguments = FORMATS[format_]
    if arguments is None:
        return FileResponse(path, media_type=media_type)
    return Response(content=await _transcode(str(path), arguments), media_type=media_type)


# Formats for a stream: what `ffmpeg` can encode on the fly. `wav` is built in Python
# (a header without a length + 16-bit samples at the engine's rate).
STREAM_ARGUMENTS = {name: arguments for name, (_, arguments) in FORMATS.items() if arguments}


async def _speech_streamed(profile, text: str, instruct, format_: str, stream_format: str, db: Session):
    """The generation through the queue with `client_stream` — chunks as they are made.

    A generation that cannot stream (panel normalization on, effects, text over the chunk
    limit) gives no stream: then the whole file goes out at the end, in the same shape
    (one SSE delta, or the plain body).
    """
    from fastapi.responses import StreamingResponse

    from ..services import client_stream
    from .generations import generate_speech

    generation = await generate_speech(
        models.GenerationRequest(
            profile_id=profile.id,
            text=text,
            language=profile.language or "en",
            engine=None,
            instruct=instruct,
            **_panel_settings(db),
        ),
        db,
        source="openai",
        client_stream=True,
    )
    media_type = FORMATS[format_][0]
    stream = await client_stream.wait_decision(generation.id, db)

    if stream is not None:
        if format_ == "wav":
            chunks = client_stream.wav(stream)
        else:
            chunks = client_stream.encoded(stream, STREAM_ARGUMENTS[format_])
    else:
        row = await client_stream.wait_end(generation.id, db)
        if row.status == "failed":
            raise HTTPException(status_code=500, detail=row.error or "The generation failed.")
        path = config.resolve_storage_path(row.audio_path)
        if not path or not path.exists():
            raise HTTPException(status_code=500, detail="No audio file after the generation.")
        arguments = FORMATS[format_][1]
        body = path.read_bytes() if arguments is None else await _transcode(str(path), arguments)
        chunks = client_stream.once(body)

    if stream_format == "sse":
        return StreamingResponse(client_stream.as_sse(chunks), media_type="text/event-stream",
                                 headers=client_stream.HEADERS)
    return StreamingResponse(chunks, media_type=media_type, headers=client_stream.HEADERS)


@router.get("/voices")
async def voices(db: Session = Depends(get_db)):
    """Voicebox profiles as voices. id = the name, because Newelle has it typed in by hand."""
    profiles = (
        db.query(DBVoiceProfile)
        .filter(DBVoiceProfile.voice_type != "import")
        .order_by(DBVoiceProfile.name)
        .all()
    )
    return {"voices": [{"id": p.name, "name": p.name} for p in profiles]}


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
    """Whisper from the panel. `model` passes only when it is one of our sizes
    (medium/large/turbo) — "whisper-1" from OpenAI clients = the panel's choice."""
    from ..services import whisper_host
    from .transcription import transcribe_audio

    result = await transcribe_audio(
        file=file,
        language=language or None,
        model=model if model in whisper_host.MODELS else None,
        prompt=prompt,
        carry_initial_prompt=None,
        max_len=None,
    )
    if response_format == "text":
        return PlainTextResponse(result.text)
    if response_format == "verbose_json":
        return {"text": result.text, "duration": result.duration, "language": language}
    return {"text": result.text}
