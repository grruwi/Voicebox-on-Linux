"""
Unified TTS generation orchestration.

Replaces the three near-identical closures (_run_generation, _run_retry,
_run_regenerate) that lived in main.py with a single ``run_generation()``
function parameterized by *mode*.

Mode differences:
  - "generate"   : full pipeline -- save clean version, optionally apply
                    effects and create a processed version.
  - "retry"      : re-runs a failed generation with the same seed.
                    No effects, no version creation.
  - "regenerate" : re-runs with seed=None for variation.  Creates a new
                    version with an auto-incremented "take-N" label.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
import traceback
from typing import Literal, Optional

from .. import config
from . import browser_stream, history, memory, profiles
from ..database import get_db
from ..utils.tasks import get_task_manager

logger = logging.getLogger(__name__)


async def run_generation(
    *,
    generation_id: str,
    profile_id: str,
    text: str,
    language: str,
    engine: str,
    model_size: str,
    seed: Optional[int],
    normalize: bool = False,
    effects_chain: Optional[list] = None,
    instruct: Optional[str] = None,
    mode: Literal["generate", "retry", "regenerate"],
    max_chunk_chars: Optional[int] = None,
    crossfade_ms: Optional[int] = None,
    version_id: Optional[str] = None,
    preset_voice_id: Optional[str] = None,
    client_stream: bool = False,
) -> None:
    """Execute TTS inference and persist the result.

    This is the single entry point for all background generation work.
    It is designed to be enqueued via ``services.task_queue.enqueue_generation``.
    """
    from ..backends import (
        engine_needs_trim,
        engine_retries_runaway,
        get_tts_backend_for_engine,
        load_engine_model,
    )
    from ..utils.chunked_tts import generate_chunked
    from ..utils.audio import has_tts_runaway, normalize_audio, save_audio, trim_tts_output

    task_manager = get_task_manager()
    bg_db = next(get_db())

    try:
        tts_model = get_tts_backend_for_engine(engine)

        if not tts_model.is_loaded():
            await history.update_generation_status(generation_id, "loading_model", bg_db)

        await load_engine_model(engine, model_size)

        # ⏱ A DIAGNOSTIC CLOCK — where the fixed overhead of a generation sits.
        # Log: "[TIMING] ...".
        _t0 = time.perf_counter()

        voice_prompt = await profiles.create_voice_prompt_for_profile(
            profile_id,
            bg_db,
            use_cache=True,
            engine=engine,
            preset_voice_override=preset_voice_id,
        )
        _t1 = time.perf_counter()

        await history.update_generation_status(generation_id, "generating", bg_db)
        trim_fn = trim_tts_output if engine_needs_trim(engine) else None
        runaway_detector = has_tts_runaway if engine_retries_runaway(engine) else None

        gen_kwargs: dict = dict(
            language=language,
            seed=seed if mode != "regenerate" else None,
            instruct=instruct,
            trim_fn=trim_fn,
            runaway_detector=runaway_detector,
        )
        if max_chunk_chars is not None:
            gen_kwargs["max_chunk_chars"] = max_chunk_chars
        if crossfade_ms is not None:
            gen_kwargs["crossfade_ms"] = crossfade_ms

        # ⭐ STREAMING — the sound goes out WHILE it is computed, not after.
        # Measured: first sound after 0.25 s instead of after the whole thing. ⚠️ THE TOTAL
        # TIME DOES NOT CHANGE — it is not a speed-up, only less silence before the first word.
        #
        # Every condition below drops out because it NEEDS THE WHOLE TAKE:
        #   normalize / effects_chain — computed on all the samples,
        #   retry / regenerate        — overwrite an existing version,
        #   long text                 — `generate_chunked` cuts it and joins the pieces
        #                               with a crossfade, which the stream does not do.
        # An engine without `generate_stream` simply does not come in here.
        audio = None
        streamed = False
        streamable = (
            mode == "generate"
            and not normalize
            and not effects_chain
            and hasattr(tts_model, "generate_stream")
            and len(text) <= (max_chunk_chars or 800)
            and (client_stream or not _source_is_openai(generation_id, bg_db))
        )
        # ⭐ `client_stream` (11.10, grruwi): the API client that asked wants the chunks
        # itself (`stream: true` on /v1/audio/speech, /generate/stream, `?client_stream=true`
        # on /generate) — they go to `browser_stream`, never to the speakers.
        twitch = streamable and _source_is_twitch(generation_id, bg_db)
        to_speakers = streamable and not client_stream and (
            _source_is_agent(generation_id, bg_db)
            or twitch
            or not _frontend_plays_itself(bg_db)
        )
        # ⭐ AUTOPLAY CHECKED (11.10, grruwi) = a generation from the GUI streams to the
        # BROWSER, not to `ffplay` — the panel plays the chunks as they are made, instead
        # of the file after the whole thing (`services/browser_stream.py`). Unchecked = ffplay.
        to_browser = streamable and not to_speakers
        if to_speakers or to_browser:
            audio, sample_rate = await _play_streaming(
                tts_model,
                text,
                voice_prompt,
                language=language,
                seed=seed,
                instruct=instruct,
                env=_player_env(twitch),
                browser=generation_id if to_browser else None,
            )
            streamed = audio is not None
        else:
            browser_stream.refuse(generation_id)

        if audio is None:
            audio, sample_rate = await generate_chunked(tts_model, text, voice_prompt, **gen_kwargs)
        _t2 = time.perf_counter()

        # --- Normalize (generate and regenerate always; retry skips) -----
        if normalize or mode == "regenerate":
            audio = normalize_audio(audio)
        _t3 = time.perf_counter()

        duration = len(audio) / sample_rate

        # --- Persist audio and update status -----------------------------
        if mode == "generate":
            final_path = _save_generate(
                generation_id=generation_id,
                audio=audio,
                sample_rate=sample_rate,
                effects_chain=effects_chain,
                save_audio=save_audio,
                db=bg_db,
            )
        elif mode == "retry":
            final_path = _save_retry(
                generation_id=generation_id,
                audio=audio,
                sample_rate=sample_rate,
                save_audio=save_audio,
            )
        elif mode == "regenerate":
            final_path = _save_regenerate(
                generation_id=generation_id,
                version_id=version_id,
                audio=audio,
                sample_rate=sample_rate,
                save_audio=save_audio,
                db=bg_db,
            )

        _t4 = time.perf_counter()
        logger.info(
            "[TIMING] audio=%.2fs | reference=%.2f  inference=%.2f  "
            "normalize=%.2f  save+effects=%.2f  TOTAL=%.2f",
            duration, _t1 - _t0, _t2 - _t1, _t3 - _t2, _t4 - _t3, _t4 - _t0,
        )

        await history.update_generation_status(
            generation_id=generation_id,
            status="completed",
            db=bg_db,
            audio_path=final_path,
            duration=duration,
        )

    except asyncio.CancelledError:
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=bg_db,
            error="Generation cancelled",
        )
        _notify_speak_end(generation_id, status="cancelled")
    except Exception as e:
        traceback.print_exc()
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=bg_db,
            error=str(e),
        )
        _notify_speak_end(generation_id, status="failed")
    else:
        # ⛔ Do not play a second time what already went to the speakers as a stream.
        if not streamed:
            # ⚠️ With autoplay UNCHECKED the frontend plays NOTHING — so a generation that
            # dropped out of streaming (text too long, effects, normalize, an engine without
            # `generate_stream`) would end in SILENCE. Then the backend plays the file, the
            # same as an agent's speech.
            _autoplay_if_agent(
                generation_id, final_path, bg_db,
                also_from_gui=not _frontend_plays_itself(bg_db),
            )
        _notify_speak_end(generation_id, status="completed")
    finally:
        task_manager.complete_generation(generation_id)
        bg_db.close()
        # Give the allocator cache back — the model stays, only the grown reserve goes.
        memory.release_memory("run_generation")


def _frontend_plays_itself(db) -> bool:
    """Whether the browser plays this file once the generation is done.

    ⚠️ This decides whether a click in the GUI may stream to the speakers. That stream
    plays from the BACKEND through `ffplay`; if the frontend played the same file, the
    sound would go TWICE. With "autoplay on generate" unchecked the frontend stays silent,
    so the stream is the only source and there is no echo.
    """
    try:
        from ..database.models import GenerationSettings

        settings = db.query(GenerationSettings).first()
        return bool(settings.autoplay_on_generate) if settings else True
    except Exception:
        return True  # when in doubt the backend does NOT play — an echo is worse than silence


def _source(generation_id: str, db) -> Optional[str]:
    try:
        from ..database.models import Generation as DBGeneration

        row = db.query(DBGeneration).filter_by(id=generation_id).first()
        return row.source if row else None
    except Exception:
        return None


def _source_is_agent(generation_id: str, db) -> bool:
    """An agent's speech (MCP/REST) or a click in the GUI.

    The frontend plays its own, so streaming to the speakers concerns the former only.
    """
    return _source(generation_id, db) in ("mcp", "rest")


# ⭐ SOURCE "twitch" (2026-09-29): viewers' TTS from stream-hub/twitch-tts. Plays with the
# same stream as an agent's speech (TTFT ~0.3 s), but on the "Twitch TTS" CHANNEL, not the
# default one. Two variables for ffplay (a pulse client):
#   PULSE_SINK        — which sink
#   PULSE_PROP        — its own name, because klodzio-audio-router moves every "ffplay"
#                       node to Klodzio's channel, i.e. to "System"
# Without them the TTS went out twice: once through the system, once on its own channel.
def _source_is_twitch(generation_id: str, db) -> bool:
    return _source(generation_id, db) == "twitch"


# ⛔ SOURCE "openai" (2026-10-01): /v1/audio/speech for WebUI and Newelle. The file goes
# back to the client in the HTTP response, and the browser may be on a phone — the
# machine's speakers stay silent, also with autoplay unchecked.
def _source_is_openai(generation_id: str, db) -> bool:
    return _source(generation_id, db) == "openai"


def _player_env(twitch: bool):
    if not twitch:
        return None
    import os

    return {**os.environ, "PULSE_SINK": "twitch-tts",
            "PULSE_PROP": "application.name=twitch-tts-play"}


async def _play_streaming(
    backend,
    text: str,
    voice_prompt: dict,
    *,
    language: str,
    seed: int | None,
    instruct: str | None,
    env: dict | None = None,
    browser: str | None = None,
):
    """Play the chunks on the speakers as they are made; return the whole sound.

    `browser` = the generation id: the chunks go to `browser_stream` (the panel plays
    them through Web Audio) INSTEAD of `ffplay`.

    Two things at once: the bytes go to `ffplay` through a pipe, and a copy lands in a
    list — otherwise there would be nothing to save to the history and `/audio/{id}`
    would return emptiness.

    ⚠️ NEVER RAISES. Returns `(None, 0)` when anything fails — then `run_generation` takes
    the old road and the user gets the sound anyway. Streaming is a convenience, not a
    condition.

    ⛔ `ffplay` gets RAW SAMPLES, no WAV header — hence `-f f32le` and the rest of the
    format description in the flags. The header carries a length we do not know at the start.
    """
    import numpy as np

    sample_rate = getattr(backend, "SAMPLE_RATE", 48000)
    player = None
    stream = None
    chunks: list = []

    try:
        if browser:
            stream = browser_stream.open_stream(browser, sample_rate)
        else:
            # ⛔ Without `-probesize 32 -analyzeduration 0 -fflags nobuffer` ffplay "probes"
            # the pipe until it has a few seconds of sound, and only then opens the output:
            # measured 30.09 — the stream in PipeWire +3.4 s after the POST, though TTFT was
            # 0.2 s. With the flags it opens 1 ms after the first chunk.
            player = subprocess.Popen(
                ["ffplay", "-f", "f32le", "-ar", str(sample_rate), "-ch_layout", "mono",
                 "-probesize", "32", "-analyzeduration", "0", "-fflags", "nobuffer",
                 "-nodisp", "-autoexit", "-loglevel", "error", "-i", "pipe:0"],
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        t_request = time.perf_counter()
        async for chunk in backend.generate_stream(
            text, voice_prompt, language=language, seed=seed, instruct=instruct
        ):
            if not chunks:
                # TTFT — the silence before the first word. The one number that decides
                # whether this is still a conversation or already reading an announcement.
                logger.info(
                    "[TIMING] TTFT=%.3fs | chars=%d",
                    time.perf_counter() - t_request, len(text),
                )
            chunks.append(chunk)
            if stream is not None:
                stream.add(chunk.tobytes())
                continue
            try:
                # ⛔ IN A THREAD, not directly. The pipe to ffplay holds ~64 KB (0.3 s of
                # sound), and ffplay reads at playback speed — `write` blocks until it has
                # played. Directly on the event loop this froze the WHOLE backend for the
                # length of the speech: the dictation timed out, /transcribe waited
                # (measured 30.09).
                await asyncio.to_thread(player.stdin.write, chunk.tobytes())
            except (BrokenPipeError, ValueError):
                # The player went away (e.g. killed by hand) — keep computing, so the
                # generation still lands in the history.
                player = None
                break

        if player is not None:
            player.stdin.close()
        if stream is not None:
            stream.finish()

        if not chunks:
            return None, 0

        logger.info("Stream: played %d chunks @ %d Hz", len(chunks), sample_rate)
        return np.concatenate(chunks), sample_rate

    except Exception:
        logger.exception("Streaming failed — falling back to a plain generation")
        if stream is not None:
            stream.finish(failed=True)
        if player is not None:
            try:
                player.kill()
            except Exception:
                pass
        return None, 0
    finally:
        # A cancelled generation (CancelledError is not an Exception) — the browser would
        # wait for the next chunk forever.
        if stream is not None and not stream.done:
            stream.finish(failed=True)


def _autoplay_if_agent(
    generation_id: str, audio_path: str | None, db, *, also_from_gui: bool = False
) -> None:
    """Play an agent's speech on the speakers — from INSIDE the container.

    The gap this closes: generations from agent sources ("mcp", "rest") skip the autoplay
    in the main window on purpose (see routes/generations.py), because the pill in the
    frontend plays them. With no frontend open nobody played them — the file landed on
    disk and silence.

    Condition: the PipeWire socket (in PulseAudio compatibility mode) let into the
    container by compose as /run/pulse/native. Without it ffplay exits with an error we
    only log — sound can never break a finished generation.
    """
    if not audio_path:
        return

    try:
        from ..database.models import Generation as DBGeneration

        row = db.query(DBGeneration).filter_by(id=generation_id).first()
        if not row:
            return
        if row.source == "openai":
            return  # the client who asked plays it (WebUI/Newelle), not the machine
        if row.source not in ("mcp", "rest", "twitch") and not also_from_gui:
            return  # the frontend plays its own

        path = config.resolve_storage_path(audio_path)
        if not path or not path.exists():
            logger.warning("Autoplay: no file %s", audio_path)
            return

        # -nodisp: no SDL window. -autoexit: ends together with the file.
        # Run in the background (Popen without wait) — the generation queue must not
        # stand still for the length of the speech.
        subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)],
            env=_player_env(row.source == "twitch"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Autoplay: playing %s", path.name)
    except Exception:
        logger.exception("Autoplay: could not play %s", generation_id)


def _notify_speak_end(generation_id: str, *, status: str) -> None:
    """Publish a speak-end event; the frontend ignores unknown ids."""
    try:
        from ..mcp_server import events as mcp_events

        mcp_events.publish(
            "speak-end",
            {"generation_id": generation_id, "status": status},
        )
    except Exception:
        # Never let event pub/sub break generation completion.
        pass


def _save_generate(
    *,
    generation_id: str,
    audio,
    sample_rate: int,
    effects_chain: Optional[list],
    save_audio,
    db,
) -> str:
    """Save clean version and optionally an effects-processed version.

    Returns the final audio path (processed if effects were applied,
    otherwise clean).
    """
    from . import versions as versions_mod

    clean_audio_path = config.get_generations_dir() / f"{generation_id}.wav"
    save_audio(audio, str(clean_audio_path), sample_rate)

    has_effects = effects_chain and any(e.get("enabled", True) for e in effects_chain)

    versions_mod.create_version(
        generation_id=generation_id,
        label="original",
        audio_path=config.to_storage_path(clean_audio_path),
        db=db,
        effects_chain=None,
        is_default=not has_effects,
    )

    final_audio_path = str(clean_audio_path)

    if has_effects:
        from ..utils.effects import apply_effects, validate_effects_chain

        assert effects_chain is not None

        error_msg = validate_effects_chain(effects_chain)
        if error_msg:
            import logging
            logging.getLogger(__name__).warning("invalid effects chain, skipping: %s", error_msg)
            versions_mod.set_default_version(
                versions_mod.list_versions(generation_id, db)[0].id, db
            )
        else:
            processed_audio = apply_effects(audio, sample_rate, effects_chain)
            processed_path = config.get_generations_dir() / f"{generation_id}_processed.wav"
            save_audio(processed_audio, str(processed_path), sample_rate)
            final_audio_path = str(processed_path)
            versions_mod.create_version(
                generation_id=generation_id,
                label="version-2",
                audio_path=config.to_storage_path(processed_path),
                db=db,
                effects_chain=effects_chain,
                is_default=True,
            )

    return config.to_storage_path(final_audio_path)


def _save_retry(
    *,
    generation_id: str,
    audio,
    sample_rate: int,
    save_audio,
) -> str:
    """Save retry output -- single file, no versions.

    Returns the audio path.
    """
    audio_path = config.get_generations_dir() / f"{generation_id}.wav"
    save_audio(audio, str(audio_path), sample_rate)
    return config.to_storage_path(audio_path)


async def generate_audio_sync(
    *,
    profile_id: str,
    text: str,
    language: str,
    engine: str,
    model_size: str,
    seed: Optional[int] = None,
    instruct: Optional[str] = None,
    normalize: bool = True,
    max_chunk_chars: Optional[int] = None,
    crossfade_ms: Optional[int] = None,
    preset_voice_id: Optional[str] = None,
) -> bytes:
    """Run a TTS generation synchronously and return the resulting wav bytes.

    Unlike :func:`run_generation`, this path does not touch the
    ``generations`` table, enqueue work, or write anything to the
    generations directory. It's used by ``POST /profiles/{id}/speak``
    when the caller passes ``persist=false`` — they just want the audio
    back in the HTTP response without polluting their history.

    Loads the engine model on demand, runs ``generate_chunked``, optional
    normalize, then encodes in-memory via :func:`tts.audio_to_wav_bytes`
    (same helper ``/generate/stream`` uses).
    """
    from ..backends import (
        engine_needs_trim,
        engine_retries_runaway,
        get_tts_backend_for_engine,
        load_engine_model,
    )
    from ..utils.chunked_tts import generate_chunked
    from ..utils.audio import has_tts_runaway, normalize_audio, trim_tts_output
    from . import tts

    bg_db = next(get_db())
    try:
        tts_model = get_tts_backend_for_engine(engine)
        await load_engine_model(engine, model_size)

        voice_prompt = await profiles.create_voice_prompt_for_profile(
            profile_id,
            bg_db,
            use_cache=True,
            engine=engine,
            preset_voice_override=preset_voice_id,
        )
    finally:
        bg_db.close()

    trim_fn = trim_tts_output if engine_needs_trim(engine) else None
    runaway_detector = has_tts_runaway if engine_retries_runaway(engine) else None

    gen_kwargs: dict = dict(
        language=language,
        seed=seed,
        instruct=instruct,
        trim_fn=trim_fn,
        runaway_detector=runaway_detector,
    )
    if max_chunk_chars is not None:
        gen_kwargs["max_chunk_chars"] = max_chunk_chars
    if crossfade_ms is not None:
        gen_kwargs["crossfade_ms"] = crossfade_ms

    try:
        audio, sample_rate = await generate_chunked(
            tts_model, text, voice_prompt, **gen_kwargs
        )

        if normalize:
            audio = normalize_audio(audio)

        return tts.audio_to_wav_bytes(audio, sample_rate)
    finally:
        # Druga trasa generacji (m.in. /generate/stream) — ten sam obowiazek.
        memory.release_memory("generate_audio_sync")


def _save_regenerate(
    *,
    generation_id: str,
    version_id: Optional[str],
    audio,
    sample_rate: int,
    save_audio,
    db,
) -> str:
    """Save regeneration output as a new version with auto-label.

    Returns the audio path.
    """
    from . import versions as versions_mod

    import uuid as _uuid

    suffix = _uuid.uuid4().hex[:8]
    audio_path = config.get_generations_dir() / f"{generation_id}_{suffix}.wav"
    save_audio(audio, str(audio_path), sample_rate)

    # Count via DB query rather than list length to avoid TOCTOU race
    from ..database import GenerationVersion as DBGenerationVersion

    count = db.query(DBGenerationVersion).filter_by(generation_id=generation_id).count()
    label = f"take-{count + 1}"

    versions_mod.create_version(
        generation_id=generation_id,
        label=label,
        audio_path=config.to_storage_path(audio_path),
        db=db,
        effects_chain=None,
        is_default=True,
    )

    return config.to_storage_path(audio_path)
