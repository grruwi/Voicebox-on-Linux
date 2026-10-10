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
from . import history, pamiec, profiles
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

        # ⏱ ZEGAR DIAGNOSTYCZNY (16.08, tymczasowy — do zdjecia po ustaleniu,
        # gdzie siedzi staly narzut generacji). Log: "[ZEGAR] ...".
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

        # ⭐ STRUMIEN — dzwiek leci na glosniki W TRAKCIE liczenia, zamiast po.
        # Zmierzone: pierwszy dzwiek po 0,25 s zamiast po calosci. ⚠️ CALKOWITY
        # CZAS SIE NIE ZMIENIA — to nie jest przyspieszenie, tylko skrocenie
        # ciszy przed pierwszym slowem.
        #
        # Kazdy warunek nizej odpada, bo POTRZEBUJE CALEGO PRZEBIEGU:
        #   normalize / effects_chain — licza sie na komplecie probek,
        #   retry / regenerate        — nadpisuja istniejaca wersje,
        #   dlugi tekst              — `generate_chunked` tnie go i skleja z
        #                              przenikaniem, czego strumien nie robi.
        # Silnik bez `generate_stream` po prostu tu nie wchodzi — zaden inny
        # backend nie wymagal zmian.
        audio = None
        zagrane_strumieniem = False
        if (
            mode == "generate"
            and not normalize
            and not effects_chain
            and hasattr(tts_model, "generate_stream")
            and len(text) <= (max_chunk_chars or 800)
            and not _zrodlo_openai(generation_id, bg_db)
            and (
                _zrodlo_jest_agenta(generation_id, bg_db)
                or _zrodlo_twitch(generation_id, bg_db)
                or not _front_zagra_sam(bg_db)
            )
        ):
            audio, sample_rate = await _graj_strumieniem(
                tts_model,
                text,
                voice_prompt,
                language=language,
                seed=seed,
                instruct=instruct,
                env=_env_odtwarzacza(_zrodlo_twitch(generation_id, bg_db)),
            )
            zagrane_strumieniem = audio is not None

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
            "[ZEGAR] audio=%.2fs | referencja=%.2f  inferencja=%.2f  "
            "normalizacja=%.2f  zapis+efekty=%.2f  RAZEM=%.2f",
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
        # ⛔ Nie gramy drugi raz tego, co juz poszlo strumieniem na glosniki.
        if not zagrane_strumieniem:
            # ⚠️ Przy ZGASZONYM autoplay front nie gra NICZEGO — wiec generacja,
            # ktora wypadla ze strumienia (za dlugi tekst, efekty, normalizacja,
            # silnik bez `generate_stream`), skonczylaby CISZA. Wtedy gramy plik
            # z backendu, tak samo jak mowe agenta.
            _autoplay_if_agent(
                generation_id, final_path, bg_db,
                takze_z_gui=not _front_zagra_sam(bg_db),
            )
        _notify_speak_end(generation_id, status="completed")
    finally:
        task_manager.complete_generation(generation_id)
        bg_db.close()
        # Oddaj cache alokatora — model zostaje, znika tylko narosly zapas.
        # Bez tego jedyna droga do odzyskania VRAM jest wyrzucenie wag.
        pamiec.zwolnij_pamiec("run_generation")


def _front_zagra_sam(db) -> bool:
    """Czy przegladarka odtworzy ten plik po zakonczeniu generacji.

    ⚠️ To decyduje, czy strumien wolno puscic przy kliknieciu w GUI. Strumien
    gra z BACKENDU przez `ffplay`; gdyby front zagral to samo z pliku, dzwiek
    poszedlby DWA RAZY. Przy zgaszonym „autoplay on generate" front milczy,
    wiec strumien jest jedynym zrodlem i echa nie ma.

    ⛔ Cena zgaszonego autoplay: silnik BEZ `generate_stream` (vox ggml, S1)
    nie zagra z GUI wcale — plik laduje w historii i trzeba go kliknac.
    """
    try:
        from ..database.models import GenerationSettings

        ustawienia = db.query(GenerationSettings).first()
        return bool(ustawienia.autoplay_on_generate) if ustawienia else True
    except Exception:
        return True  # w razie watpliwosci NIE gramy z backendu — echo gorsze niz cisza


def _zrodlo_jest_agenta(generation_id: str, db) -> bool:
    """Czy to mowa agenta (MCP/REST), czy klikniecie w GUI.

    Front gra swoje sam, wiec strumien na glosniki dotyczy tylko tego pierwszego.
    """
    try:
        from ..database.models import Generation as DBGeneration

        wiersz = db.query(DBGeneration).filter_by(id=generation_id).first()
        return bool(wiersz and wiersz.source in ("mcp", "rest"))
    except Exception:
        return False


# ⭐ ZRODLO "twitch" (2026-09-29): TTS widzow z /klocki/stream-hub/twitch-tts.
# Gra tym samym strumieniem co mowa agenta (TTFT ~0,3 s), ale na KANALE
# "Twitch TTS", nie na domyslnym. Dwie zmienne dla ffplay (klient pulse):
#   PULSE_SINK        — na ktory sink
#   PULSE_PROP        — wlasna nazwa, bo klodzio-audio-router przeklada kazdy
#                       node "ffplay" na kanal Klodzia, czyli do "System"
# Bez tego TTS lecial podwojnie: raz systemem, raz swoim kanalem.
def _zrodlo_twitch(generation_id: str, db) -> bool:
    try:
        from ..database.models import Generation as DBGeneration

        wiersz = db.query(DBGeneration).filter_by(id=generation_id).first()
        return bool(wiersz and wiersz.source == "twitch")
    except Exception:
        return False


# ⛔ ZRODLO "openai" (2026-10-01): /v1/audio/speech dla WebUI i Newelle.
# Plik wraca do klienta w odpowiedzi HTTP, a przegladarka moze stac na
# telefonie — glosniki kombajnu maja milczec, takze przy zgaszonym autoplay.
def _zrodlo_openai(generation_id: str, db) -> bool:
    try:
        from ..database.models import Generation as DBGeneration

        wiersz = db.query(DBGeneration).filter_by(id=generation_id).first()
        return bool(wiersz and wiersz.source == "openai")
    except Exception:
        return False


def _env_odtwarzacza(twitch: bool):
    if not twitch:
        return None
    import os

    return {**os.environ, "PULSE_SINK": "twitch-tts",
            "PULSE_PROP": "application.name=twitch-tts-play"}


async def _graj_strumieniem(
    backend,
    text: str,
    voice_prompt: dict,
    *,
    language: str,
    seed: int | None,
    instruct: str | None,
    env: dict | None = None,
):
    """Graj kawalki na glosnikach w miare powstawania; zwroc caly dzwiek.

    Dwie rzeczy naraz: bajty ida do `ffplay` przez rure, a kopia laduje w
    liscie — inaczej nie byloby czego zapisac do historii i `/audio/{id}`
    oddawaloby pustke.

    ⚠️ NIGDY NIE RZUCA. Zwraca `(None, 0)`, gdy cokolwiek zawiedzie — wtedy
    `run_generation` leci stara droga i uzytkownik dostaje dzwiek mimo wszystko.
    Strumien jest wygoda, nie warunkiem dzialania.

    ⛔ `ffplay` dostaje GOLE PROBKI, bez naglowka WAV — stad `-f f32le` i reszta
    opisu formatu we flagach. Naglowek niesie dlugosc, ktorej na starcie nie znamy.
    """
    import numpy as np

    czestotliwosc = getattr(backend, "SAMPLE_RATE", 48000)
    odtwarzacz = None
    kawalki: list = []

    try:
        # ⛔ Bez `-probesize 32 -analyzeduration 0 -fflags nobuffer` ffplay „bada”
        # rurę, aż uzbiera kilka sekund dźwięku, i dopiero wtedy otwiera wyjście:
        # zmierzone 30.09 — strumień w PipeWire +3,4 s po POST, choć TTFT 0,2 s.
        # Z flagami otwiera się 1 ms po pierwszym kawałku.
        odtwarzacz = subprocess.Popen(
            ["ffplay", "-f", "f32le", "-ar", str(czestotliwosc), "-ch_layout", "mono",
             "-probesize", "32", "-analyzeduration", "0", "-fflags", "nobuffer",
             "-nodisp", "-autoexit", "-loglevel", "error", "-i", "pipe:0"],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        _t_zadanie = time.perf_counter()
        async for kawalek in backend.generate_stream(
            text, voice_prompt, language=language, seed=seed, instruct=instruct
        ):
            if not kawalki:
                # TTFT — cisza przed pierwszym slowem. Jedyna liczba, ktora
                # decyduje, czy to jeszcze rozmowa, czy juz odczyt komunikatu.
                logger.info(
                    "[ZEGAR] TTFT=%.3fs | znakow=%d",
                    time.perf_counter() - _t_zadanie, len(text),
                )
            kawalki.append(kawalek)
            try:
                # ⛔ W WATKU, nie wprost. Rura do ffplay ma ~64 KB (0,3 s dzwieku),
                # a ffplay czyta w tempie odtwarzania — `write` blokuje, az dogra.
                # Wprost w petli zdarzen zamrazalo to CALY backend na czas mowy:
                # dyktafon dostawal timeout, /transcribe czekal (zmierzone 30.09).
                await asyncio.to_thread(odtwarzacz.stdin.write, kawalek.tobytes())
            except (BrokenPipeError, ValueError):
                # Odtwarzacz zszedl (np. ubity recznie) — liczymy dalej,
                # zeby generacja mimo to trafila do historii.
                odtwarzacz = None
                break

        if odtwarzacz is not None:
            odtwarzacz.stdin.close()

        if not kawalki:
            return None, 0

        logger.info("Strumien: zagrane %d kawalkow @ %d Hz", len(kawalki), czestotliwosc)
        return np.concatenate(kawalki), czestotliwosc

    except Exception:
        logger.exception("Strumien nie wyszedl — wracam do zwyklej generacji")
        if odtwarzacz is not None:
            try:
                odtwarzacz.kill()
            except Exception:
                pass
        return None, 0


def _autoplay_if_agent(
    generation_id: str, audio_path: str | None, db, *, takze_z_gui: bool = False
) -> None:
    """Zagraj wypowiedz agenta na glosnikach — z WNETRZA kontenera.

    Luka, ktora to zatyka: generacje ze zrodel agentowych ("mcp", "rest")
    swiadomie omijaja autoplay w glownym oknie (patrz routes/generations.py),
    bo tam gra pigulka we froncie. Bez otwartego frontu nikt ich nie gral —
    plik ladowal na dysk i cisza.

    Warunek dzialania: socket PipeWire (w trybie zgodnosci z PulseAudio)
    wpuszczony do kontenera przez compose jako /run/pulse/native. Bez niego
    ffplay wychodzi bledem, ktory tu tylko logujemy — dzwiek nigdy nie moze
    wywrocic zakonczonej generacji.
    """
    if not audio_path:
        return

    try:
        from ..database.models import Generation as DBGeneration

        row = db.query(DBGeneration).filter_by(id=generation_id).first()
        if not row:
            return
        if row.source == "openai":
            return  # gra klient, ktory pytal (WebUI/Newelle), nie kombajn
        if row.source not in ("mcp", "rest", "twitch") and not takze_z_gui:
            return  # front sam zagra swoje

        path = config.resolve_storage_path(audio_path)
        if not path or not path.exists():
            logger.warning("Autoplay: brak pliku %s", audio_path)
            return

        # -nodisp: bez okna SDL. -autoexit: konczy sie razem z plikiem.
        # Puszczamy w tle (Popen bez wait) — kolejka generacji nie ma
        # stac przez czas mowienia.
        subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)],
            env=_env_odtwarzacza(row.source == "twitch"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Autoplay: gram %s", path.name)
    except Exception:
        logger.exception("Autoplay: nie udalo sie zagrac %s", generation_id)


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
        pamiec.zwolnij_pamiec("generate_audio_sync")


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
