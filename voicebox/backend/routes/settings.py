"""User settings endpoints — capture/refine and generation defaults."""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Literal, Optional

from .. import models
from ..database import get_db
from ..services import audiocpp_settings
from ..services import gpu_card
from ..services import settings as settings_service
from ..services import whisper_host

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
    changes = patch.model_dump(exclude_unset=True)
    result = settings_service.update_generation_settings(db, changes)
    # Whisper in its own container keeps a mirror of the choice, so it can come up without Voicebox.
    if any(k.startswith("whisper_") for k in changes):
        try:
            await asyncio.to_thread(whisper_host.sync)
        except RuntimeError as e:
            logger.warning("whisper: syncing the selection mirror failed (%s)", e)
    return result


# ── audio.cpp engines (Vox, OmniVoice) ──────────────────────────────────────
# base_lm  q8 | f16    the language part
# acoustic f16 | f32   the rest of the generator
#
# The precision is chosen when the engine STARTS, so a change in the panel needs a
# reload — these routes do it without touching the backend. The choice lives IN THE
# DATABASE (`generation_settings`) and both roads to the engine read it.


class WeightVariant(BaseModel):
    key: str
    file: str
    on_disk: bool
    size_mib: int


class TTSState(BaseModel):
    available: bool          # False = AUDIOCPP_ENGINE is empty, no engine at all
    vox_loaded: bool
    ms: Optional[int] = None  # how long the last (re)load took — only in reload replies
    # ⭐ Picked in the panel vs what really sits on the card. They differ exactly
    # until a reload, and the panel has to show it.
    loaded_base_lm: Optional[str] = None
    loaded_acoustic: Optional[str] = None
    selected_base_lm: Optional[str] = None
    selected_acoustic: Optional[str] = None
    # hip | vulkan — which binary the engine runs on vs the panel
    loaded_backend: Optional[str] = None
    selected_backend: Optional[str] = None
    # OmniVoice — the same engine, one model at a time (either it or Vox)
    omni_loaded: bool = False
    loaded_omni: Optional[str] = None
    loaded_omni_flash: Optional[bool] = None
    selected_omni: Optional[str] = None
    omni: list[WeightVariant] = []
    base_lm: list[WeightVariant] = []
    acoustic: list[WeightVariant] = []
    # Stream granularity — in decoder STEPS (~0.17 s of sound each).
    loaded_stream_min: Optional[int] = None
    loaded_stream_every: Optional[int] = None
    selected_stream_min: Optional[int] = None
    selected_stream_every: Optional[int] = None


class TTSReloadRequest(BaseModel):
    # Empty = reload with what the panel has.
    base_lm: Optional[Literal["q8", "f16", "f32"]] = None
    acoustic: Optional[Literal["f16", "f32"]] = None
    backend: Optional[Literal["hip", "vulkan"]] = None


@router.get("/generation/tts-state", response_model=TTSState)
async def get_tts_state_endpoint():
    return audiocpp_settings.state()


# ⚠️ A plain `def`, not `async def` — FastAPI runs it in the thread pool. As a coroutine
# it would block the event loop for the whole load (f16 ~10 s) and all of Voicebox would stall.
@router.post("/generation/tts-reload", response_model=TTSState)
def reload_tts_endpoint(request: TTSReloadRequest):
    # ⛔ BUG 11.10: the button always loaded the ggml Vox — NEXT TO a running Vox PT,
    # bypassing the Offloader (two models on the card). Now: Vox PT runs → reload IT
    # with the panel's set; otherwise ggml. Every road first unloads the other voices.
    from ..backends import evict_other_models, get_tts_backend_for_engine

    try:
        pt = get_tts_backend_for_engine("voxcpm_pt")
        if pt.is_loaded():
            evict_other_models("voxcpm_pt")
            pt.reload(request.base_lm, request.acoustic)
            return audiocpp_settings.state()
        evict_other_models("voxcpm")
        return audiocpp_settings.reload_vox(request.base_lm, request.acoustic, request.backend)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))


# OmniVoice — the same engine as Vox, one model at a time: reloading takes Vox down.
@router.post("/generation/omni-reload", response_model=TTSState)
def reload_omni_endpoint():
    from ..backends import evict_other_models

    try:
        evict_other_models("omnivoice")  # the same bug as Vox: Vox PT stayed next to it
        return audiocpp_settings.reload_omni()
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/generation/omni-unload", response_model=TTSState)
def unload_omni_endpoint():
    try:
        return audiocpp_settings.unload_omni()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


# ── whisper (whisper.cpp) — a separate process, a separate click from the voice ──
# The choice (model/quant/device) goes through the plain PUT /generation. These routes
# only show the process state and start/stop it.


@router.get("/generation/whisper-state")
async def get_whisper_state_endpoint():
    return whisper_host.state()


# ⚠️ A plain `def` — starting takes seconds and must not sit on the event loop.
@router.post("/generation/whisper-load")
def whisper_load_endpoint():
    # The "Load/Reload" button on the Whisper tab — restarts even with the same set.
    try:
        return whisper_host.start(force=True)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/generation/whisper-unload")
def whisper_unload_endpoint():
    whisper_host.stop()
    return whisper_host.state()


# ── the GPU for the models ───────────────────────────────────────────────────
# ⛔ The choice is HARD: the other card becomes INVISIBLE to ROCm and to Vulkan at
# once, so nothing can spill onto it. Details and the reason — `services/gpu_card.py`.
# It takes effect with the next model load (the engine hosts start the model's process
# with the visibility variables).


class GpuCard(BaseModel):
    index: int
    pci: str
    name: str
    vram_mib: int


class GpuState(BaseModel):
    cards: list[GpuCard]
    selected_pci: Optional[str]    # what the file holds
    resolved: Optional[int]        # the card the engines get on THIS machine
    reason: str                    # where this choice comes from — in plain words, for the panel


class GpuChoice(BaseModel):
    index: Optional[int]           # None = lift the restriction, every card visible


@router.get("/gpu", response_model=GpuState)
async def get_gpu_endpoint():
    return gpu_card.state()


@router.put("/gpu", response_model=GpuState)
async def set_gpu_endpoint(choice: GpuChoice):
    try:
        return gpu_card.select(choice.index)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"could not save: {e}")
