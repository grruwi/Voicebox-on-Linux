"""Model management endpoints."""

import asyncio
import shutil
from pathlib import Path

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import models
from ..utils.platform_detect import get_backend_type
from ..services.task_queue import create_background_task
from ..utils.progress import get_progress_manager
from ..utils.tasks import get_task_manager

router = APIRouter()


def _get_dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _copy_with_progress(src: Path, dst: Path, progress_manager, copied_so_far: int, total_bytes: int) -> int:
    """Copy a directory tree with byte-level progress tracking."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest_item = dst / item.name
        if item.is_dir():
            copied_so_far = _copy_with_progress(item, dest_item, progress_manager, copied_so_far, total_bytes)
        else:
            size = item.stat().st_size
            shutil.copy2(str(item), str(dest_item))
            copied_so_far += size
            progress_manager.update_progress(
                "migration",
                copied_so_far,
                total_bytes,
                filename=item.name,
                status="downloading",
            )
    return copied_so_far


@router.post("/models/load")
async def load_model(engine: str = "voxcpm"):
    """Load a TTS engine's model from disk (the Offloader makes room first)."""
    from ..backends import load_engine_model

    try:
        await load_engine_model(engine)
        return {"message": f"{engine} loaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/unload")
async def unload_model():
    """Unload every voice model — every engine host gives its VRAM back."""
    from ..services import engine_host, memory

    try:
        unloaded = await asyncio.to_thread(engine_host.offload_except, None)
        memory.release_memory("unload_model")
        return {"message": f"Unloaded: {', '.join(unloaded) or 'nothing was loaded'}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/load")
async def load_model_by_name(model_name: str):
    """Load a model from the Models screen — from disk, never downloaded."""
    from ..backends import get_model_config
    from ..services import whisper_host

    config = get_model_config(model_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    try:
        if config.engine == "whisper":
            # ⚠️ Starting `whisper-server` takes seconds — off the event loop.
            state = await asyncio.to_thread(whisper_host.start, config.model_size)
            return {"message": f"Model {model_name} loaded", **state}
        from ..backends import load_engine_model

        await load_engine_model(config.engine, config.model_size)
        return {"message": f"Model {model_name} loaded"}
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.post("/models/{model_name}/unload")
async def unload_model_by_name(model_name: str):
    """Unload a specific model from memory without deleting it from disk."""
    from ..backends import get_model_config, unload_model_by_config

    config = get_model_config(model_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    try:
        was_loaded = await asyncio.to_thread(unload_model_by_config, config)
        if not was_loaded:
            return {"message": f"Model {model_name} is not loaded"}
        # ⛔ Without this an unloaded model gives its room back to the ALLOCATOR, not the
        # kernel — `unload` reports success and RSS stays where it was.
        from ..services import memory

        memory.release_memory(f"unload:{model_name}")
        return {"message": f"Model {model_name} unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/models/progress/{model_name}")
async def get_model_progress(model_name: str):
    """Get model download progress via Server-Sent Events."""
    progress_manager = get_progress_manager()

    async def event_generator():
        async for event in progress_manager.subscribe(model_name):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/models/cache-dir")
async def get_models_cache_dir():
    """Get the path to the HuggingFace model cache directory."""
    from huggingface_hub import constants as hf_constants

    return {"path": str(Path(hf_constants.HF_HUB_CACHE))}


@router.post("/models/migrate")
async def migrate_models(request: models.ModelMigrateRequest):
    """Move all downloaded models to a new directory with byte-level progress via SSE."""
    from huggingface_hub import constants as hf_constants

    source = Path(hf_constants.HF_HUB_CACHE)
    destination = Path(request.destination)

    if not source.exists():
        raise HTTPException(status_code=404, detail="Current model cache directory not found")

    if source.resolve() == destination.resolve():
        raise HTTPException(status_code=400, detail="Source and destination are the same directory")

    if destination.resolve().is_relative_to(source.resolve()):
        raise HTTPException(status_code=400, detail="Destination cannot be inside the current cache directory")

    progress_manager = get_progress_manager()
    model_dirs = [d for d in source.iterdir() if d.name.startswith("models--") and d.is_dir()]
    if not model_dirs:
        progress_manager.update_progress("migration", 1, 1, status="complete")
        progress_manager.mark_complete("migration")
        return {"moved": 0, "errors": [], "source": str(source), "destination": str(destination)}

    destination.mkdir(parents=True, exist_ok=True)

    same_fs = False
    try:
        same_fs = source.stat().st_dev == destination.stat().st_dev
    except OSError:
        pass

    async def migrate_background():
        moved = 0
        errors = []
        try:
            if same_fs:
                total = len(model_dirs)
                for i, item in enumerate(model_dirs):
                    dest_item = destination / item.name
                    try:
                        if dest_item.exists():
                            shutil.rmtree(dest_item)
                        shutil.move(str(item), str(dest_item))
                        moved += 1
                        progress_manager.update_progress(
                            "migration",
                            i + 1,
                            total,
                            filename=item.name,
                            status="downloading",
                        )
                    except Exception as e:
                        errors.append(f"{item.name}: {str(e)}")
            else:
                total_bytes = sum(_get_dir_size(d) for d in model_dirs)
                progress_manager.update_progress(
                    "migration", 0, total_bytes, filename="Calculating...", status="downloading"
                )

                copied = 0
                for item in model_dirs:
                    dest_item = destination / item.name
                    try:
                        if dest_item.exists():
                            shutil.rmtree(dest_item)
                        copied = await asyncio.to_thread(
                            _copy_with_progress, item, dest_item, progress_manager, copied, total_bytes
                        )
                        await asyncio.to_thread(shutil.rmtree, str(item))
                        moved += 1
                    except Exception as e:
                        errors.append(f"{item.name}: {str(e)}")

            progress_manager.update_progress("migration", 1, 1, status="complete")
            progress_manager.mark_complete("migration")
        except Exception as e:
            progress_manager.update_progress("migration", 0, 0, status="error")
            progress_manager.mark_error("migration", str(e))

    create_background_task(migrate_background())

    return {"source": str(source), "destination": str(destination)}


@router.get("/models/migrate/progress")
async def get_migration_progress():
    """Get model migration progress via Server-Sent Events."""
    progress_manager = get_progress_manager()

    async def event_generator():
        async for event in progress_manager.subscribe("migration"):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/models/status", response_model=models.ModelStatusListResponse)
async def get_model_status():
    """Status of every model this build knows — from the disk, not the HuggingFace cache.

    ⛔ The weights live next to the engine containers (`download-models.sh`), OUTSIDE the
    HF cache, so the cache scan upstream did reported "to download" next to ready files.
    The backends are the source of truth here.
    """
    from ..backends import check_model_loaded, get_all_model_configs, get_tts_backend_for_engine
    from ..services import whisper_host

    def disk_size_mb(path: Path) -> Optional[float]:
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
        return None

    statuses = []
    for config in get_all_model_configs():
        downloaded, size_mb = False, None
        try:
            if config.engine == "whisper":
                _, quant, _ = whisper_host.selected()
                path = whisper_host.file(config.model_size, quant)
            else:
                backend = get_tts_backend_for_engine(config.engine)
                path = Path(backend._get_model_path())
                downloaded = backend._is_model_cached()
            downloaded = downloaded or path.is_file()
            if downloaded:
                size_mb = disk_size_mb(path)
        except Exception:
            pass
        try:
            loaded = await asyncio.to_thread(check_model_loaded, config)
        except Exception:
            loaded = False
        statuses.append(
            models.ModelStatus(
                model_name=config.model_name,
                display_name=config.display_name,
                hf_repo_id=config.hf_repo_id,
                downloaded=downloaded,
                downloading=False,
                size_mb=size_mb,
                loaded=loaded,
            )
        )

    return models.ModelStatusListResponse(models=statuses)


@router.post("/models/download")
async def trigger_model_download(request: models.ModelDownloadRequest):
    """⛔ Voicebox downloads nothing. The weights come from `download-models.sh` and
    live next to the engine containers."""
    raise HTTPException(
        status_code=400,
        detail=f"{request.model_name}: Voicebox does not download models — run download-models.sh",
    )


@router.post("/models/download/cancel")
async def cancel_model_download(request: models.ModelDownloadRequest):
    """Cancel or dismiss an errored/stale download task."""
    task_manager = get_task_manager()
    progress_manager = get_progress_manager()

    removed = task_manager.cancel_download(request.model_name)

    progress_removed = False
    with progress_manager._lock:
        if request.model_name in progress_manager._progress:
            del progress_manager._progress[request.model_name]
            progress_removed = True

    if removed or progress_removed:
        return {"message": f"Download task for {request.model_name} cancelled"}
    return {"message": f"No active task found for {request.model_name}"}


@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """⛔ Not from here: the weights live next to the engine containers, outside the
    HuggingFace cache, and the engines' containers mount them read-only."""
    from ..backends import get_model_config

    config = get_model_config(model_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    raise HTTPException(
        status_code=400,
        detail=f"{model_name}: the weights live in the voice models directory — delete them there by hand",
    )
