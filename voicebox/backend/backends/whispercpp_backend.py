"""
STT through whisper.cpp — `whisper-server` in the `whisper` container (`services/whisper_host.py`).

Why: upstream's `pytorch_backend` pulls whisper into Voicebox's process in PyTorch, onto
the same card as the voice. Here the model lives in a separate ggml process (Vulkan on
the GPU, or `-ng` on the CPU), comes up when loaded and goes when unloaded.

⚠️ Upstream's backend also has a BUG that goes away this way: `WhisperProcessor` without
`truncation=False` cuts the input at **30 seconds** and the rest is lost without a trace.
whisper.cpp handles long recordings by itself.

OpenAI-compatible endpoint: POST /v1/audio/transcriptions (multipart).
"""

import asyncio
import json
import logging
import mimetypes
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from ..services import whisper_host

logger = logging.getLogger(__name__)

_TIMEOUT_S = 600


def _multipart(fields: dict, path: Path) -> tuple[bytes, str]:
    """Put together by hand, so `requests` is not needed in the read-only venv."""
    boundary = f"----voicebox{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        )
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{path.name}\"\r\nContent-Type: {content_type}\r\n\r\n".encode()
    )
    parts.append(path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class WhisperCppSTTBackend:
    """STT with no weights in Voicebox's process. The model lives in `whisper-server`."""

    @property
    def model_size(self) -> str:
        """The model the process HOLDS; when nothing runs — the panel's one.

        `check_model_loaded` / `unload_model_by_config` compare this field with the
        registry entry (medium/large/turbo), so it has to describe the process.
        """
        loaded = whisper_host.loaded()
        if loaded and loaded[0]:
            return loaded[0]
        return whisper_host.selected()[0]

    @property
    def selected_size(self) -> str:
        """The panel's model — `/transcribe` uses it when the caller gives no `model`."""
        return whisper_host.selected()[0]

    def is_loaded(self) -> bool:
        return whisper_host.loaded() is not None

    # The layers above ask about the model cache to show a download bar. The ggml files
    # sit in the voice models directory (:ro) — `load_model` reports a missing one.
    def _is_model_cached(self, model_size: Optional[str] = None) -> bool:
        return True

    def _get_model_path(self, model_size: Optional[str] = None) -> str:
        model, quant, _ = whisper_host.selected()
        return str(whisper_host.file(model_size or model, quant))

    async def load_model(self, model_size: Optional[str] = None) -> None:
        """Starts `whisper-server` with the model; quantization and device from the panel."""
        await asyncio.to_thread(whisper_host.start, model_size)

    def unload_model(self) -> None:
        whisper_host.stop()

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
        prompt: Optional[str] = None,
        carry_initial_prompt: Optional[bool] = None,
        max_len: Optional[int] = None,
    ) -> str:
        await self.load_model(model_size)
        return await asyncio.to_thread(
            self._transcribe, audio_path, language, prompt, carry_initial_prompt, max_len
        )

    def _transcribe(
        self,
        audio_path: str,
        language: Optional[str],
        prompt: Optional[str],
        carry_initial_prompt: Optional[bool],
        max_len: Optional[int],
    ) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise RuntimeError(f"no such file: {audio_path}")

        fields = {"response_format": "json"}
        # ⛔ Without a language whisper.cpp detects it from the first 30 s (the server runs
        # with `-l auto`), and on purpose we do not override it when the caller gives
        # nothing — an English sentence forced into Polish comes out as gibberish.
        if language:
            fields["language"] = language
        # Dictation fields — empty ones are not sent, the server uses its defaults.
        if prompt:
            fields["prompt"] = prompt
        if carry_initial_prompt is not None:
            fields["carry_initial_prompt"] = "true" if carry_initial_prompt else "false"
        if max_len is not None:
            fields["max_len"] = str(max_len)

        data, content_type = _multipart(fields, path)
        req = urllib.request.Request(whisper_host.URL, data=data, headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
                reply = json.loads(r.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"whisper-server not answering ({e})") from e

        return (reply.get("text") or "").strip()
