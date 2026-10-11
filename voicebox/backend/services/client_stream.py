"""Real streaming for API clients (11.10, grruwi's call).

The same pipe the panel uses (`browser_stream`): `run_generation` with `client_stream=True`
puts the engine's chunks in as they are made, and the routes hand them to the client
while the rest is still being computed. Before this, every API road waited for the
whole take: `/v1/audio/speech` returned the finished file, and upstream's
`/generate/stream` generated everything first and only then cut the WAV into pieces.

Roads (all through the one generation queue, all land in the history as "openai"):
  POST /v1/audio/speech  `stream: true` or `stream_format: "audio"|"sse"`
  POST /generate/stream  a WAV whose header has no length, data as it is computed
  POST /generate?client_stream=true  →  GET /generate/{id}/browser-stream  (raw f32le)

⚠️ When the generation cannot stream (loudness normalization on, effects, text longer
than the chunk limit, an engine without `generate_stream`) the decision is "not
streamed" — the routes then wait for the file and send it whole. Nothing breaks, the
first sound just comes later.
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import time
from typing import AsyncIterator, Optional

from . import browser_stream


async def wait_decision(generation_id: str, db, timeout_s: float = 600) -> Optional[browser_stream.Stream]:
    """The stream once `run_generation` decides; None = not streamed (or already over).

    The generation may wait in the queue behind another — hence the wait. The database
    is the net: a generation that failed before the decision would leave us hanging.
    Raises LookupError when there is no such generation.
    """
    from ..database import Generation as DBGeneration

    deadline = time.monotonic() + timeout_s
    next_db_read = 0.0
    while True:
        decided, stream = browser_stream.get(generation_id)
        if decided:
            return stream
        now = time.monotonic()
        if now > deadline:
            return None
        if now >= next_db_read:
            next_db_read = now + 0.25
            db.expire_all()
            status = db.query(DBGeneration.status).filter_by(id=generation_id).scalar()
            if status is None and not db.query(DBGeneration.id).filter_by(id=generation_id).first():
                raise LookupError(generation_id)
            if status in ("completed", "failed"):
                return None
        await asyncio.sleep(0.02)


def wav_header(sample_rate: int, bits: int = 16) -> bytes:
    """A WAV header for a stream: the lengths are unknown, so 0xFFFFFFFF (players read on)."""
    block = bits // 8
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * block, block, bits)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


async def wav(stream: browser_stream.Stream) -> AsyncIterator[bytes]:
    """The stream as a 16-bit WAV at the engine's own rate — no ffmpeg, no resampling."""
    import numpy as np

    reader = stream.read()
    yield wav_header(int.from_bytes(await reader.__anext__(), "little"))
    async for chunk in reader:
        samples = np.frombuffer(chunk, dtype=np.float32)
        yield (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


async def encoded(stream: browser_stream.Stream, arguments: list[str]) -> AsyncIterator[bytes]:
    """The stream's f32le through `ffmpeg` (`arguments` = output format), chunk by chunk.

    `-flush_packets 1`: every encoded packet goes out at once — without it ffmpeg holds
    output back in its own buffer and the first sound arrives seconds late.
    ⛔ `-probesize 32 -analyzeduration 0 -fflags nobuffer` on the INPUT — without them
    ffmpeg "probes" the pipe until it has seconds of sound, even with the format given
    (measured 11.10: first byte after 2.4 s of a 2.9 s take). Same lesson as ffplay.
    """
    reader = stream.read()
    sample_rate = int.from_bytes(await reader.__anext__(), "little")
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-probesize", "32", "-analyzeduration", "0", "-fflags", "nobuffer",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        *arguments, "-flush_packets", "1", "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    async def feed() -> None:
        try:
            async for chunk in reader:
                process.stdin.write(chunk)
                await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                process.stdin.close()
            except Exception:
                pass

    feeder = asyncio.create_task(feed())
    try:
        while True:
            data = await process.stdout.read(8192)
            if not data:
                break
            yield data
    finally:
        # The client went away (or the end) — the generation itself keeps going and still
        # lands in the history; only the encoder goes.
        feeder.cancel()
        if process.returncode is None:
            process.kill()
        await process.wait()


async def as_sse(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """OpenAI's `stream_format: "sse"`: `speech.audio.delta` events with base64 audio,
    then `speech.audio.done`."""
    async for chunk in chunks:
        event = {"type": "speech.audio.delta", "audio": base64.b64encode(chunk).decode()}
        yield f"data: {json.dumps(event)}\n\n".encode()
    yield f"data: {json.dumps({'type': 'speech.audio.done'})}\n\n".encode()


async def once(data: bytes) -> AsyncIterator[bytes]:
    yield data


HEADERS = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}


async def wait_end(generation_id: str, db):
    """The finished generation row (completed/failed) — the fallback when not streamed."""
    from ..database import Generation as DBGeneration

    while True:
        db.expire_all()
        row = db.query(DBGeneration).filter_by(id=generation_id).first()
        if row is None:
            raise LookupError(generation_id)
        if row.status in ("completed", "failed"):
            return row
        await asyncio.sleep(0.25)
