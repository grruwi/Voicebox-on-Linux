"""The speech stream to the BROWSER (11.10, grruwi's call).

⭐ "Autoplay on generate" CHECKED = a generation from the GUI does not go to `ffplay`, but
here: `run_generation` (`_play_streaming`) puts the engine's chunks in as they are made,
and `GET /generate/{id}/stream` hands them to the browser, which plays them right away
(Web Audio). Unchecked = the old road, `ffplay` on the server's speakers — nothing lands here.

Stream format: 4 bytes of sample rate (uint32 LE), then raw f32le mono.
The sample rate is in the body, not an HTTP header — a header without `expose_headers`
is invisible to a `fetch` from another origin (the panel from dev, Tauri).

Chunks stay in memory until the generation ends + `_LIFE_AFTER_END_S`, so a browser that
connects later gets everything from the start. The "not streamed" decision is recorded
too (`refuse`), so the route answers 204 right away and the panel plays the file.

All on one event loop (the generation and the route are coroutines) — no locks.
"""

import asyncio
import time
from typing import Optional

_LIFE_AFTER_END_S = 60


class Stream:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.chunks: list[bytes] = []
        self.done = False
        self.failed = False
        self._changed = asyncio.Event()

    def add(self, data: bytes) -> None:
        self.chunks.append(data)
        self._wake()

    def finish(self, failed: bool = False) -> None:
        self.done = True
        self.failed = failed
        self._wake()

    def _wake(self) -> None:
        # The waiters hold the OLD event — set it and put a fresh one in place.
        self._changed.set()
        self._changed = asyncio.Event()

    async def read(self):
        yield self.sample_rate.to_bytes(4, "little")
        i = 0
        while True:
            while i < len(self.chunks):
                yield self.chunks[i]
                i += 1
            if self.done:
                return
            await self._changed.wait()


# generation id -> (time of the last change, Stream or None = "not streamed")
_registry: dict[str, tuple[float, Optional[Stream]]] = {}


def _clean_up() -> None:
    now = time.monotonic()
    for gid, (when, s) in list(_registry.items()):
        if (s is None or s.done) and now - when > _LIFE_AFTER_END_S:
            del _registry[gid]


def open_stream(generation_id: str, sample_rate: int) -> Stream:
    _clean_up()
    s = Stream(sample_rate)
    _registry[generation_id] = (time.monotonic(), s)
    return s


def finish(generation_id: str, failed: bool = False) -> None:
    entry = _registry.get(generation_id)
    if entry and entry[1] is not None:
        entry[1].finish(failed)
        _registry[generation_id] = (time.monotonic(), entry[1])


def refuse(generation_id: str) -> None:
    """This generation does NOT go to the browser — the route answers 204, the panel plays the file."""
    _clean_up()
    _registry[generation_id] = (time.monotonic(), None)


def get(generation_id: str) -> tuple[bool, Optional[Stream]]:
    """(whether the decision was made yet, the stream or None)."""
    entry = _registry.get(generation_id)
    return (entry is not None, entry[1] if entry else None)
