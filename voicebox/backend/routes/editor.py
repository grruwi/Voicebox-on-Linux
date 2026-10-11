"""POST /editor/filters — the sample editor's filters (11.10, stage 2): sibilants, clicks, top end, silence.

The frontend sends the current version of the sample (WAV) + the settings; it gets the
same sound back through ffmpeg. Length and sample rate stay the same, so the selection
on the waveform still fits. Nothing is saved — a non-destructive layer, the frontend
does the swap.

Filter choice MEASURED on a whispering Vox sample (`08-szept-sekret`):
- `deesser` (i=0.8): sibilants −12 dB, the rest of the top −4 dB, the mids −1.7 dB — selective.
- `agate`: silence −6 dB, speech untouched.
- `adeclick`: on a sample without clicks it moves nothing (−0.4 dB).
- REJECTED: `afftdn` (silence +25 dB — adds noise to clean TTS), `anlmdn` (cuts the
  ending, the host's ffmpeg crashed), `adynamicequalizer` (auto=adaptive kills the whole
  top end −50 dB, and the threshold cannot be scaled sensibly).
"""

import asyncio
import json
import math
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

router = APIRouter()


def _number(d: dict, key: str, low: float, high: float, default: float) -> float:
    try:
        v = float(d.get(key, default))
    except (TypeError, ValueError):
        v = default
    if not math.isfinite(v):
        v = default
    return min(high, max(low, v))


def chain(filters: dict) -> str:
    """Settings from the frontend → an ffmpeg chain. Numbers only, clamped to their ranges."""
    parts: list[str] = []
    dc = filters.get("declick") or {}
    if dc.get("on"):
        # sensitivity 0..1 → adeclick threshold 8..1 (a lower threshold catches more clicks)
        sensitivity = _number(dc, "sensitivity", 0, 1, 0.5)
        parts.append(f"adeclick=t={8 - 7 * sensitivity:.2f}")
    de = filters.get("deesser") or {}
    if de.get("on"):
        strength = _number(de, "strength", 0, 1, 0.7)
        frequency = _number(de, "frequency", 0, 1, 0.5)
        parts.append(f"deesser=i={strength:.3f}:m=0.6:f={frequency:.3f}")
    tr = filters.get("treble") or {}
    if tr.get("on"):
        hz = _number(tr, "hz", 3000, 16000, 8000)
        db = _number(tr, "db", -24, 0, -6)
        parts.append(f"treble=g={db:.1f}:f={hz:.0f}")
    gate = filters.get("gate") or {}
    if gate.get("on"):
        threshold = _number(gate, "threshold_db", -80, -10, -45)
        reduction = _number(gate, "reduction_db", -60, 0, -20)
        parts.append(
            f"agate=threshold={10 ** (threshold / 20):.6f}:range={10 ** (reduction / 20):.6f}"
            ":ratio=4:attack=5:release=150"
        )
    return ",".join(parts)


@router.post("/editor/filters")
async def filters(file: UploadFile = File(...), settings: str = Form("{}")):
    try:
        ffmpeg_chain = chain(json.loads(settings))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"settings are not JSON: {e}")
    data = await file.read()
    if not ffmpeg_chain:
        return Response(content=data, media_type="audio/wav")
    with tempfile.TemporaryDirectory() as directory:
        src, dst = Path(directory) / "in.wav", Path(directory) / "out.wav"
        src.write_bytes(data)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(src), "-af", ffmpeg_chain,
            "-c:a", "pcm_f32le", str(dst),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0 or not dst.exists():
            raise HTTPException(status_code=500, detail=f"ffmpeg ({ffmpeg_chain}): {err.decode(errors='replace')[-500:]}")
        return Response(content=dst.read_bytes(), media_type="audio/wav")
