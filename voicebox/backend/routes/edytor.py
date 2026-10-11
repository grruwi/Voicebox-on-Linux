"""POST /edytor/filtry — filtry edytora probki (11.10, etap 2): esy, cykanie, gora, cisza.

Front wysyla biezaca wersje probki (WAV) + ustawienia; dostaje ten sam dzwiek po
ffmpegu. Dlugosc i czestotliwosc zostaja bez zmian, wiec zaznaczenie na fali dalej
pasuje. Nic nie zapisujemy — warstwa niedestrukcyjna, podmiane robi front.

Dobor filtrow ZMIERZONY na szepczacej probce voxa (`08-szept-sekret`):
- `deesser` (i=0.8): esy −12 dB, reszta gory −4 dB, srodek −1,7 dB — wybiorczy.
- `agate`: cisza −6 dB, mowa nietknieta.
- `adeclick`: na probce bez klikow nic nie rusza (−0,4 dB).
- ODRZUCONE: `afftdn` (cisza +25 dB — dosypuje szum na czystym TTS), `anlmdn` (ucina
  koncowke, ffmpeg z hosta sie wywalil), `adynamicequalizer` (auto=adaptive zabija
  cala gore −50 dB, a progu nie da sie sensownie wyskalowac).
"""

import asyncio
import json
import math
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

router = APIRouter()


def _liczba(d: dict, klucz: str, od: float, do: float, domyslna: float) -> float:
    try:
        v = float(d.get(klucz, domyslna))
    except (TypeError, ValueError):
        v = domyslna
    if not math.isfinite(v):
        v = domyslna
    return min(do, max(od, v))


def lancuch(filtry: dict) -> str:
    """Ustawienia z frontu → lancuch ffmpeg. Tylko liczby, przyciete do zakresow."""
    czesci: list[str] = []
    dc = filtry.get("declick") or {}
    if dc.get("on"):
        # czulosc 0..1 → prog adeclick 8..1 (nizszy prog = lapie wiecej klikow)
        czul = _liczba(dc, "czulosc", 0, 1, 0.5)
        czesci.append(f"adeclick=t={8 - 7 * czul:.2f}")
    de = filtry.get("deesser") or {}
    if de.get("on"):
        sila = _liczba(de, "sila", 0, 1, 0.7)
        wys = _liczba(de, "wysokosc", 0, 1, 0.5)
        czesci.append(f"deesser=i={sila:.3f}:m=0.6:f={wys:.3f}")
    go = filtry.get("gora") or {}
    if go.get("on"):
        hz = _liczba(go, "hz", 3000, 16000, 8000)
        db = _liczba(go, "db", -24, 0, -6)
        czesci.append(f"treble=g={db:.1f}:f={hz:.0f}")
    br = filtry.get("bramka") or {}
    if br.get("on"):
        prog = _liczba(br, "prog_db", -80, -10, -45)
        scisz = _liczba(br, "sciszenie_db", -60, 0, -20)
        czesci.append(
            f"agate=threshold={10 ** (prog / 20):.6f}:range={10 ** (scisz / 20):.6f}"
            ":ratio=4:attack=5:release=150"
        )
    return ",".join(czesci)


@router.post("/edytor/filtry")
async def filtry(file: UploadFile = File(...), ustawienia: str = Form("{}")):
    try:
        lan = lancuch(json.loads(ustawienia))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"ustawienia to nie JSON: {e}")
    dane = await file.read()
    if not lan:
        return Response(content=dane, media_type="audio/wav")
    with tempfile.TemporaryDirectory() as kat:
        we, wy = Path(kat) / "we.wav", Path(kat) / "wy.wav"
        we.write_bytes(dane)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(we), "-af", lan,
            "-c:a", "pcm_f32le", str(wy),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0 or not wy.exists():
            raise HTTPException(status_code=500, detail=f"ffmpeg ({lan}): {err.decode(errors='replace')[-500:]}")
        return Response(content=wy.read_bytes(), media_type="audio/wav")
