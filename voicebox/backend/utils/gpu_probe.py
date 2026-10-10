"""Co widac o karcie BEZ ladowania torcha.

Zdrowie i panel GPU sa odpytywane co kilka sekund. Gdyby siegaly po
``torch.cuda``, pierwszy poll wciagnalby caly stos PyTorch do procesu — czyli
dokladnie to, czego ma nie byc, dopoki nie zalado wany jest zaden model.

Wiec: **jesli torch juz siedzi w procesie, liczymy nim** (pelna wiernosc,
CUDA/MPS/XPU tez); jesli nie — bierzemy to, co kernel wystawia w sysfs i
`rocminfo`, i to wystarcza na nazwe karty oraz zajetosc VRAM.

⚠️ Roznica w znaczeniu liczby VRAM: ``torch.cuda.memory_allocated()`` mowi
tylko o alokacjach torcha, a ``mem_info_vram_used`` o CALEJ karcie (pulpit,
przegladarka, ggml, wszystko). Dlatego przy nieobecnym torchu wracamy z
liczba sysfs i oznaczamy ja jako calosc karty.
"""

import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

KATALOG_DRM = Path("/sys/class/drm")


def torch_zaladowany() -> bool:
    """Czy torch siedzi juz w procesie. Nie importuje go."""
    return "torch" in sys.modules


@lru_cache(maxsize=1)
def _karta_amdgpu() -> Optional[Path]:
    """Katalog `device` pierwszej karty na sterowniku amdgpu."""
    try:
        uevents = sorted(KATALOG_DRM.glob("card*/device/uevent"))
    except OSError:
        return None
    for uevent in uevents:
        try:
            if "DRIVER=amdgpu" in uevent.read_text():
                return uevent.parent
        except OSError:
            continue
    return None


def vram_sysfs_mb() -> Optional[Tuple[float, float]]:
    """(uzyte, calkowite) w MB dla calej karty. None, gdy kernel nie wystawia."""
    karta = _karta_amdgpu()
    if karta is None:
        return None
    try:
        uzyte = int((karta / "mem_info_vram_used").read_text().strip())
        calosc = int((karta / "mem_info_vram_total").read_text().strip())
    except (OSError, ValueError):
        return None
    return uzyte / 1024 / 1024, calosc / 1024 / 1024


@lru_cache(maxsize=1)
def nazwa_gpu() -> Optional[str]:
    """Nazwa handlowa karty z `rocminfo` (~20 ms, liczona raz na proces).

    rocminfo wypisuje agenta CPU przed GPU i oba maja `Marketing Name`, wiec
    nazwe bierzemy dopiero po linii `Device Type: GPU`.

    ⚠️ `*_VISIBLE_DEVICES` ze srodowiska Voiceboxa sa tu ZDEJMOWANE. Pytamy o
    zelazo, nie o to, co dany silnik ma widziec — a przy jednej karcie
    `ROCR_VISIBLE_DEVICES=1` chowa ja calkowicie i panel zostalby bez nazwy.
    """
    srodowisko = {
        k: v
        for k, v in os.environ.items()
        if not k.endswith("_VISIBLE_DEVICES")
    }
    try:
        wynik = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=5,
            env=srodowisko,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if wynik.returncode != 0:
        return None

    nazwa = None
    for linia in wynik.stdout.splitlines():
        trafienie = re.search(r"Marketing Name:\s*(.+)", linia)
        if trafienie:
            nazwa = trafienie.group(1).strip()
        elif re.search(r"Device\s+Type:\s*GPU", linia) and nazwa:
            return nazwa
    return None


def etykieta_bez_torcha() -> Optional[str]:
    """Opis karty na panel, gdy torcha nie ma w procesie."""
    nazwa = nazwa_gpu()
    if nazwa:
        return f"ROCm ({nazwa})"
    if _karta_amdgpu() is not None:
        return "AMD GPU"
    return None
