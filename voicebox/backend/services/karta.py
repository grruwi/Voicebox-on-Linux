"""Wybor KARTY GRAFICZNEJ dla modeli — jedno miejsce, ktore o tym wie.

Po co: w kombajnie stoja dwie karty (9070 XT + 7900 XTX). ggml i PyTorch
domyslnie widza OBIE i rozkladaja model po obu — zmierzone 02.09: jeden
`llama-tts-server` trzymal 5728 MiB na jednej i 2545 MiB na drugiej
ROWNOCZESNIE, czyli polowki modelu na dwoch roznych architekturach, z ruchem
przez PCIe miedzy nimi.

⛔ WYBOR JEST TWARDY, NIE "PREFERENCJA". Niewybrana karta ma byc NIEWIDZIALNA,
zeby nic nie mialo szansy sie na nia rozlac. Stad dwie zmienne, nie jedna:

    ROCR_VISIBLE_DEVICES    HIP/ROCm  → PyTorch (STT, silniki PT) oraz ggml-hip
    GGML_VK_VISIBLE_DEVICES ggml/Vulkan → llama-tts-server, fish s2

⚠️ Te dwie warstwy NIE WIDZA SIEBIE NAWZAJEM. Vulkan nie patrzy na ROCR i
odwrotnie — ustawienie jednej zostawia druga polowe stacku rozlana, i to CICHO.

⚠️ Zmienne czyta sie przy STARCIE PROCESU. Backend nie przestawi ich sobie w
locie, dlatego zapis idzie do pliku, ktory czyta launcher, a zmiana wchodzi po
restarcie Voiceboxa. Tak to zamowil grruwi ("a ja se bede restartowal").

⭐ INDEKS SAM W SOBIE NIE PRZEZYWA PRZEPROWADZKI. Numeruje pozycje na liscie
kart TEJ maszyny, wiec wyjecie karty albo przeniesienie stacku na inne pudlo
przesuwa go pod stopami. Zmierzone 15.09: `GGML_VK_VISIBLE_DEVICES=1` przy
jednej karcie daje `Invalid device index 1` → zero urzadzen → ggml liczy na
CPU i przewraca sie na `GGML_ASSERT(src0->type == GGML_TYPE_F16)`.

Stad wybor jest zapisywany ADRESEM PCI, a indeks liczony przy KAZDYM starcie
(`rozwiaz()`). Kolejnosc awaryjna, gdy zapisanej karty nie ma w maszynie:
jedna karta w srodku → ona; wiecej niz jedna → zadnego ograniczenia i glosna
linia w logu, zeby wybral w panelu. Nigdy pusto — lepiej model na zlej karcie
niz cichy zjazd na procesor.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

# Plik CELOWO osobny od `ustawienia.env`: tamten jest reczny, z komentarzami
# grruwiego, i backend nie ma prawa go przepisywac. Ten nalezy do GUI.
PLIK = Path(__file__).resolve().parents[2] / "karta.env"

_DRM = Path("/sys/class/drm")


def _nazwy_z_lspci() -> dict[str, str]:
    """{'0000:03:00.0': 'Radeon RX 9070 XT'} — jeden strzal, nie na karte."""
    try:
        out = subprocess.run(
            ["lspci", "-D", "-nn"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    nazwy: dict[str, str] = {}
    for linia in out.splitlines():
        if not re.search(r"VGA compatible|Display controller", linia):
            continue
        adres = linia.split()[0]
        # "... [AMD/ATI] Navi 48 [Radeon RX 9070/9070 XT/9070 GRE] [1002:7550]"
        marki = re.findall(r"\[([^\[\]]+)\]", linia)
        ladna = next((m for m in marki if "Radeon" in m or "GeForce" in m), None)
        nazwy[adres] = ladna or linia.split(":", 2)[-1].strip()
    return nazwy


def karty() -> list[dict]:
    """Karty amdgpu w kolejnosci PCI — TEJ SAMEJ, ktorej uzywaja ROCm i Vulkan.

    ⚠️ Indeks w tej liscie to indeks dla obu zmiennych widocznosci. Kolejnosc
    bierze sie z sortowania po adresie PCI, bo tak enumeruja oba backendy —
    NIE z numeru `cardN` w jadrze, ktory bywa odwrotny (zmierzone 02.09:
    card0 = 07:00.0, card1 = 03:00.0).
    """
    nazwy = _nazwy_z_lspci()
    znalezione = []
    for kat in sorted(_DRM.glob("card[0-9]*")):
        dev = kat / "device"
        uevent = dev / "uevent"
        if not uevent.is_file():
            continue
        tresc = uevent.read_text(errors="ignore")
        if "DRIVER=amdgpu" not in tresc:
            continue
        m = re.search(r"PCI_SLOT_NAME=(\S+)", tresc)
        if not m:
            continue
        pci = m.group(1)
        try:
            vram = int((dev / "mem_info_vram_total").read_text()) // (1024 * 1024)
        except (OSError, ValueError):
            vram = 0
        znalezione.append({"pci": pci, "nazwa": nazwy.get(pci, "GPU"), "vram_mib": vram})

    znalezione.sort(key=lambda k: k["pci"])
    for i, k in enumerate(znalezione):
        k["indeks"] = i
    return znalezione


def _zapisane() -> Optional[int]:
    if not PLIK.is_file():
        return None
    m = re.search(r"^VOICEBOX_KARTA=(\d+)\s*$", PLIK.read_text(), re.M)
    return int(m.group(1)) if m else None


def _zapisany_pci() -> Optional[str]:
    """Adres PCI wybranej karty — jedyna czesc zapisu, ktora przezywa zmiane maszyny."""
    if not PLIK.is_file():
        return None
    m = re.search(r"^VOICEBOX_KARTA_PCI=(\S+)\s*$", PLIK.read_text(), re.M)
    return m.group(1) if m else None


def rozwiaz() -> dict:
    """Ktora karte pokazac silnikom TERAZ, na TEJ maszynie.

    Zwraca `{"indeks": int|None, "karta": dict|None, "powod": str}`.
    `indeks is None` znaczy: zadnego ograniczenia, silnik bierze swoje.
    """
    dostepne = karty()
    if not dostepne:
        return {
            "indeks": None,
            "karta": None,
            "powod": "nie widze zadnej karty amdgpu — bez ograniczenia",
        }

    pci = _zapisany_pci()
    if pci:
        trafiona = next((k for k in dostepne if k["pci"] == pci), None)
        if trafiona:
            return {
                "indeks": trafiona["indeks"],
                "karta": trafiona,
                "powod": f"wybor z panelu: {trafiona['nazwa']} ({pci})",
            }

    # Plik sprzed wprowadzenia adresu PCI — indeks jest wszystkim, co mamy.
    zapis = _zapisane()
    if pci is None and zapis is not None:
        trafiona = next((k for k in dostepne if k["indeks"] == zapis), None)
        if trafiona:
            return {
                "indeks": zapis,
                "karta": trafiona,
                "powod": f"zapisany indeks {zapis}: {trafiona['nazwa']}",
            }

    if len(dostepne) == 1:
        k = dostepne[0]
        powod = f"jedyna karta w maszynie: {k['nazwa']} ({k['pci']})"
        if pci or zapis is not None:
            powod = f"zapisanej karty tu nie ma — biore {k['nazwa']} ({k['pci']})"
        return {"indeks": k["indeks"], "karta": k, "powod": powod}

    return {
        "indeks": None,
        "karta": None,
        "powod": (
            f"{len(dostepne)} karty, zapisanej nie ma — bez ograniczenia. "
            "Wybierz w panelu, inaczej model rozlozy sie po obu"
        ),
    }


def _aktywna() -> Optional[int]:
    """Co ten proces FAKTYCZNIE widzi — nie to, co stoi w pliku.

    Rozjazd miedzy tym a `_zapisane()` znaczy dokladnie jedno: ktos zmienil
    wybor i nie zrestartowal jeszcze Voiceboxa. GUI ma to pokazac, zeby nikt
    nie szukal potem, czemu "ustawienie nie dziala".
    """
    w = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
    return int(w) if w.isdigit() else None


def stan() -> dict:
    zapis, akt = _zapisane(), _aktywna()
    wybor = rozwiaz()
    return {
        "karty": karty(),
        "wybrana": zapis,
        "aktywna": akt,
        "wymaga_restartu": wybor["indeks"] != akt,
        "rozwiazana": wybor["indeks"],
        "powod": wybor["powod"],
    }


def ustaw(indeks: Optional[int]) -> dict:
    """`None` zdejmuje ograniczenie — obie karty znowu widoczne."""
    dostepne = karty()
    if indeks is not None and not any(k["indeks"] == indeks for k in dostepne):
        raise ValueError(f"nie ma karty o indeksie {indeks}")

    if indeks is None:
        # ⛔ Pusty plik, NIE wpisana wartosc domyslna: zadna zmienna = silnik
        # bierze swoje, i widac na pierwszy rzut oka, ze nikt tu nic nie wybral.
        PLIK.write_text(
            "# Wybor karty z GUI Voiceboxa. Pusto = bez ograniczenia, obie widoczne.\n"
        )
    else:
        k = next(x for x in dostepne if x["indeks"] == indeks)
        PLIK.write_text(
            "# ⛔ PLIK GENEROWANY PRZEZ GUI VOICEBOXA — recznie nie edytowac,\n"
            "# nastepny klik w panelu i tak go nadpisze. Reczne pokretla:\n"
            "# `ustawienia.env` obok.\n"
            f"# Wybrana karta: {k['nazwa']} ({k['pci']}, {k['vram_mib']} MiB)\n"
            "#\n"
            "# ⭐ ADRES PCI jest wiazacy — indeks ponizej to tylko podglad na TE\n"
            "# maszyne. Launcher przelicza go przy kazdym starcie, wiec wyjecie\n"
            "# karty ani przeprowadzka na inne pudlo nie zostawia martwego numeru.\n"
            f"VOICEBOX_KARTA_PCI={k['pci']}\n"
            f"VOICEBOX_KARTA={indeks}\n"
        )
    return stan()


# --------------------------------------------------------------------------
# Wejscie dla launchera
# --------------------------------------------------------------------------


def _eksport() -> str:
    """Linie dla `eval` w bashu — wybor liczony na zywo, nie z pliku."""
    wybor = rozwiaz()
    if wybor["indeks"] is None:
        return "unset ROCR_VISIBLE_DEVICES GGML_VK_VISIBLE_DEVICES"
    i = wybor["indeks"]
    # Obie zmienne sa KONIECZNE — Vulkan nie patrzy na ROCR i odwrotnie.
    return f"export ROCR_VISIBLE_DEVICES={i}\nexport GGML_VK_VISIBLE_DEVICES={i}"


if __name__ == "__main__":
    import json
    import sys

    if "--eksport" in sys.argv:
        print(_eksport())
        print(f"karta: {rozwiaz()['powod']}", file=sys.stderr)
    else:
        print(json.dumps(stan(), indent=2, ensure_ascii=False))
