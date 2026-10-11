"""The GPU choice for the models — the one place that knows about it.

Why: a machine can hold two cards (here a 9070 XT + a 7900 XTX). ggml and PyTorch see
BOTH by default and spread a model over both — measured 02.09: one `llama-tts-server`
held 5728 MiB on one and 2545 MiB on the other AT THE SAME TIME, halves of the model on
two different architectures, with traffic over PCIe between them.

⛔ THE CHOICE IS HARD, NOT A "PREFERENCE". The other card has to be INVISIBLE, so nothing
gets a chance to spill onto it. Hence two variables, not one:

    ROCR_VISIBLE_DEVICES    HIP/ROCm    → PyTorch engines and ggml-hip
    GGML_VK_VISIBLE_DEVICES ggml/Vulkan → audio.cpp (Vulkan) and whisper.cpp

⚠️ These two layers DO NOT SEE EACH OTHER. Vulkan ignores ROCR and the other way round —
setting one leaves the other half of the stack spread, and SILENTLY.

⭐ The engine hosts apply it: Voicebox passes the resolved index with every `/load`, the
container's manager sets both variables for the model's process, and a different card =
a new process. A change in the panel takes effect with the next load.

⭐ AN INDEX ALONE DOES NOT SURVIVE A MOVE. It numbers positions on THIS machine's card
list, so taking a card out or moving the stack to another box shifts it underfoot.
Measured 15.09: `GGML_VK_VISIBLE_DEVICES=1` with one card gives `Invalid device index 1`
→ zero devices → ggml computes on the CPU and falls over on
`GGML_ASSERT(src0->type == GGML_TYPE_F16)`.

So the choice is stored as a PCI ADDRESS, and the index is computed on EVERY load
(`resolve()`). The fallback order when the stored card is not in the machine: one card
inside → that one; more than one → no restriction and a loud line in the log, so it gets
picked in the panel. Never empty — a model on the wrong card is better than a silent
slide onto the CPU.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

_DRM = Path("/sys/class/drm")


def _choice_file() -> Path:
    """Written by the GUI, in the data directory (it survives container re-creation)."""
    from ..config import get_data_dir

    return get_data_dir() / "gpu.env"


def _names_from_lspci() -> dict[str, str]:
    """{'0000:03:00.0': 'Radeon RX 9070 XT'} — one call, not one per card."""
    try:
        out = subprocess.run(
            ["lspci", "-D", "-nn"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    names: dict[str, str] = {}
    for line in out.splitlines():
        if not re.search(r"VGA compatible|Display controller", line):
            continue
        address = line.split()[0]
        # "... [AMD/ATI] Navi 48 [Radeon RX 9070/9070 XT/9070 GRE] [1002:7550]"
        brackets = re.findall(r"\[([^\[\]]+)\]", line)
        pretty = next((b for b in brackets if "Radeon" in b or "GeForce" in b), None)
        names[address] = pretty or line.split(":", 2)[-1].strip()
    return names


def cards() -> list[dict]:
    """amdgpu cards in PCI order — THE SAME order ROCm and Vulkan use.

    ⚠️ The index in this list is the index for both visibility variables. The order comes
    from sorting by PCI address, because that is how both backends enumerate — NOT from
    the kernel's `cardN` number, which can be reversed (measured 02.09:
    card0 = 07:00.0, card1 = 03:00.0).
    """
    names = _names_from_lspci()
    found = []
    for node in sorted(_DRM.glob("card[0-9]*")):
        dev = node / "device"
        uevent = dev / "uevent"
        if not uevent.is_file():
            continue
        content = uevent.read_text(errors="ignore")
        if "DRIVER=amdgpu" not in content:
            continue
        m = re.search(r"PCI_SLOT_NAME=(\S+)", content)
        if not m:
            continue
        pci = m.group(1)
        try:
            vram = int((dev / "mem_info_vram_total").read_text()) // (1024 * 1024)
        except (OSError, ValueError):
            vram = 0
        found.append({"pci": pci, "name": names.get(pci, "GPU"), "vram_mib": vram})

    found.sort(key=lambda c: c["pci"])
    for i, c in enumerate(found):
        c["index"] = i
    return found


def _stored_pci() -> Optional[str]:
    """The chosen card's PCI address — the only part of the choice that survives a machine change."""
    path = _choice_file()
    if not path.is_file():
        return None
    m = re.search(r"^VOICEBOX_GPU_PCI=(\S+)\s*$", path.read_text(), re.M)
    return m.group(1) if m else None


def resolve() -> dict:
    """Which card to show the engines NOW, on THIS machine.

    Returns `{"index": int|None, "card": dict|None, "reason": str}`.
    `index is None` means: no restriction, the engine takes its own.
    """
    available = cards()
    if not available:
        return {"index": None, "card": None, "reason": "no amdgpu card visible — no restriction"}

    pci = _stored_pci()
    if pci:
        hit = next((c for c in available if c["pci"] == pci), None)
        if hit:
            return {"index": hit["index"], "card": hit, "reason": f"chosen in the panel: {hit['name']} ({pci})"}

    if len(available) == 1:
        c = available[0]
        reason = f"the only card in the machine: {c['name']} ({c['pci']})"
        if pci:
            reason = f"the stored card is not here — taking {c['name']} ({c['pci']})"
        return {"index": c["index"], "card": c, "reason": reason}

    return {
        "index": None,
        "card": None,
        "reason": (
            f"{len(available)} cards, none stored — no restriction. "
            "Pick one in the panel, otherwise a model spreads over both"
        ),
    }


def state() -> dict:
    choice = resolve()
    return {
        "cards": cards(),
        "selected_pci": _stored_pci(),
        "resolved": choice["index"],
        "reason": choice["reason"],
    }


def select(index: Optional[int]) -> dict:
    """`None` lifts the restriction — every card visible again."""
    available = cards()
    if index is not None and not any(c["index"] == index for c in available):
        raise ValueError(f"no card with index {index}")

    path = _choice_file()
    if index is None:
        # ⛔ An empty file, NOT a written default: no variable = the engine takes its own,
        # and it is obvious at first sight that nobody picked anything here.
        path.write_text("# GPU choice from Voicebox's GUI. Empty = no restriction, every card visible.\n")
    else:
        c = next(x for x in available if x["index"] == index)
        path.write_text(
            "# ⛔ WRITTEN BY VOICEBOX'S GUI — do not edit by hand, the next click overwrites it.\n"
            f"# Chosen card: {c['name']} ({c['pci']}, {c['vram_mib']} MiB)\n"
            "# ⭐ The PCI ADDRESS is binding — the index is recomputed on every load.\n"
            f"VOICEBOX_GPU_PCI={c['pci']}\n"
        )
    return state()


def engine_env() -> dict:
    """The visibility variables for an engine process — `{}` = no restriction.
    Both are NEEDED — Vulkan ignores ROCR and the other way round."""
    index = resolve()["index"]
    if index is None:
        return {}
    return {"ROCR_VISIBLE_DEVICES": str(index), "GGML_VK_VISIBLE_DEVICES": str(index)}
