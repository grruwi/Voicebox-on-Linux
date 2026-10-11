"""What can be seen about the card WITHOUT loading torch.

Health and the GPU panel are polled every few seconds. If they reached for
``torch.cuda``, the first poll would pull the whole PyTorch stack into the process —
exactly what must not happen: Voicebox runs no model of its own.

So: **if torch already sits in the process, we use it** (full fidelity, CUDA/MPS/XPU
too); if not — we take what the kernel exposes in sysfs and `rocminfo`, which is
enough for the card name and VRAM usage.

⚠️ The VRAM number means something different: ``torch.cuda.memory_allocated()`` only
covers torch's allocations, while ``mem_info_vram_used`` covers the WHOLE card (desktop,
browser, ggml, everything). So without torch we return the sysfs number and mark it as
the whole card.
"""

import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

DRM_DIR = Path("/sys/class/drm")


def torch_loaded() -> bool:
    """Whether torch already sits in the process. Does not import it."""
    return "torch" in sys.modules


@lru_cache(maxsize=1)
def _amdgpu_card() -> Optional[Path]:
    """The `device` directory of the first card on the amdgpu driver."""
    try:
        uevents = sorted(DRM_DIR.glob("card*/device/uevent"))
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
    """(used, total) in MB for the whole card. None when the kernel does not expose it."""
    card = _amdgpu_card()
    if card is None:
        return None
    try:
        used = int((card / "mem_info_vram_used").read_text().strip())
        total = int((card / "mem_info_vram_total").read_text().strip())
    except (OSError, ValueError):
        return None
    return used / 1024 / 1024, total / 1024 / 1024


@lru_cache(maxsize=1)
def gpu_name() -> Optional[str]:
    """The card's marketing name from `rocminfo` (~20 ms, computed once per process).

    rocminfo prints the CPU agent before the GPU and both have a `Marketing Name`, so
    the name is taken only after the `Device Type: GPU` line.

    ⚠️ `*_VISIBLE_DEVICES` from Voicebox's environment are REMOVED here. We ask about the
    hardware, not about what an engine should see — with one card
    `ROCR_VISIBLE_DEVICES=1` hides it completely and the panel would have no name.
    """
    env = {k: v for k, v in os.environ.items() if not k.endswith("_VISIBLE_DEVICES")}
    try:
        result = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=5, env=env)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None

    name = None
    for line in result.stdout.splitlines():
        hit = re.search(r"Marketing Name:\s*(.+)", line)
        if hit:
            name = hit.group(1).strip()
        elif re.search(r"Device\s+Type:\s*GPU", line) and name:
            return name
    return None


def label_without_torch() -> Optional[str]:
    """The card description for the panel when torch is not in the process."""
    name = gpu_name()
    if name:
        return f"ROCm ({name})"
    if _amdgpu_card() is not None:
        return "AMD GPU"
    return None
