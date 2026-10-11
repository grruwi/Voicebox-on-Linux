"""Builds the prebuilt VoxCPM2 (PyTorch) weight sets — once, on the CPU, from `weights/` (BF16).

The worker loads ONLY these sets — run this once before the first load. Converting the
BF16 original in place on every load would be slow and spike RAM (both copies in memory).

    docker compose -f docker-compose.yml -f docker-compose.pytorch.yml run --rm \
        -w /app/workers/voxcpm2 pytorch python make_weights.py [base_lm-acoustic ...]

No arguments = all four: q8-f16, q8-f32, f16-f16, f16-f32.
Output: `$VOXCPM_PREBUILT/<set>/model.pt` — the `tts_model` weights in their target dtypes
(AudioVAE f32, BaseLM q8 as `Int8Linear`) + rope buffers. No KV cache.
An existing set is skipped — delete its directory to rebuild it.
"""

import os
import sys
import time
from pathlib import Path

from precision import ACOUSTIC, BASE_LM, convert, report, save, set_name

SOURCE = os.environ.get("VOXCPM_WEIGHTS", "/weights")
TARGET = Path(os.environ.get("VOXCPM_PREBUILT", "/prebuilt"))


def build(base_lm: str, acoustic: str) -> None:
    from voxcpm import VoxCPM

    directory = TARGET / set_name(base_lm, acoustic)
    path = directory / "model.pt"
    if path.exists():
        print(f"[set] {directory.name}: exists, skipping", flush=True)
        return
    t0 = time.time()
    m = VoxCPM.from_pretrained(SOURCE, optimize=False, load_denoiser=False, device="cpu")
    tm = m.tts_model
    convert(tm, base_lm, acoustic)
    print(report(tm), flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".pt.partial")
    save(tm, partial)
    partial.rename(path)  # never half a file under the final name
    print(f"[set] {directory.name}: {path.stat().st_size / 2**30:.2f} GB "
          f"in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    sets = sys.argv[1:] or [set_name(b, a) for b in BASE_LM for a in ACOUSTIC]
    for s in sets:
        b, a = s.split("-")
        assert b in BASE_LM and a in ACOUSTIC, f"unknown set: {s}"
        build(b, a)
