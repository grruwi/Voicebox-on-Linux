"""Prebuilt VoxCPM2 weights for audio.cpp — one file per precision set, so the engine
converts NOTHING when it loads.

Without them the engine takes the BF16 original and rewrites the whole generator into the
panel's precision on every load: most of the load time, and a RAM spike while both copies
sit in memory. With them a load is ~2 s.

    docker compose run --rm audiocpp python /app/make_weights.py q8_0 f16

    base_lm  = q8_0 | f16 | f32   the BaseLM (`voxcpm2.base_lm_weight_type`)
    acoustic = f16 | f32          the rest of the generator: ResidualLM, LocEnc, DiT,
                                  projections (`voxcpm2.weight_type`)

In:  /models/audiocpp/weights/voxcpm2-orig.gguf
Out: /models/audiocpp/prebuilt/<base_lm>-<acoustic>/voxcpm2.gguf — Voicebox picks it up
     by itself when the panel asks for that set.

The types in the file are EXACTLY what the engine would make on the fly: generator biases
and norms → F32, BaseLM matrices → <base_lm>, the rest of the generator → <acoustic>,
AudioVAE untouched (F32). Same conversion code (ggml), so the weights are bit for bit
the same.
"""

import os
import subprocess
import sys
from pathlib import Path

import gguf

TOOL = "/opt/audiocpp/bin/audiocpp_gguf"
MODELS = Path(os.environ.get("VOICEBOX_VOICE_MODELS") or "/models")
SOURCE = Path(os.environ.get("AUDIOCPP_WEIGHTS") or MODELS / "audiocpp" / "weights") / "voxcpm2-orig.gguf"
TARGET = Path(os.environ.get("AUDIOCPP_PREBUILT") or MODELS / "audiocpp" / "prebuilt")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in ("q8_0", "f16", "f32") or sys.argv[2] not in ("f16", "f32"):
        sys.exit(__doc__)
    base_lm, acoustic = sys.argv[1:]

    # generator biases and norms (1-D tensors under `weights/`, except special_token) → f32
    reader = gguf.GGUFReader(str(SOURCE))
    names = reader.fields["audiocpp.tensor_names"].contents()
    keep = []
    for tensor, name in zip(reader.tensors, names):
        if name.startswith("weights/") and sum(1 for x in tensor.shape if x > 1) < 2 \
                and name != "weights/feat_encoder.special_token":
            keep += ["--keep-type", f"{name}=f32"]

    out = TARGET / f"{base_lm}-{acoustic}"
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "voxcpm2.gguf.tmp"
    subprocess.run([
        TOOL, "--input", str(SOURCE), "--output", str(partial),
        "--type", "orig", "--family", "voxcpm2", *keep,
        "--keep-type", f"weights/base_lm*={base_lm}",
        "--keep-type", "audiovae_weights*=native",   # first matching rule wins
        "--keep-type", f"weights/*={acoustic}",
        "--overwrite",
    ], check=True)
    partial.rename(out / "voxcpm2.gguf")   # never half a file under the final name
    print(f"{out / 'voxcpm2.gguf'}: {(out / 'voxcpm2.gguf').stat().st_size / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
