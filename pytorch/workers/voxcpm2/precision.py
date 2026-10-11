"""VoxCPM2 precision sets on PyTorch — shared by `make_weights.py` and `server.py`.

⭐ PREBUILT SETS ON DISK, NO CONVERSION ON THE FLY. `make_weights.py` builds the
whole model once in a given layout and saves it to `prebuilt/<base_lm>-<acoustic>/`.
The server only loads it. Same knobs as the ggml engine (`vox_base_lm`,
`vox_acoustic`):

    base_lm   q8  → int8 weights + per-row scale (`Int8Linear`), computed in BF16
              f16 → FP16
    acoustic  f16 | f32 → the rest of the generator: ResidualLM, LocEnc
              (feat_encoder), DiT (feat_decoder), projections, stop
    AudioVAE  ALWAYS F32 — upstream hard-codes it (`from_local`).

Source: `weights/model.safetensors`, ALL in BF16 — f32 adds no quality that is not
in the file, only computation accuracy.

Voxcpm assumes ONE precision for the whole model (`config.dtype`), so a BaseLM in
another one gets casts at its boundaries (`wire_up`). Casting is not converting
weights — the weights already sit on disk in their target type.
"""

import torch
import torch.nn.functional as F

BASE_LM = ("q8", "f16")
ACOUSTIC = ("f16", "f32")


class Int8Linear(torch.nn.Module):
    """int8 weight + per-row scale; `torch.compile` fuses the dequant into the matmul
    (the gpt-fast recipe). Replaces torchao `Int8WeightOnly`, whose ROCm path is the
    generic `weight_int8pack_mm` kernel: ~44 GB/s, 29 ms per BaseLM step (2026-10-11)."""

    def __init__(self, out_features, in_features, bias, dtype, device=None):
        super().__init__()
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer("scale", torch.empty(out_features, dtype=dtype, device=device))
        self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=dtype, device=device)) if bias else None

    def forward(self, x):
        y = F.linear(x, self.weight.to(x.dtype)) * self.scale
        return y if self.bias is None else y + self.bias


def _int8_linears(module, dtype, quantize: bool) -> None:
    """Swap every `nn.Linear` under `module` for `Int8Linear`. `quantize=True` (only
    `make_weights.py`) fills it from the float weight; False leaves an empty skeleton
    for `load_state_dict(assign=True)`."""
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            w = child.weight
            new = Int8Linear(w.shape[0], w.shape[1], child.bias is not None, dtype, device=w.device)
            if quantize:
                with torch.no_grad():
                    scale = w.float().abs().amax(1).clamp(min=1e-8) / 127
                    new.weight.copy_((w.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8))
                    new.scale.copy_(scale.to(dtype))
                    if child.bias is not None:
                        new.bias.copy_(child.bias.to(dtype))
            setattr(module, name, new)
        else:
            _int8_linears(child, dtype, quantize)


def set_name(base_lm: str, acoustic: str) -> str:
    return f"{base_lm}-{acoustic}"


def _dtypes(base_lm: str, acoustic: str):
    rest = {"f16": torch.float16, "f32": torch.float32}[acoustic]
    # ⛔ q8 computes in BF16, NEVER in FP16: the scale is applied AFTER the matmul,
    # so the sums are ~127× larger than the real ones and overflow in FP16 (11.10:
    # the model never finished its sentences, 580 steps, retries). Same advice as gpt-fast.
    return (torch.bfloat16 if base_lm == "q8" else torch.float16), rest


def convert(tm, base_lm: str, acoustic: str) -> None:
    """ONLY for `make_weights.py`: BF16 from the file → the target layout, on the CPU."""
    base, rest = _dtypes(base_lm, acoustic)
    names = {torch.float16: "float16", torch.float32: "float32"}
    tm.config.dtype = names[rest]
    for name, part in tm.named_children():
        if name not in ("base_lm", "audio_vae"):
            part.to(rest)
    tm.base_lm.to(base)
    if base_lm == "q8":
        _int8_linears(tm.base_lm, base, quantize=True)


def save(tm, path) -> None:
    """Weights + buffers outside `state_dict` (rope: inv_freq/cos/sin — `persistent=False`).
    The KV cache does NOT go to disk — it is a buffer object sized for max_length, not weights."""
    state = tm.state_dict()
    buffers = {n: b for n, b in tm.named_buffers() if n not in state}
    torch.save({"state": state, "buffers": buffers, "dtype": tm.config.dtype}, path)


def load(path, weights_dir: str, device: str = "cuda"):
    """Skeleton on `meta` (no allocation, no random init) + ready tensors from disk
    straight onto the card (`assign=True` — types from the file, no conversion).
    Only the config and the tokenizer come from `weights_dir`."""
    from transformers import LlamaTokenizerFast
    from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig
    from voxcpm.modules.audiovae import AudioVAEV2

    with open(f"{weights_dir}/config.json", encoding="utf-8") as f:
        config = VoxCPMConfig.model_validate_json(f.read())
    tokenizer = LlamaTokenizerFast.from_pretrained(weights_dir)
    data = torch.load(path, map_location=device, weights_only=True)
    # Sets built by the old `voxcpm-pt` container (still in use) keep its key names.
    for old_key, key in (("stan", "state"), ("bufory", "buffers")):
        if old_key in data:
            data[key] = data.pop(old_key)
    with torch.device("meta"):
        audio_vae = AudioVAEV2(config=config.audio_vae_config)
        tm = VoxCPM2Model(config, tokenizer, audio_vae, None, device="cpu")
        # q8 = `Int8Linear` (keys `*.scale`) — the skeleton has to match the file.
        if any(k.startswith("base_lm.") and k.endswith(".scale") for k in data["state"]):
            _int8_linears(tm.base_lm, torch.bfloat16, quantize=False)
    tm.load_state_dict(data["state"], assign=True, strict=True)
    for name, tensor in data["buffers"].items():
        owner, _, attr = name.rpartition(".")
        tm.get_submodule(owner).register_buffer(attr, tensor, persistent=False)
    left = [n for n, t in list(tm.named_parameters()) + list(tm.named_buffers()) if t.is_meta]
    if left:
        raise RuntimeError(f"still on meta after loading: {left[:5]}")
    tm.config.dtype = data["dtype"]
    tm.device = device
    return tm


def wire_up(tm, base_lm: str, acoustic: str) -> None:
    """After loading from disk: KV cache in each part's dtype + casts at the BaseLM
    boundaries. BEFORE `optimize()` — compilation remembers the dtypes."""
    base, rest = _dtypes(base_lm, acoustic)
    tm.residual_lm.setup_cache(1, tm.config.max_length, tm.device, rest)
    bl = tm.base_lm
    bl.setup_cache(1, tm.config.max_length, tm.device, base)
    if base == rest:
        return

    def to(t, dtype):
        return t.to(dtype) if torch.is_tensor(t) and t.is_floating_point() else t

    # voxcpm calls `embed_tokens` DIRECTLY and joins the result with audio features (the rest's dtype).
    bl.embed_tokens.register_forward_hook(lambda m, inp, out: to(out, rest))
    bl.register_forward_pre_hook(
        lambda m, a, k: (tuple(to(x, base) for x in a), {n: to(v, base) for n, v in k.items()}),
        with_kwargs=True,
    )
    # `forward` output = (states, kv) — kv stays in the BaseLM dtype, it goes into its cache.
    bl.register_forward_hook(
        lambda m, a, k, out: (to(out[0], rest),) + tuple(out[1:]),
        with_kwargs=True,
    )
    # `forward_step` is not `__call__` — hooks don't see it, hence the wrapper.
    step = bl.forward_step
    bl.forward_step = lambda x, pos, *a, **k: to(step(to(x, base), pos, *a, **k), rest)


def report(tm) -> str:
    """Which dtype actually sits in each part — evidence instead of faith."""
    lines = []
    for name in ("base_lm", "residual_lm", "feat_encoder", "feat_decoder", "audio_vae"):
        part = getattr(tm, "_feat_encoder_raw", None) if name == "feat_encoder" else None
        part = part if part is not None else getattr(tm, name)
        dtypes = sorted({(type(w).__name__ + "/" if type(w).__name__ not in ("Parameter", "Tensor") else "")
                         + str(w.dtype).replace("torch.", "") for w in part.parameters()})
        int8 = sum(isinstance(m, Int8Linear) for m in part.modules())
        if int8:
            dtypes.append(f"Int8Linear×{int8}")
        lines.append(f"[precision] {name}: {', '.join(dtypes)}")
    return "\n".join(lines)
