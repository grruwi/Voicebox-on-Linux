"""Flash attention for the single-token decode step of MiniCPM (BaseLM + ResidualLM).

Upstream `MiniCPMAttention.forward_step` attends over the WHOLE static KV cache
(8192 slots) with a boolean mask. On ROCm there is no flash/efficient kernel for
mask + GQA, so SDPA falls back to MATH: K/V expanded 2 -> 16 heads and the matmuls
upcast to FP32 — ~2.2 ms per layer, 99 ms per BaseLM step (profiled 2026-10-11).

`varlen_attn` reads the live length from a tensor (`cu_seq_k`), so every shape stays
static and CUDA graphs (`reduce-overhead`) survive: ~0.024 ms per layer.
Prefill, DiT and LocEnc already hit flash (no mask) — this covers the last path.

Second fix — per-layer KV cache. Upstream keeps ONE tensor for all layers
(2, 28, 1, 2, 8192, 128) and hands out views; writing one token into a view made
inductor copy the WHOLE stacked cache (~235 MB) — once per layer, ~1 ms each,
29 ms per BaseLM step (profiled 2026-10-11). Separate tensors per layer, same API.
"""

import torch
from torch.nn.attention.varlen import varlen_attn
from voxcpm.modules.minicpm4 import model as minicpm

_upstream_forward_step = minicpm.MiniCPMAttention.forward_step


def _flash_forward_step(self, hidden_states, position_emb, position_id, kv_cache):
    bsz, _ = hidden_states.size()
    if bsz != 1:
        return _upstream_forward_step(self, hidden_states, position_emb, position_id, kv_cache)

    q = self.q_proj(hidden_states).view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(1, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(1, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    if position_emb is not None:
        cos, sin = position_emb
        q, k = minicpm.apply_rotary_pos_emb(q, k, cos, sin)

    key_cache, value_cache = kv_cache
    key_cache[:, :, position_id, :] = k
    value_cache[:, :, position_id, :] = v

    # (1, Hkv, L, D) -> (L, Hkv, D): a view, the head dim stays contiguous.
    keys = key_cache[0].transpose(0, 1)
    values = value_cache[0].transpose(0, 1)
    live = (position_id.reshape(1) + 1).to(torch.int32)
    cu_seq_k = torch.cat([live.new_zeros(1), live])
    cu_seq_q = torch.arange(2, device=live.device, dtype=torch.int32)

    out = varlen_attn(
        q.reshape(1, self.num_heads, self.head_dim), keys, values,
        cu_seq_q, cu_seq_k, 1, key_cache.size(2), enable_gqa=True,
    )
    return self.o_proj(out.reshape(1, self.num_heads * self.head_dim))


def _cache_init(self, num_layers, num_kv_heads, dim_kv_head, batch_size, device, dtype, max_length=8192):
    self.max_length = max_length
    self.num_layers = num_layers
    shape = (batch_size, num_kv_heads, max_length, dim_kv_head)
    self.layers = [
        (torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype))
        for _ in range(num_layers)
    ]
    self.current_length = 0


def _cache_get_layer(self, layer_idx):
    return self.layers[layer_idx]


def _cache_fill(self, kv_caches):
    self.current_length = kv_caches[0][0].size(2)
    for (k, v), (new_k, new_v) in zip(self.layers, kv_caches):
        k.zero_()
        v.zero_()
        k[:, :, : self.current_length, :] = new_k
        v[:, :, : self.current_length, :] = new_v


def enable_fast_decode() -> None:
    """Patch the classes (BaseLM and ResidualLM share them).
    Call BEFORE the model is built — `setup_cache` allocates the KV cache."""
    minicpm.MiniCPMAttention.forward_step = _flash_forward_step
    cache = minicpm.StaticKVCache
    cache.__init__ = _cache_init
    cache.get_layer_cache = _cache_get_layer
    cache.fill_caches = _cache_fill
