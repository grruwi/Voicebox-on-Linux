"""VoxCPM2 worker of the `pytorch` container — a thin HTTP layer over `voxcpm.VoxCPM`.

Started by `manager.py`: the precision set comes in `WORKER_OPTIONS`
(`{"base_lm": "q8"|"f16", "acoustic": "f16"|"f32", "cuda_graphs": true, "compile": true}`), the model
loads during startup, and the port opens only once it is ready. Another set = the
manager starts a new process; this one never swaps models.

The VoxCPM2 repo on HuggingFace ships WEIGHTS ONLY, no server code (unlike
fish S1, which has `tools/api_server.py` in the box). Upstream's official route
is vLLM-Omni — a separate heavy stack. This replaces it: a few hundred lines that
do exactly what is needed to call the model over HTTP.

⛔ NO DEFAULT VALUES ON THIS SIDE. A field that was not sent is not passed to
`generate()`, so the engine uses its own. The knobs belong to Voicebox.

⚠️ TWO CLONING PATHS, which upstream does NOT combine freely:
    reference_wav_path            — timbre only, no transcript
    prompt_wav_path + prompt_text — continuation, both or neither (otherwise it raises)
Voicebox decides which one to use — this side only passes it on.

Output: 48 kHz, native, no resampling.
"""

import base64
import io
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

WEIGHTS_DIR = os.environ.get("VOXCPM_WEIGHTS", "/weights")
SAMPLE_RATE = 48000

model = None
# Loaded precision set: (base_lm, acoustic).
precision = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    options = json.loads(os.environ.get("WORKER_OPTIONS") or "{}")
    _load(options.get("base_lm"), options.get("acoustic"), options.get("cuda_graphs") is not False,
          options.get("compile") is not False)
    yield
    global model
    model = None


app = FastAPI(lifespan=lifespan)


# -------------------------------------------------------------- precision
#
# ⭐ PREBUILT SETS FROM DISK (`prebuilt/<base_lm>-<acoustic>/model.pt`, made by
# `make_weights.py`). The server converts NOTHING — it reads a ready set, sets up
# the KV cache and the casts at the BaseLM boundaries (`precision.py`), compiles.
# Compilation (`torch.compile`) is a separate matter: once per set, then from the
# on-disk caches (`inductor-cache`, `triton-cache`).

PREBUILT_DIR = Path(os.environ.get("VOXCPM_PREBUILT", "/prebuilt"))
# Encoded voice samples (see prompt_cache.py) — a mounted dir, survives restarts.
PROMPT_CACHE_DIR = Path(os.environ.get("VOXCPM_PROMPT_CACHE", "/prompt-cache"))


def _kill_compile_workers() -> None:
    """⚠️ `torch.compile` starts a worker pool — one per core the container has
    (14 × ~240 MiB). After compiling they just sit there. Killed workers come back
    on their own when something needs compiling (`process_pool` is a cache,
    `after_fork` clears it).
    """
    try:
        from torch._inductor.async_compile import shutdown_compile_workers

        shutdown_compile_workers()
    except Exception as error:
        print(f"[memory] compile workers stay: {error}", flush=True)


def _compile_without_graphs(tm) -> None:
    """Same as upstream `optimize()`, only WITHOUT CUDA graphs (`reduce-overhead`)
    — Triton kernels stay, graph recording goes. Tests grruwi's hypothesis: in ggml
    HIP graphs slowed things down (first sound 1.2 → 0.12 s once disabled)."""
    import torch

    tm.base_lm.forward_step = torch.compile(tm.base_lm.forward_step, fullgraph=True)
    tm.residual_lm.forward_step = torch.compile(tm.residual_lm.forward_step, fullgraph=True)
    tm._feat_encoder_raw = tm.feat_encoder
    tm.feat_encoder = torch.compile(tm.feat_encoder, fullgraph=True)
    tm.feat_decoder.estimator = torch.compile(tm.feat_decoder.estimator, fullgraph=True)


def _load(base_lm: Optional[str], acoustic: Optional[str], cuda_graphs: bool = True,
          compile_model: bool = True) -> None:
    global model, precision
    from voxcpm import VoxCPM

    from precision import ACOUSTIC, BASE_LM, load, report, set_name, wire_up

    if base_lm not in BASE_LM or acoustic not in ACOUSTIC:
        raise ValueError(f"unknown precision set: base_lm={base_lm}, acoustic={acoustic}")
    path = PREBUILT_DIR / set_name(base_lm, acoustic) / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"no prebuilt set {path.parent.name} — run make_weights.py")

    # ⭐ Compiles WITH THE WORKER POOL (as many as cores in the cpuset, × ~400 MiB
    # in Mission Center) — grruwi's call: faster, and `_kill_compile_workers`
    # kills the pool right after loading. These are NOT leftovers from generations.

    print(f"[voxcpm2] loading prebuilt set {path.parent.name} ...", flush=True)
    # Flash attention in the decode step + per-layer KV cache — see attention.py.
    from attention import enable_fast_decode

    enable_fast_decode()
    tm = load(path, WEIGHTS_DIR)
    wire_up(tm, base_lm, acoustic)
    from prompt_cache import enable_prompt_cache

    enable_prompt_cache(tm, PROMPT_CACHE_DIR)
    # `compile: false` = eager, no torch.compile at all (no Dynamo trace on every load).
    if not compile_model:
        mode = "off (eager)"
    elif cuda_graphs:
        tm.optimize()
        mode = "reduce-overhead (CUDA graphs)"
    else:
        _compile_without_graphs(tm)
        mode = "no CUDA graphs"
    print(f"[voxcpm2] compile: {mode}", flush=True)
    # Same warm-up as upstream with `optimize=True` — compiles the graphs.
    tm.generate(target_text="Hello, this is the first test sentence.", max_len=10)
    m = VoxCPM.__new__(VoxCPM)
    # ⛔ No denoiser (ZipEnhancer, 11.10, grruwi's call): 7.8 s of every load (import +
    # ModelScope pipeline), cleans only the profile sample, and in his listening it
    # provoked hallucinations. `denoise` in a request is ignored.
    m.tts_model, m.text_normalizer, m.denoiser = tm, None, None
    model, precision = m, (base_lm, acoustic)
    _kill_compile_workers()
    print(report(tm), flush=True)
    print("[voxcpm2] ready", flush=True)


class Request(BaseModel):
    text: str
    # Samples as base64 — same form as fish S1, so protocols don't multiply.
    reference_wav: Optional[str] = None
    prompt_wav: Optional[str] = None
    prompt_text: Optional[str] = None
    cfg_value: Optional[float] = None
    inference_timesteps: Optional[int] = None
    normalize: Optional[bool] = None
    retry_badcase: Optional[bool] = None
    seed: Optional[int] = None


@app.get("/health")
def health():
    # The port opens only after the model is loaded (see `lifespan`).
    return {"status": "ok", "precision": precision}


def _release() -> None:
    """⚠️ Voicebox's garbage collector (`services/memory.py`) does NOT reach this —
    this is a separate process with its own torch allocator. Without this the cache
    grows between sentences and the container holds several GB it does not use.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            before = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()
            after = torch.cuda.memory_reserved()
            print(f"[memory] GPU {before/1048576:.0f} -> {after/1048576:.0f} MiB "
                  f"(freed {(before-after)/1048576:.0f})", flush=True)
    except Exception as error:
        print(f"[memory] empty_cache skipped: {error}", flush=True)
    _kill_compile_workers()
    # ⚠️ Loading weights goes through RAM (the state from the file sits on the CPU).
    # Freed pages stay in glibc arenas and RSS sits at ~4 GB — malloc_trim returns
    # them. Same trick as `services/memory.py` in Voicebox.
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as error:
        print(f"[memory] malloc_trim skipped: {error}", flush=True)


def _save(data_b64: str, directory: str, name: str) -> str:
    path = Path(directory) / name
    path.write_bytes(base64.b64decode(data_b64))
    return str(path)


def _params(request: Request, directory: str) -> dict:
    params = {"text": request.text}
    for name in ("cfg_value", "inference_timesteps", "normalize", "retry_badcase"):
        value = getattr(request, name)
        if value is not None:
            params[name] = value
    if request.reference_wav:
        params["reference_wav_path"] = _save(request.reference_wav, directory, "ref.wav")
    if request.prompt_wav:
        params["prompt_wav_path"] = _save(request.prompt_wav, directory, "prompt.wav")
        if request.prompt_text:
            params["prompt_text"] = request.prompt_text
    if request.seed is not None:
        import torch

        torch.manual_seed(request.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.seed)
    return params


@app.post("/generate")
def generate(request: Request):
    with _busy, tempfile.TemporaryDirectory() as directory:
        params = _params(request, directory)
        try:
            audio = model.generate(**params)
        except Exception as error:
            raise HTTPException(500, f"{type(error).__name__}: {error}")
        finally:
            _release()

    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio, dtype=np.float32), SAMPLE_RATE, format="WAV")
    return Response(content=buffer.getvalue(), media_type="audio/wav")


@app.post("/profile")
def profile(request: Request):
    """DIAGNOSTICS, Voicebox never calls this. Two generations of the same text:
    (1) time of each model part (synchronized before and after), (2) `torch.profiler`
    — kernels sorted by GPU time."""
    import time
    from collections import defaultdict

    import torch
    from fastapi.responses import PlainTextResponse
    from torch.profiler import ProfilerActivity, profile

    tm = model.tts_model
    # ⛔ Safety limits (11.10: a broken model never finished its sentences, voxcpm
    # retried, and the profiler recorded every kernel — >20 GB RAM). No retries,
    # a ceiling on steps.
    params = {"text": request.text, "retry_badcase": False, "max_len": 300}
    for name in ("cfg_value", "inference_timesteps"):
        if getattr(request, name) is not None:
            params[name] = getattr(request, name)

    times = defaultdict(list)
    restore = []

    def measure(obj, attr, name, module=False):
        original = getattr(obj, attr)

        def wrapped(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            result = original(*a, **k)
            torch.cuda.synchronize()
            times[name].append(time.perf_counter() - t)
            return result

        # A child module (`feat_encoder`, `feat_decoder`) lives in `_modules` — a
        # function in the instance `__dict__` shadows it, `pop` restores it.
        if module:
            object.__setattr__(obj, attr, wrapped)
            restore.append(lambda: obj.__dict__.pop(attr))
        elif attr in obj.__dict__:  # e.g. a compiled `forward_step`
            setattr(obj, attr, wrapped)
            restore.append(lambda: setattr(obj, attr, original))
        else:  # a class method (`forward`, `decode`) — removing ours uncovers it again
            setattr(obj, attr, wrapped)
            restore.append(lambda: obj.__dict__.pop(attr))

    measure(tm.base_lm, "forward", "BaseLM prefill")
    measure(tm.base_lm, "forward_step", "BaseLM step")
    measure(tm.residual_lm, "forward", "ResidualLM prefill")
    measure(tm.residual_lm, "forward_step", "ResidualLM step")
    measure(tm, "feat_encoder", "LocEnc", module=True)
    measure(tm, "feat_decoder", "DiT (all CFM steps)", module=True)
    measure(tm.audio_vae, "decode", "AudioVAE decode")
    try:
        torch.manual_seed(1234)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        audio = model.generate(**params)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
    finally:
        for f in reversed(restore):
            f()

    seconds = len(audio) / SAMPLE_RATE
    steps = len(times["BaseLM step"]) or 1
    lines = [f"text {len(request.text)} chars, audio {seconds:.2f} s, generation {total:.2f} s, "
             f"RTF {total / seconds:.2f}, steps {steps}, {total / steps * 1e3:.1f} ms/step", ""]
    measured = 0.0
    for name, values in times.items():
        measured += sum(values)
        lines.append(f"{name:28s} {len(values):4d}× {sum(values):7.2f} s  {sum(values) / len(values) * 1e3:8.2f} ms/call  "
                     f"{sum(values) / steps * 1e3:7.2f} ms/step")
    lines.append(f"{'rest (Python, stop, proj.)':28s}       {total - measured:7.2f} s  "
                 f"{'':21s}{(total - measured) / steps * 1e3:7.2f} ms/step")

    if steps >= params["max_len"] - 1:
        _release()
        lines.append("⛔ the model did not finish the sentence (step ceiling) — not running the profiler")
        return PlainTextResponse("\n".join(lines))

    torch.manual_seed(1234)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        model.generate(**params)
        torch.cuda.synchronize()
    _release()
    lines += ["", p.key_averages().table(sort_by="self_device_time_total", row_limit=30, max_name_column_width=70)]
    return PlainTextResponse("\n".join(lines))


@app.post("/generate/stream")
def generate_stream(request: Request):
    """Same as `/generate`, only chunks go out as they are made.

    ⚠️ TOTAL TIME IS THE SAME — measured: RTF 0.648 vs 0.653.
    There is one gain, but a big one: FIRST SOUND AFTER ~0.5 s instead of after 19 s.

    ⛔ NO WAV HEADER. Raw `float32 little-endian`, 48 kHz, mono — the receiver
    knows the format up front. A WAV header carries the LENGTH, which is not
    known yet when the stream starts.
    """
    directory = tempfile.mkdtemp()
    params = _params(request, directory)

    def chunks():
        import shutil

        with _busy:
            try:
                for chunk in model.generate_streaming(**params):
                    yield np.asarray(chunk, dtype=np.float32).tobytes()
            finally:
                shutil.rmtree(directory, ignore_errors=True)
                _release()

    return StreamingResponse(chunks(), media_type="application/octet-stream")


# ── OpenAI-compatible layer ───────────────────────────────────────────────────
# The request shape of the audio.cpp server (which follows OpenAI's), so any OpenAI
# TTS client (Open WebUI, SillyTavern, ...) can call this model without Voicebox:
#
#   GET  /v1/models
#   POST /v1/audio/speech
#     input            the text (required)
#     voice            required by OpenAI's schema, ignored here (no presets)
#     voice_ref        cloning sample: a path in the container, or
#                      {"type": "path", "path": ...} | {"type": "base64", "data": ...}
#     reference_text   its transcript → continuation cloning; without it, timbre only
#     instructions     voice design — goes in front of the text as "(...)"
#     seed
#     response_format  wav (default) | flac | pcm | json | b64_json
#     sample_format    s16 (default) | f32
#     stream           true → pcm chunks; stream_format sse (default) | audio
#
# One request at a time — the model is shared with Voicebox's routes (`_busy`).

import threading  # noqa: E402

_busy = threading.Lock()
_MAX_REF_BYTES = 5 * 1024 * 1024  # same bound as audio.cpp: a sample, not a file dump


class SpeechRequest(BaseModel):
    input: str
    model: Optional[str] = None
    voice: Optional[str] = None
    voice_ref: Optional[object] = None
    reference_text: Optional[str] = None
    instructions: Optional[str] = None
    seed: Optional[int] = None
    speed: Optional[float] = None
    response_format: Optional[str] = None
    sample_format: Optional[str] = None
    stream: Optional[bool] = None
    stream_format: Optional[str] = None


def _ref_b64(voice_ref) -> Optional[str]:
    if voice_ref is None:
        return None
    if isinstance(voice_ref, str):
        voice_ref = {"type": "path", "path": voice_ref}
    if not isinstance(voice_ref, dict):
        raise HTTPException(400, 'voice_ref: a path, or {"type": "path"|"base64", ...}')
    if voice_ref.get("type") == "path":
        path = Path(str(voice_ref.get("path") or ""))
        if not path.is_file():
            raise HTTPException(400, f"voice_ref: no file {path}")
        data = path.read_bytes()
    elif voice_ref.get("type") == "base64":
        data = base64.b64decode(str(voice_ref.get("data") or "").split(",")[-1])
    else:
        raise HTTPException(400, 'voice_ref type must be "path" or "base64"')
    if not data or len(data) > _MAX_REF_BYTES:
        raise HTTPException(400, "voice_ref: empty or over 5 MiB")
    return base64.b64encode(data).decode()


def _as_request(body: SpeechRequest) -> Request:
    if body.speed is not None and body.speed != 1.0:
        raise HTTPException(400, "speed: VoxCPM2 has no speed control")
    text = f"({body.instructions}){body.input}" if body.instructions else body.input
    ref = _ref_b64(body.voice_ref)
    if ref and body.reference_text:
        return Request(text=text, prompt_wav=ref, prompt_text=body.reference_text, seed=body.seed)
    return Request(text=text, reference_wav=ref, seed=body.seed)


def _pcm(audio, f32: bool) -> bytes:
    samples = np.asarray(audio, dtype=np.float32)
    if f32:
        return samples.tobytes()
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": "voxcpm2", "object": "model", "owned_by": "openbmb", "loaded": model is not None,
         "precision": precision},
    ]}


@app.post("/v1/audio/speech")
def speech(body: SpeechRequest):
    sample_format = body.sample_format or "s16"
    if sample_format not in ("s16", "f32"):
        raise HTTPException(400, "sample_format must be s16 or f32")
    f32 = sample_format == "f32"
    request = _as_request(body)
    headers = {"X-Sample-Format": sample_format, "X-Sample-Rate": str(SAMPLE_RATE)}

    if body.stream or body.stream_format:
        if (body.response_format or "pcm") != "pcm":
            raise HTTPException(400, "streaming speech supports response_format=pcm")
        stream_format = body.stream_format or "sse"
        if stream_format not in ("sse", "audio"):
            raise HTTPException(400, "stream_format must be sse or audio")
        directory = tempfile.mkdtemp()
        params = _params(request, directory)

        def chunks():
            import shutil

            with _busy:
                try:
                    for chunk in model.generate_streaming(**params):
                        pcm = _pcm(chunk, f32)
                        if stream_format == "audio":
                            yield pcm
                        else:
                            event = {"type": "speech.audio.delta", "audio": base64.b64encode(pcm).decode()}
                            yield f"data: {json.dumps(event)}\n\n".encode()
                    if stream_format == "sse":
                        yield b'data: {"type": "speech.audio.done"}\n\ndata: [DONE]\n\n'
                finally:
                    shutil.rmtree(directory, ignore_errors=True)
                    _release()

        media = "application/octet-stream" if stream_format == "audio" else "text/event-stream"
        return StreamingResponse(chunks(), media_type=media, headers=headers)

    response_format = body.response_format or "wav"
    if response_format not in ("wav", "flac", "pcm", "json", "b64_json"):
        raise HTTPException(400, "response_format: wav | flac | pcm | json | b64_json")
    with _busy, tempfile.TemporaryDirectory() as directory:
        params = _params(request, directory)
        try:
            audio = model.generate(**params)
        except Exception as error:
            raise HTTPException(500, f"{type(error).__name__}: {error}")
        finally:
            _release()

    if response_format == "pcm":
        return Response(content=_pcm(audio, f32), media_type="application/octet-stream", headers=headers)
    buffer = io.BytesIO()
    subtype = "FLOAT" if f32 else "PCM_16"
    if response_format == "flac":
        sf.write(buffer, np.asarray(audio, dtype=np.float32), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        return Response(content=buffer.getvalue(), media_type="audio/flac", headers=headers)
    sf.write(buffer, np.asarray(audio, dtype=np.float32), SAMPLE_RATE, format="WAV", subtype=subtype)
    if response_format in ("json", "b64_json"):
        return {"audio": base64.b64encode(buffer.getvalue()).decode(), "format": "wav"}
    return Response(content=buffer.getvalue(), media_type="audio/wav", headers=headers)
