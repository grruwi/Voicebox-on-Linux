# Voicebox on Linux with audio.cpp, Vulkan, ROCm and PyTorch to choose from.

[Voicebox](https://github.com/jamiepine/voicebox) (a voice-cloning panel, MIT) reworked so that
speech comes from [audio.cpp](https://github.com/0xShug0/audio.cpp) (ggml, Vulkan or ROCm), speech
recognition from [whisper.cpp](https://github.com/ggml-org/whisper.cpp), and — optionally — from a
PyTorch engine on ROCm. Any card with Vulkan is enough for the default setup.

An unofficial rework, not affiliated with the Voicebox or audio.cpp authors — report bugs here.

**Prebuilt VoxCPM2 weights (our quants):** [huggingface.co/grruwi](https://huggingface.co/grruwi) —
[for audio.cpp](https://huggingface.co/grruwi/VoxCPM2-audiocpp-prebuilt) ·
[for PyTorch](https://huggingface.co/grruwi/VoxCPM2-pytorch-prebuilt)

Voices (both clone from a sample):
- **VoxCPM2**: streaming, first sound after ~0.2 s; weight precision (q8/f16/f32) picked in the panel;
- **OmniVoice** (k2-fsa): diffusion, the whole utterance at once, ~3× faster than real time;
  q8 or f16, optional chunked streaming (by lines or sentences).

Sample editor next to each profile: select on the waveform, cut, loop, A/B with the original,
Ctrl+Z/Y, live filters — de-esser, de-click, treble shelf, gate (ffmpeg).

## Containers

| container | what it does |
|---|---|
| `voicebox` | API, web panel (port 17493), MCP server (`/mcp`), voice profiles. Starts empty — no model loads with it |
| `audiocpp-backend` | audio.cpp with our patches; holds **one model at a time** (Vox or OmniVoice) |
| `whisper` | whisper.cpp; model, quantization and GPU/CPU are chosen in the panel |
| `pytorch-backend` | *optional* — VoxCPM2 on PyTorch/ROCm (see below) |

**One model on the card.** Before a model loads, Voicebox's Offloader unloads every other engine
container (`/state` + `/unload` — any container that speaks this contract, present or future).
The containers stay up, empty (a few MB). *Generation → General → Keep all models in VRAM* turns
the Offloader off.

**Whisper is separate on purpose** — its own process on Vulkan means full duplex: you can dictate
while a voice is speaking. Inside Voicebox's process the two would queue behind each other.
Its settings still live in the Voicebox panel.

**Headless.** Every engine container runs without Voicebox. Their servers are trimmed down but
engine-agnostic and OpenAI-compatible (`POST /v1/audio/speech`, voice cloning included) — load a
model through the container's manager first (`POST /load`, see `*/manager.py`).

## What it costs

| | disk | VRAM | RAM |
|---|---|---|---|
| images: voicebox + audiocpp + whisper | ~3 GB (with the ROCm variant ~6.3 GB) | — | — |
| VoxCPM2, audio.cpp | 4.96 GB original + 3.4–4.7 GB per prebuilt set | ~4–5 GB | ~1.5 GB |
| OmniVoice, audio.cpp | 1.35 GB (q8) / 1.64 GB (f16) | ~3 GB | |
| Whisper large-v3 q5 | ~1.1 GB | ~1.6 GB | |
| PyTorch image | ~13 GB | — | — |
| VoxCPM2, PyTorch | 4.7 GB original + 3.5–6.3 GB per prebuilt set | ~5.1 GB (q8/f16) | ~3.3 GB |

Vox and OmniVoice never sit on the card together (unless you turn the Offloader off).

## Requirements

- Linux, Docker with `docker compose`
- a card with Vulkan:
  - **AMD and Intel** work out of the box (a new Mesa from trixie-backports is in the images);
  - **NVIDIA** is untested: needs `nvidia-container-toolkit`; a commented variant is in `docker-compose.yml`.

## Start

```bash
git clone https://github.com/grruwi/Voicebox-on-Linux.git
cd Voicebox-on-Linux
./download-models.sh            # minimum ~11 GB; `all` = every variant, `pytorch` = the PyTorch engine
cp .env.example .env            # optional: an LLM server for transcript refine
docker compose up -d --build    # the first build compiles audio.cpp and whisper.cpp
```

Panel: `http://localhost:17493`.

1. **Profiles** → new profile → a 5–30 s voice sample + its exact transcript.
2. Engine **VoxCPM2** or **OmniVoice** → generate.

On the first start Vulkan also compiles its shaders, so the first load takes longer.

## Prebuilt weights (quants) — do it

Prebuilt sets save resources big time. Without them the engine takes the BF16 original and
converts it into the panel's precision **on every load** — slow, and RAM can spike hard while
both copies sit in memory. With them a Vox load is ~2 s instead of ~7 s (~23 s from a cold disk).
On PyTorch they are also faster, and **required** (the worker loads only prebuilt sets).

**Download ours** — cut the way the panel splits the model (BaseLM × acoustic, 4 sets each):
[VoxCPM2-audiocpp-prebuilt](https://huggingface.co/grruwi/VoxCPM2-audiocpp-prebuilt) and
[VoxCPM2-pytorch-prebuilt](https://huggingface.co/grruwi/VoxCPM2-pytorch-prebuilt).
`download-models.sh` fetches the q8/f16 set by default (`all` / `pytorch-all` = every set).
The `.pt` files load with `weights_only=True` — no code runs from them.

**Or make them yourself** from the original:

```bash
# audio.cpp: <base_lm> q8_0|f16|f32, <acoustic> f16|f32 → models/audiocpp/prebuilt/<set>/
docker compose run --rm audiocpp python /app/make_weights.py q8_0 f16

# PyTorch: no argument = all four sets (q8-f16, q8-f32, f16-f16, f16-f32), on the CPU
docker compose -f docker-compose.yml -f docker-compose.pytorch.yml run --rm \
    -w /app/workers/voxcpm2 pytorch python make_weights.py q8-f16
```

The panel picks a prebuilt set by itself when one matches; the bits are the same as the
on-the-fly conversion.

## Optional: ROCm for audio.cpp (Radeons only)

**Not needed** — Vulkan runs on any card. The ROCm variant adds a HIP binary (ROCm 10, TheRock)
next to the Vulkan one; switch in the panel (Generation → General → Backend), no rebuild.

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d --build
```

ROCm is **compiled for specific GPUs** (`GPU_TARGETS` in `docker-compose.rocm.yml`, default
RX 9000 + RX 7000). Every extra card = a longer build and ~0.5 GB more in the image, plus a
~2.4 GB ROCm download per card family. List only your cards.

## Optional: PyTorch engine (ROCm)

Implemented properly, and **not needed** if audio.cpp's ggml is all you want. Its own perks —
on Radeons it is faster per sentence — and it is the **only road to training** (fine-tuning,
LoRA), at the cost of resources (table above).

```bash
./download-models.sh pytorch
docker compose -f docker-compose.yml -f docker-compose.pytorch.yml up -d --build
```

That fetches the original + the q8/f16 set (`pytorch-all` = all four). The image carries a prebuilt kernel package per card
(RX 9070 XT + RX 7900 XTX by default, ~0.6–0.7 GB each) — another card = add its package in
`pytorch/Dockerfile`. The first load compiles (`torch.compile`); after that it runs from the caches.

**The cache paths in the compose files are not optional** (MIOpen, Inductor, Triton, Mesa, the
voice-sample caches). Without them every start compiles and encodes everything again.

**ROCm 10 on RDNA4:** the PyTorch container sets `HSA_TOOLS_DISABLE_REGISTER=1`. Without it one
HSA thread spins a whole CPU core forever while a model sits loaded
([ROCm/rocm-systems#12431](https://github.com/ROCm/rocm-systems/issues/12431)). It only turns off
the profiler registration (`rocprofv3`).

## API

On the panel's port (17493). Both APIs go to the same queue and the same engine.

### OpenAI-compatible — for other programs

Base URL `http://<host>:17493/v1`, any API key (not checked).

| endpoint | what it does |
|---|---|
| `POST /v1/audio/speech` | speech; `input` = text, `voice` = a Voicebox profile name (empty = default), `response_format` = `mp3` (default) / `opus` / `aac` / `flac` / `wav` / `pcm` |
| `POST /v1/audio/transcriptions` | speech recognition (the whisper picked in the panel); `response_format` = `json` / `text` / `verbose_json` |
| `GET /v1/audio/voices` | the profiles |
| `GET /v1/audio/models` | one model, `voicebox` |

`model` and `speed` are ignored: the engine comes from the profile, tempo and the other knobs
from the panel. `instructions`: a style for Vox (e.g. "whispering"), a voice description for
OmniVoice (gender, age, pitch, accent).

```bash
curl http://<host>:17493/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "Hi, this is Voicebox.", "voice": "my profile"}' -o speech.mp3
```

**Open WebUI:** Admin panel → Settings → Audio → TTS and STT: engine *OpenAI*,
URL `http://<host>:17493/v1`, any key, voice = a profile name.

### Voicebox's own API

What the panel uses: per-request engine, seed, effects, generation history, chunked streaming.
Full list: `http://<host>:17493/docs`.

### MCP

`http://<host>:17493/mcp` — tools for AI agents (`voicebox_speak`, `voicebox_transcribe`…).

## Notes

- **Port 17493 has no login.** It is bound to loopback; do not expose it without a reverse proxy with auth.
- **Autoplay on generate** (Settings → General): on = the panel plays the speech as a stream in
  the browser while it is computed (~0.2 s to the first sound with Vox); off = the server's speakers.
- **Loudness normalization turns streaming off** — it needs the whole recording. Same for effects and retry.
- `voicebox_speak` (MCP) plays on the **server's** speakers — uncomment the PulseAudio socket in the compose file.
- Data lives in `./data/voicebox`: database, profiles, generations. Backup = copy that directory.

## What changed against the upstreams

- `audiocpp/voicebox.patch` — 12 patches on audio.cpp @ `497c7f0`:
  - stateful AudioVAE stream decoder; upsample weights rewritten once at load (decoder 36 → 3.6 ms/chunk);
  - LM KV cache in F16 + flash attention on views; prefill state stays on the card;
  - f32 output (`sample_format`);
  - on-disk cache of encoded voice samples (profile change 2.4 s → 4 ms);
  - no LM graph rebuild per request; prefill on flash attention + one-graph text embed
    (first sound 276 → 170 ms);
  - fast load of prebuilt weight files (7 → 1.75 s); a separate BaseLM precision.
- `audiocpp/manager.py`, `pytorch/manager.py`, `whisper/manager.py` — small HTTP servers through
  which Voicebox starts and stops the engine processes (one model each).
- `pytorch/workers/voxcpm2/` — VoxCPM2 on PyTorch: prebuilt q8/f16 sets, flash-attention decode,
  voice-sample cache, an OpenAI-compatible layer.
- `voicebox/` — a Voicebox fork: talks to the engines over HTTP, the engine-agnostic Offloader,
  engine tabs in the panel, the engine list cut down to VoxCPM2 (ggml / PyTorch) and OmniVoice.

## Licenses

- `voicebox/` — Voicebox, [MIT](voicebox/LICENSE), © Voicebox Contributors; our changes under the same license.
- audio.cpp — [Apache 2.0](audiocpp/LICENSE-audio.cpp), © ShugoAI LLC. Fetched at build time;
  `audiocpp/voicebox.patch` is our change to their files (Apache 2.0).
- whisper.cpp — MIT, fetched at build time.
- ROCm, PyTorch — AMD / PyTorch licenses inside the packages, fetched at build time.

## Authors

Put together by [grruwi](https://github.com/grruwi) with Claude
