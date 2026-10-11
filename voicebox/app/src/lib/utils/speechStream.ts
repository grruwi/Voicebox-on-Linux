// Player for the backend's speech stream (`GET /generate/{id}/browser-stream`).
//
// "Autoplay on generate" checked = the backend does not play through ffplay, it hands the
// chunks over here as they are computed: 4 bytes of sample rate (uint32 LE), then raw f32le
// mono. Every chunk goes straight into Web Audio, scheduled right after the previous one —
// the first sound after the engine's TTFT + a few dozen ms. Next generations queue up behind
// each other, they do not interrupt.
//
// 204 = this generation is not streamed (effects, retry, text too long) — then the panel
// plays the whole file as before.

// Headroom at the start and after a gap. Less = faster, but every network delay is heard as a click.
const HEADROOM_S = 0.04;

let context: AudioContext | null = null;
let nextStart = 0;
const active = new Set<AudioBufferSourceNode>();
const aborts = new Set<AbortController>();
// Generations that played at least one chunk — the panel does NOT play these again from the file.
const played = new Set<string>();

function audio(): AudioContext {
  if (!context) context = new AudioContext({ latencyHint: 'interactive' });
  // The browser keeps the context suspended until the first interaction with the page;
  // clicking "Generate" is such an interaction, so it wakes up here.
  if (context.state === 'suspended') void context.resume();
  return context;
}

export function playedByStream(id: string): boolean {
  return played.has(id);
}

export function stopStreams(): void {
  for (const a of aborts) a.abort();
  aborts.clear();
  for (const source of active) {
    try {
      source.stop();
    } catch {
      // already finished
    }
  }
  active.clear();
  nextStart = 0;
}

export async function playStream(url: string, id: string): Promise<void> {
  const ctx = audio();
  const abort = new AbortController();
  aborts.add(abort);
  try {
    const response = await fetch(url, { signal: abort.signal, cache: 'no-store' });
    if (response.status !== 200 || !response.body) return;

    const reader = response.body.getReader();
    let sampleRate = 0;
    let rest = new Uint8Array(0);

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (!value?.length) continue;

      let data: Uint8Array;
      if (rest.length) {
        data = new Uint8Array(rest.length + value.length);
        data.set(rest);
        data.set(value, rest.length);
      } else {
        data = value;
      }

      if (!sampleRate) {
        if (data.length < 4) {
          rest = data.slice();
          continue;
        }
        sampleRate = new DataView(data.buffer, data.byteOffset, 4).getUint32(0, true);
        data = data.subarray(4);
      }

      const count = Math.floor(data.length / 4);
      rest = data.slice(count * 4);
      if (!count) continue;

      // a copy — Float32Array needs 4-byte alignment, and a chunk from the network does not guarantee it
      const samples = new Float32Array(data.slice(0, count * 4).buffer);
      const buffer = ctx.createBuffer(1, count, sampleRate);
      buffer.copyToChannel(samples, 0);

      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.onended = () => active.delete(source);

      // First chunk or a gap (the engine fell behind) — start with a little headroom.
      const now = ctx.currentTime;
      if (nextStart < now + 0.005) nextStart = now + HEADROOM_S;
      source.start(nextStart);
      nextStart += buffer.duration;
      active.add(source);
      played.add(id);
    }
  } catch (e) {
    if ((e as Error)?.name !== 'AbortError') console.warn('[speech stream]', e);
  } finally {
    aborts.delete(abort);
  }
}
