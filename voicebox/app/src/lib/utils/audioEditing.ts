/** Operations on a sample in the editor: cutting, trimming, saving to WAV.
 *  All on an AudioBuffer in the browser — no server, no waiting. */

/** The crossfade when joining and the edges when trimming — without it a cut pops with a click. */
const FADE_S = 0.005;

function newBuffer(template: AudioBuffer, length: number): AudioBuffer {
  return new AudioBuffer({
    length: Math.max(1, length),
    numberOfChannels: template.numberOfChannels,
    sampleRate: template.sampleRate,
  });
}

/** Removes [start, end) and joins the rest with a short crossfade. */
export function cut(buf: AudioBuffer, startS: number, endS: number): AudioBuffer {
  const sr = buf.sampleRate;
  const s = Math.max(0, Math.min(buf.length, Math.round(startS * sr)));
  const e = Math.max(s, Math.min(buf.length, Math.round(endS * sr)));
  if (e <= s) return buf;
  // The crossfade comes from the material being CUT: [s, s+n) fades out, [e-n, e) fades in —
  // both edges stay continuous, and what remains loses not a single sample.
  const n = Math.min(Math.round(FADE_S * sr), Math.floor((e - s) / 2));
  const result = newBuffer(buf, buf.length - (e - s) + n);
  for (let k = 0; k < buf.numberOfChannels; k++) {
    const input = buf.getChannelData(k);
    const output = result.getChannelData(k);
    output.set(input.subarray(0, s), 0);
    for (let i = 0; i < n; i++) {
      const w = (i + 1) / (n + 1);
      output[s + i] = input[s + i] * (1 - w) + input[e - n + i] * w;
    }
    output.set(input.subarray(e), s + n);
  }
  return result;
}

/** Keeps only [start, end), with a gentle fade in and out. */
export function keep(buf: AudioBuffer, startS: number, endS: number): AudioBuffer {
  const sr = buf.sampleRate;
  const s = Math.max(0, Math.min(buf.length, Math.round(startS * sr)));
  const e = Math.max(s, Math.min(buf.length, Math.round(endS * sr)));
  if (e <= s) return buf;
  const result = newBuffer(buf, e - s);
  const n = Math.min(Math.round(FADE_S * sr), Math.floor((e - s) / 2));
  for (let k = 0; k < buf.numberOfChannels; k++) {
    const output = result.getChannelData(k);
    output.set(buf.getChannelData(k).subarray(s, e));
    for (let i = 0; i < n; i++) {
      const w = i / n;
      output[i] *= w;
      output[output.length - 1 - i] *= w;
    }
  }
  return result;
}

/** WAV float32 — same as the rest of Voicebox (Vox returns float32). */
export function toWav(buf: AudioBuffer): Blob {
  const channels = buf.numberOfChannels;
  const frames = buf.length;
  const dataSize = frames * channels * 4;
  const ab = new ArrayBuffer(44 + dataSize);
  const v = new DataView(ab);
  const text = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  text(0, 'RIFF');
  v.setUint32(4, 36 + dataSize, true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 3, true); // IEEE float
  v.setUint16(22, channels, true);
  v.setUint32(24, buf.sampleRate, true);
  v.setUint32(28, buf.sampleRate * channels * 4, true);
  v.setUint16(32, channels * 4, true);
  v.setUint16(34, 32, true);
  text(36, 'data');
  v.setUint32(40, dataSize, true);
  const data = Array.from({ length: channels }, (_, k) => buf.getChannelData(k));
  let o = 44;
  for (let i = 0; i < frames; i++) {
    for (let k = 0; k < channels; k++) {
      v.setFloat32(o, data[k][i], true);
      o += 4;
    }
  }
  return new Blob([ab], { type: 'audio/wav' });
}

export async function decode(file: Blob): Promise<AudioBuffer> {
  const ctx = new AudioContext();
  try {
    return await ctx.decodeAudioData(await file.arrayBuffer());
  } finally {
    void ctx.close();
  }
}
