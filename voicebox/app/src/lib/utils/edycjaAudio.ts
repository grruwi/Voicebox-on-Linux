/** Operacje na probce w edytorze (11.10): ciecie, kadrowanie, zapis do WAV.
 *  Wszystko na AudioBuffer w przegladarce — zero serwera, zero czekania. */

/** Przejscie przy sklejaniu i brzegi przy kadrowaniu — bez tego ciecie strzela klikiem. */
const PRZEJSCIE_S = 0.005;

function nowyBufor(wzor: AudioBuffer, dlugosc: number): AudioBuffer {
  return new AudioBuffer({
    length: Math.max(1, dlugosc),
    numberOfChannels: wzor.numberOfChannels,
    sampleRate: wzor.sampleRate,
  });
}

/** Wywala [start, end) i skleja reszte z krotkim przejsciem (crossfade). */
export function wytnij(buf: AudioBuffer, startS: number, endS: number): AudioBuffer {
  const sr = buf.sampleRate;
  const s = Math.max(0, Math.min(buf.length, Math.round(startS * sr)));
  const e = Math.max(s, Math.min(buf.length, Math.round(endS * sr)));
  if (e <= s) return buf;
  // Przejscie z materialu WYCINANEGO: [s, s+n) gasnie, [e-n, e) wschodzi — oba brzegi
  // zostaja ciagle, a to, co zostaje, nie traci ani probki.
  const n = Math.min(Math.round(PRZEJSCIE_S * sr), Math.floor((e - s) / 2));
  const wynik = nowyBufor(buf, buf.length - (e - s) + n);
  for (let k = 0; k < buf.numberOfChannels; k++) {
    const we = buf.getChannelData(k);
    const wy = wynik.getChannelData(k);
    wy.set(we.subarray(0, s), 0);
    for (let i = 0; i < n; i++) {
      const w = (i + 1) / (n + 1);
      wy[s + i] = we[s + i] * (1 - w) + we[e - n + i] * w;
    }
    wy.set(we.subarray(e), s + n);
  }
  return wynik;
}

/** Zostawia tylko [start, end), z lagodnym wejsciem i wyjsciem. */
export function zostaw(buf: AudioBuffer, startS: number, endS: number): AudioBuffer {
  const sr = buf.sampleRate;
  const s = Math.max(0, Math.min(buf.length, Math.round(startS * sr)));
  const e = Math.max(s, Math.min(buf.length, Math.round(endS * sr)));
  if (e <= s) return buf;
  const wynik = nowyBufor(buf, e - s);
  const n = Math.min(Math.round(PRZEJSCIE_S * sr), Math.floor((e - s) / 2));
  for (let k = 0; k < buf.numberOfChannels; k++) {
    const wy = wynik.getChannelData(k);
    wy.set(buf.getChannelData(k).subarray(s, e));
    for (let i = 0; i < n; i++) {
      const w = i / n;
      wy[i] *= w;
      wy[wy.length - 1 - i] *= w;
    }
  }
  return wynik;
}

/** WAV float32 — tak samo jak reszta Voiceboxa (vox oddaje float32). */
export function doWav(buf: AudioBuffer): Blob {
  const kanaly = buf.numberOfChannels;
  const ramki = buf.length;
  const dane = ramki * kanaly * 4;
  const ab = new ArrayBuffer(44 + dane);
  const v = new DataView(ab);
  const tekst = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  tekst(0, 'RIFF');
  v.setUint32(4, 36 + dane, true);
  tekst(8, 'WAVE');
  tekst(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 3, true); // IEEE float
  v.setUint16(22, kanaly, true);
  v.setUint32(24, buf.sampleRate, true);
  v.setUint32(28, buf.sampleRate * kanaly * 4, true);
  v.setUint16(32, kanaly * 4, true);
  v.setUint16(34, 32, true);
  tekst(36, 'data');
  v.setUint32(40, dane, true);
  const kan = Array.from({ length: kanaly }, (_, k) => buf.getChannelData(k));
  let o = 44;
  for (let i = 0; i < ramki; i++) {
    for (let k = 0; k < kanaly; k++) {
      v.setFloat32(o, kan[k][i], true);
      o += 4;
    }
  }
  return new Blob([ab], { type: 'audio/wav' });
}

export async function zdekoduj(plik: Blob): Promise<AudioBuffer> {
  const ctx = new AudioContext();
  try {
    return await ctx.decodeAudioData(await plik.arrayBuffer());
  } finally {
    void ctx.close();
  }
}
