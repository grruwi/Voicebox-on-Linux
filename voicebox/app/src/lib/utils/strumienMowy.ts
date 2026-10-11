// Odtwarzacz strumienia mowy z backendu (`GET /generate/{id}/strumien`, 11.10).
//
// „Autoplay on generate" zaznaczony = backend nie gra przez ffplay, tylko oddaje tu kawalki
// w miare liczenia: 4 bajty czestotliwosci (uint32 LE), dalej surowe f32le mono. Kazdy kawalek
// idzie od razu w Web Audio, zaplanowany tuz za poprzednim — pierwszy dzwiek po TTFT silnika
// + kilkadziesiat ms. Kolejne generacje ustawiaja sie w kolejce za soba, nie przerywaja.
//
// 204 = ta generacja nie leci strumieniem (efekty, retry, za dlugi tekst) — wtedy panel gra
// plik po calosci jak dawniej.

// Zapas na starcie i po dziurze. Mniej = szybciej, ale kazde opoznienie sieci slychac jako trzask.
const ZAPAS_S = 0.04;

let kontekst: AudioContext | null = null;
let nastepnyStart = 0;
const aktywne = new Set<AudioBufferSourceNode>();
const przerwania = new Set<AbortController>();
// Generacje, z ktorych zagral choc jeden kawalek — tych panel NIE gra drugi raz z pliku.
const zagrane = new Set<string>();

function audio(): AudioContext {
  if (!kontekst) kontekst = new AudioContext({ latencyHint: 'interactive' });
  // Przegladarka trzyma kontekst uspiony do pierwszej interakcji ze strona; klikniecie
  // „Generate" jest taka interakcja, wiec tu juz wstaje.
  if (kontekst.state === 'suspended') void kontekst.resume();
  return kontekst;
}

export function zagranoStrumieniem(id: string): boolean {
  return zagrane.has(id);
}

export function zatrzymajStrumienie(): void {
  for (const p of przerwania) p.abort();
  przerwania.clear();
  for (const z of aktywne) {
    try {
      z.stop();
    } catch {
      // juz skonczone
    }
  }
  aktywne.clear();
  nastepnyStart = 0;
}

export async function grajStrumien(url: string, id: string): Promise<void> {
  const ctx = audio();
  const przerwanie = new AbortController();
  przerwania.add(przerwanie);
  try {
    const odp = await fetch(url, { signal: przerwanie.signal, cache: 'no-store' });
    if (odp.status !== 200 || !odp.body) return;

    const czytnik = odp.body.getReader();
    let czestotliwosc = 0;
    let reszta = new Uint8Array(0);

    for (;;) {
      const { value, done } = await czytnik.read();
      if (done) break;
      if (!value?.length) continue;

      let dane: Uint8Array;
      if (reszta.length) {
        dane = new Uint8Array(reszta.length + value.length);
        dane.set(reszta);
        dane.set(value, reszta.length);
      } else {
        dane = value;
      }

      if (!czestotliwosc) {
        if (dane.length < 4) {
          reszta = dane.slice();
          continue;
        }
        czestotliwosc = new DataView(dane.buffer, dane.byteOffset, 4).getUint32(0, true);
        dane = dane.subarray(4);
      }

      const ile = Math.floor(dane.length / 4);
      reszta = dane.slice(ile * 4);
      if (!ile) continue;

      // kopia — Float32Array wymaga wyrownania do 4 bajtow, a kawalek z sieci go nie gwarantuje
      const probki = new Float32Array(dane.slice(0, ile * 4).buffer);
      const bufor = ctx.createBuffer(1, ile, czestotliwosc);
      bufor.copyToChannel(probki, 0);

      const zrodlo = ctx.createBufferSource();
      zrodlo.buffer = bufor;
      zrodlo.connect(ctx.destination);
      zrodlo.onended = () => aktywne.delete(zrodlo);

      // Pierwszy kawalek albo dziura (silnik nie nadazyl) — startujemy z malym zapasem.
      const teraz = ctx.currentTime;
      if (nastepnyStart < teraz + 0.005) nastepnyStart = teraz + ZAPAS_S;
      zrodlo.start(nastepnyStart);
      nastepnyStart += bufor.duration;
      aktywne.add(zrodlo);
      zagrane.add(id);
    }
  } catch (e) {
    if ((e as Error)?.name !== 'AbortError') console.warn('[strumien mowy]', e);
  } finally {
    przerwania.delete(przerwanie);
  }
}
