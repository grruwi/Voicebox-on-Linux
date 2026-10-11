import { Crop, Loader2, Pause, Play, Redo2, Repeat, Scissors, Undo2 } from 'lucide-react';
import { type ReactNode, useCallback, useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/plugins/regions';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Slider } from '@/components/ui/slider';
import { apiClient } from '@/lib/api/client';
import { doWav, wytnij, zdekoduj, zostaw } from '@/lib/utils/edycjaAudio';

/** Edytor probki przy profilu (11.10).
 *  Etap 1: fala, zaznaczanie, ciecie, petla, A/B, cofanie — na AudioBuffer w przegladarce.
 *  Etap 2: filtry (esy, cykanie, gora, cisza) — ffmpeg na serwerze, warstwa NIEDESTRUKCYJNA:
 *  ciecie dziala na surowej probce, filtry licza sie na nowo na kazda zmiane (~0,1 s),
 *  a wszystko (ciecia i ruchy suwakow) siedzi w jednej historii Ctrl+Z / Ctrl+Y. */
interface EdytorProbkiProps {
  file: File | null | undefined;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onApply: (file: File) => void;
}

interface Filtry {
  deesser: { on: boolean; sila: number; wysokosc: number };
  declick: { on: boolean; czulosc: number };
  gora: { on: boolean; hz: number; db: number };
  bramka: { on: boolean; prog_db: number; sciszenie_db: number };
}

/** Wszystko wylaczone; suwaki startuja z pozycji, ktore zmierzylem na szepczacej probce voxa. */
const FILTRY_START: Filtry = {
  deesser: { on: false, sila: 0.7, wysokosc: 0.5 },
  declick: { on: false, czulosc: 0.5 },
  gora: { on: false, hz: 8000, db: -6 },
  bramka: { on: false, prog_db: -45, sciszenie_db: -20 },
};

const aktywne = (f: Filtry) => f.deesser.on || f.declick.on || f.gora.on || f.bramka.on;

interface Stan {
  buf: AudioBuffer;
  filtry: Filtry;
}

const KOLOR_ZAZNACZENIA = 'rgba(179, 154, 61, 0.28)';

const sek = (s: number) => `${s.toFixed(2)} s`;
const proc = (v: number) => `${Math.round(v * 100)}%`;

function Modul({
  tytul,
  opis,
  on,
  onToggle,
  children,
}: {
  tytul: string;
  opis: string;
  on: boolean;
  onToggle: (on: boolean) => void;
  children: ReactNode;
}) {
  return (
    <div className={`space-y-2 rounded-md border p-3 ${on ? 'border-primary/60 bg-primary/5' : ''}`}>
      <label className="flex cursor-pointer items-center gap-2 text-sm font-medium">
        <Checkbox checked={on} onCheckedChange={(v) => onToggle(v === true)} />
        {tytul}
      </label>
      <p className="text-xs text-muted-foreground">{opis}</p>
      {children}
    </div>
  );
}

function Suwak({
  etykieta,
  wartosc,
  pokaz,
  min,
  max,
  step,
  wylaczony,
  onZmiana,
  onKoniec,
}: {
  etykieta: string;
  wartosc: number;
  pokaz: string;
  min: number;
  max: number;
  step: number;
  wylaczony: boolean;
  onZmiana: (v: number) => void;
  onKoniec: (v: number) => void;
}) {
  return (
    <div className={`space-y-1 ${wylaczony ? 'opacity-50' : ''}`}>
      <div className="flex justify-between text-xs">
        <span>{etykieta}</span>
        <span className="tabular-nums text-muted-foreground">{pokaz}</span>
      </div>
      <Slider
        value={[wartosc]}
        min={min}
        max={max}
        step={step}
        disabled={wylaczony}
        onValueChange={([v]) => onZmiana(v)}
        onValueCommit={([v]) => onKoniec(v)}
      />
    </div>
  );
}

export function EdytorProbki({ file, open, onOpenChange, onApply }: EdytorProbkiProps) {
  const kontener = useRef<HTMLDivElement>(null);
  const ws = useRef<WaveSurfer | null>(null);
  const regiony = useRef<RegionsPlugin | null>(null);
  const url = useRef<string | null>(null);
  const [historia, setHistoria] = useState<Stan[]>([]);
  const [krok, setKrok] = useState(0);
  const [filtry, setFiltry] = useState<Filtry>(FILTRY_START);
  const [wynik, setWynik] = useState<{ dla: AudioBuffer; filtry: Filtry; buf: AudioBuffer } | null>(null);
  const [liczy, setLiczy] = useState(false);
  const [bladFiltra, setBladFiltra] = useState<string | null>(null);
  const [sluchamOryginalu, setSluchamOryginalu] = useState(false);
  const [petla, setPetla] = useState(false);
  const [gra, setGra] = useState(false);
  const [zazn, setZazn] = useState<{ start: number; end: number } | null>(null);
  const [zoom, setZoom] = useState(0);
  const [blad, setBlad] = useState<string | null>(null);
  const petlaRef = useRef(petla);
  petlaRef.current = petla;

  // Gotowe wersje po filtrach: ta sama probka + te same ustawienia = bez pytania serwera.
  const idBufora = useRef(new WeakMap<AudioBuffer, number>());
  const licznik = useRef(0);
  const pamiec = useRef(new Map<string, AudioBuffer>());
  const klucz = (buf: AudioBuffer, f: Filtry) => {
    let id = idBufora.current.get(buf);
    if (id === undefined) {
      id = ++licznik.current;
      idBufora.current.set(buf, id);
    }
    return `${id}|${JSON.stringify(f)}`;
  };
  const renderuj = async (buf: AudioBuffer, f: Filtry, sygnal?: AbortSignal) => {
    if (!aktywne(f)) return buf;
    const k = klucz(buf, f);
    const gotowy = pamiec.current.get(k);
    if (gotowy) return gotowy;
    const wav = await apiClient.filtrujProbke(doWav(buf), f, sygnal);
    const wy = await zdekoduj(wav);
    pamiec.current.set(k, wy);
    return wy;
  };

  const oryginal = historia[0]?.buf;
  const biezacy = historia[krok];
  const poFiltrach =
    biezacy && aktywne(filtry) && wynik && wynik.dla === biezacy.buf ? wynik.buf : biezacy?.buf;
  const pokazywany = sluchamOryginalu ? oryginal : poFiltrach;
  const zmieniona = krok > 0 || aktywne(filtry);

  // Plik → AudioBuffer przy kazdym otwarciu.
  useEffect(() => {
    if (!open || !file) return;
    let porzuc = false;
    setBlad(null);
    setBladFiltra(null);
    setHistoria([]);
    setKrok(0);
    setFiltry(FILTRY_START);
    setWynik(null);
    setSluchamOryginalu(false);
    setZazn(null);
    pamiec.current.clear();
    zdekoduj(file)
      .then((b) => !porzuc && setHistoria([{ buf: b, filtry: FILTRY_START }]))
      .catch((e) => !porzuc && setBlad(e instanceof Error ? e.message : String(e)));
    return () => {
      porzuc = true;
    };
  }, [open, file]);

  // Fala z zaznaczaniem — raz na otwarcie (kontener istnieje dopiero w otwartym oknie).
  useEffect(() => {
    if (!open || !oryginal || !kontener.current) return;
    const styl = getComputedStyle(document.documentElement);
    const css = (nazwa: string, zapas: string) => {
      const v = styl.getPropertyValue(nazwa).trim();
      return v ? `hsl(${v})` : zapas;
    };
    const plugin = RegionsPlugin.create();
    const w = WaveSurfer.create({
      container: kontener.current,
      height: 150,
      waveColor: css('--muted-foreground', '#888'),
      progressColor: css('--primary', '#b39a3d'),
      cursorColor: css('--foreground', '#fff'),
      cursorWidth: 2,
      plugins: [plugin],
    });
    plugin.enableDragSelection({ color: KOLOR_ZAZNACZENIA });
    plugin.on('region-created', (r) => {
      for (const inny of plugin.getRegions()) if (inny !== r) inny.remove();
      setZazn({ start: r.start, end: r.end });
    });
    plugin.on('region-updated', (r) => setZazn({ start: r.start, end: r.end }));
    plugin.on('region-out', (r) => {
      if (petlaRef.current) r.play();
    });
    w.on('play', () => setGra(true));
    w.on('pause', () => setGra(false));
    w.on('finish', () => {
      if (petlaRef.current && plugin.getRegions().length === 0) {
        w.setTime(0);
        void w.play();
      }
    });
    ws.current = w;
    regiony.current = plugin;
    return () => {
      w.destroy();
      ws.current = null;
      regiony.current = null;
      setGra(false);
      if (url.current) URL.revokeObjectURL(url.current);
      url.current = null;
    };
  }, [open, oryginal]);

  // Filtry licza sie po kazdej zmianie (z oddechem 120 ms na przeciaganie suwaka).
  useEffect(() => {
    if (!biezacy || !aktywne(filtry)) {
      setLiczy(false);
      return;
    }
    const ac = new AbortController();
    const t = setTimeout(() => {
      setLiczy(true);
      renderuj(biezacy.buf, filtry, ac.signal)
        .then((b) => {
          if (ac.signal.aborted) return;
          setWynik({ dla: biezacy.buf, filtry, buf: b });
          setBladFiltra(null);
        })
        .catch((e) => {
          if (!ac.signal.aborted) setBladFiltra(e instanceof Error ? e.message : String(e));
        })
        .finally(() => {
          if (!ac.signal.aborted) setLiczy(false);
        });
    }, 120);
    return () => {
      clearTimeout(t);
      ac.abort();
    };
    // renderuj czyta tylko refy
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [biezacy, filtry]);

  // Kazda wersja (edycja, filtr, cofniecie, A/B) laduje sie od razu; odtwarzanie leci dalej z tego miejsca.
  useEffect(() => {
    const w = ws.current;
    if (!w || !pokazywany) return;
    const t = w.getCurrentTime();
    const gralo = w.isPlaying();
    const nowy = URL.createObjectURL(doWav(pokazywany));
    const stary = url.current;
    url.current = nowy;
    w.load(nowy, [pokazywany.getChannelData(0)], pokazywany.duration)
      .then(() => {
        if (stary) URL.revokeObjectURL(stary);
        w.setTime(Math.min(t, pokazywany.duration));
        if (zoom > 0) w.zoom(zoom);
        if (gralo) void w.play();
      })
      .catch(() => {});
    // zoom celowo poza zaleznosciami — ma wlasny efekt nizej
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pokazywany]);

  useEffect(() => {
    try {
      ws.current?.zoom(zoom);
    } catch {
      // fala jeszcze sie nie zaladowala
    }
  }, [zoom]);

  const odznacz = useCallback(() => {
    regiony.current?.clearRegions();
    setZazn(null);
  }, []);

  const dopisz = (stan: Stan) => {
    setHistoria((h) => [...h.slice(0, krok + 1), stan]);
    setKrok(krok + 1);
  };

  /** zapisz=false: suwak w ruchu (slychac od razu); zapisz=true: puszczony — wpis do historii. */
  const zmienFiltry = (nowe: Filtry, zapisz: boolean) => {
    setFiltry(nowe);
    if (zapisz && biezacy) dopisz({ buf: biezacy.buf, filtry: nowe });
  };
  const modul = <K extends keyof Filtry>(nazwa: K, zmiana: Partial<Filtry[K]>, zapisz: boolean) =>
    zmienFiltry({ ...filtry, [nazwa]: { ...filtry[nazwa], ...zmiana } }, zapisz);

  const edytuj = (operacja: (b: AudioBuffer, s: number, e: number) => AudioBuffer) => {
    if (!zazn || !biezacy || sluchamOryginalu) return;
    dopisz({ buf: operacja(biezacy.buf, zazn.start, zazn.end), filtry });
    odznacz();
  };
  const idzDo = (k: number) => {
    if (k < 0 || k >= historia.length || k === krok) return;
    setKrok(k);
    setFiltry(historia[k].filtry);
    odznacz();
  };
  const cofnij = () => idzDo(krok - 1);
  const ponow = () => idzDo(krok + 1);
  const grajPauza = () => {
    const w = ws.current;
    if (!w) return;
    if (w.isPlaying()) {
      w.pause();
      return;
    }
    const r = regiony.current?.getRegions()[0];
    if (r) r.play(!petlaRef.current);
    else void w.play();
  };

  // Klawiatura: spacja graj, Delete wytnij, K zostaw, B oryginal/wersja, L petla, Ctrl+Z/Y.
  useEffect(() => {
    if (!open) return;
    const klawisz = (e: KeyboardEvent) => {
      const cel = e.target as HTMLElement | null;
      if (cel && (cel.tagName === 'INPUT' || cel.tagName === 'TEXTAREA' || cel.isContentEditable)) return;
      const k = e.key.toLowerCase();
      if (e.ctrlKey && k === 'z' && !e.shiftKey) {
        e.preventDefault();
        cofnij();
      } else if (e.ctrlKey && (k === 'y' || (k === 'z' && e.shiftKey))) {
        e.preventDefault();
        ponow();
      } else if (e.ctrlKey || e.altKey || e.metaKey) {
        return;
      } else if (k === ' ') {
        e.preventDefault();
        // spacja na guziku/suwaku z fokusem kliknelaby go drugi raz
        cel?.blur?.();
        grajPauza();
      } else if (k === 'delete' || k === 'backspace') {
        e.preventDefault();
        edytuj(wytnij);
      } else if (k === 'k') {
        edytuj(zostaw);
      } else if (k === 'b') {
        setSluchamOryginalu((v) => !v);
      } else if (k === 'l') {
        setPetla((v) => !v);
      }
    };
    window.addEventListener('keydown', klawisz);
    return () => window.removeEventListener('keydown', klawisz);
  });

  const zastosuj = async () => {
    if (!biezacy || !file || !zmieniona) {
      onOpenChange(false);
      return;
    }
    setLiczy(true);
    try {
      const gotowy = await renderuj(biezacy.buf, filtry);
      const nazwa = file.name.replace(/\.[^.]+$/, '') || 'probka';
      onApply(new File([doWav(gotowy)], `${nazwa}-edyt.wav`, { type: 'audio/wav' }));
      onOpenChange(false);
    } catch (e) {
      setBladFiltra(e instanceof Error ? e.message : String(e));
    } finally {
      setLiczy(false);
    }
  };

  const edycjaZablokowana = !zazn || sluchamOryginalu;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="max-h-[92vh] max-w-5xl overflow-y-auto"
        onEscapeKeyDown={(e) => {
          if (zazn) {
            e.preventDefault();
            odznacz();
          }
        }}
        onInteractOutside={(e) => {
          if (zmieniona) e.preventDefault();
        }}
      >
        <DialogHeader>
          <DialogTitle>Edytor próbki</DialogTitle>
          <DialogDescription>
            Przeciągnij po fali, żeby zaznaczyć. Spacja — graj · Delete — wytnij · K — zostaw tylko
            zaznaczone · B — oryginał/wersja · L — pętla · Ctrl+Z / Ctrl+Y (cięcia i filtry) · Esc —
            odznacz
          </DialogDescription>
        </DialogHeader>

        {blad ? (
          <p className="text-sm text-destructive">Nie umiem odczytać tej próbki: {blad}</p>
        ) : (
          <div className="space-y-3">
            <div
              ref={kontener}
              className={`rounded-md border bg-muted/30 ${sluchamOryginalu ? 'ring-2 ring-amber-500' : ''}`}
            />
            <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-muted-foreground">
              <span className="flex items-center gap-2">
                {sluchamOryginalu ? (
                  <b className="text-amber-500">SŁUCHASZ ORYGINAŁU</b>
                ) : (
                  <>
                    Długość: {biezacy ? sek(biezacy.buf.duration) : '…'}
                    {oryginal && krok > 0 && ` (oryginał ${sek(oryginal.duration)})`}
                  </>
                )}
                {liczy && <Loader2 className="h-4 w-4 animate-spin" />}
              </span>
              <span>
                {zazn
                  ? `Zaznaczone: ${sek(zazn.start)} – ${sek(zazn.end)} (${sek(zazn.end - zazn.start)})`
                  : 'Nic nie zaznaczone'}
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button type="button" size="icon" variant="outline" onClick={grajPauza} title="Graj / pauza (spacja)">
                {gra ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <Button
                type="button"
                variant={petla ? 'default' : 'outline'}
                onClick={() => setPetla((v) => !v)}
                title="Pętla (L) — zaznaczenia albo całości"
                className="gap-2"
              >
                <Repeat className="h-4 w-4" /> Pętla
              </Button>
              <Button
                type="button"
                variant={sluchamOryginalu ? 'default' : 'outline'}
                onClick={() => setSluchamOryginalu((v) => !v)}
                disabled={!zmieniona}
                title="Oryginał / wersja (B)"
              >
                A/B
              </Button>
              <div className="mx-1 h-6 w-px bg-border" />
              <Button
                type="button"
                variant="outline"
                onClick={() => edytuj(wytnij)}
                disabled={edycjaZablokowana}
                title="Wytnij zaznaczone (Delete)"
                className="gap-2"
              >
                <Scissors className="h-4 w-4" /> Wytnij
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => edytuj(zostaw)}
                disabled={edycjaZablokowana}
                title="Zostaw tylko zaznaczone (K)"
                className="gap-2"
              >
                <Crop className="h-4 w-4" /> Zostaw zaznaczone
              </Button>
              <div className="mx-1 h-6 w-px bg-border" />
              <Button
                type="button"
                size="icon"
                variant="outline"
                onClick={cofnij}
                disabled={krok === 0}
                title="Cofnij (Ctrl+Z)"
              >
                <Undo2 className="h-4 w-4" />
              </Button>
              <Button
                type="button"
                size="icon"
                variant="outline"
                onClick={ponow}
                disabled={krok >= historia.length - 1}
                title="Ponów (Ctrl+Y)"
              >
                <Redo2 className="h-4 w-4" />
              </Button>
              <div className="ml-auto flex w-48 items-center gap-2 text-xs text-muted-foreground">
                Zoom
                <Slider value={[zoom]} min={0} max={500} step={10} onValueChange={([v]) => setZoom(v)} />
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <Modul
                tytul="Esy (de-esser)"
                opis="Ścisza ostre s / sz / cz tylko w chwili, gdy lecą — reszta głosu zostaje."
                on={filtry.deesser.on}
                onToggle={(on) => modul('deesser', { on }, true)}
              >
                <Suwak
                  etykieta="Siła"
                  wartosc={filtry.deesser.sila}
                  pokaz={proc(filtry.deesser.sila)}
                  min={0}
                  max={1}
                  step={0.01}
                  wylaczony={!filtry.deesser.on}
                  onZmiana={(v) => modul('deesser', { sila: v }, false)}
                  onKoniec={(v) => modul('deesser', { sila: v }, true)}
                />
                <Suwak
                  etykieta="Wysokość (niżej łapie też sz/cz, wyżej tylko ostre s)"
                  wartosc={filtry.deesser.wysokosc}
                  pokaz={proc(filtry.deesser.wysokosc)}
                  min={0}
                  max={1}
                  step={0.01}
                  wylaczony={!filtry.deesser.on}
                  onZmiana={(v) => modul('deesser', { wysokosc: v }, false)}
                  onKoniec={(v) => modul('deesser', { wysokosc: v }, true)}
                />
              </Modul>

              <Modul
                tytul="Cyknięcia i mlaski (de-click)"
                opis="Wyłapuje krótkie trzaski i łata je dźwiękiem z otoczenia."
                on={filtry.declick.on}
                onToggle={(on) => modul('declick', { on }, true)}
              >
                <Suwak
                  etykieta="Czułość"
                  wartosc={filtry.declick.czulosc}
                  pokaz={proc(filtry.declick.czulosc)}
                  min={0}
                  max={1}
                  step={0.01}
                  wylaczony={!filtry.declick.on}
                  onZmiana={(v) => modul('declick', { czulosc: v }, false)}
                  onKoniec={(v) => modul('declick', { czulosc: v }, true)}
                />
              </Modul>

              <Modul
                tytul="Przyciemnij górę (półka)"
                opis="Łagodnie ścisza wszystko powyżej progu — na syk i ostrość w całym nagraniu."
                on={filtry.gora.on}
                onToggle={(on) => modul('gora', { on }, true)}
              >
                <Suwak
                  etykieta="Od"
                  wartosc={filtry.gora.hz}
                  pokaz={`${(filtry.gora.hz / 1000).toFixed(1)} kHz`}
                  min={3000}
                  max={16000}
                  step={100}
                  wylaczony={!filtry.gora.on}
                  onZmiana={(v) => modul('gora', { hz: v }, false)}
                  onKoniec={(v) => modul('gora', { hz: v }, true)}
                />
                <Suwak
                  etykieta="Ile"
                  wartosc={filtry.gora.db}
                  pokaz={`${filtry.gora.db.toFixed(1)} dB`}
                  min={-24}
                  max={0}
                  step={0.5}
                  wylaczony={!filtry.gora.on}
                  onZmiana={(v) => modul('gora', { db: v }, false)}
                  onKoniec={(v) => modul('gora', { db: v }, true)}
                />
              </Modul>

              <Modul
                tytul="Cisza między słowami (bramka)"
                opis="Co cichsze od progu, zostaje ściszone — oddechy i szum w przerwach."
                on={filtry.bramka.on}
                onToggle={(on) => modul('bramka', { on }, true)}
              >
                <Suwak
                  etykieta="Próg"
                  wartosc={filtry.bramka.prog_db}
                  pokaz={`${filtry.bramka.prog_db} dB`}
                  min={-80}
                  max={-10}
                  step={1}
                  wylaczony={!filtry.bramka.on}
                  onZmiana={(v) => modul('bramka', { prog_db: v }, false)}
                  onKoniec={(v) => modul('bramka', { prog_db: v }, true)}
                />
                <Suwak
                  etykieta="Ściszenie"
                  wartosc={filtry.bramka.sciszenie_db}
                  pokaz={`${filtry.bramka.sciszenie_db} dB`}
                  min={-60}
                  max={0}
                  step={1}
                  wylaczony={!filtry.bramka.on}
                  onZmiana={(v) => modul('bramka', { sciszenie_db: v }, false)}
                  onKoniec={(v) => modul('bramka', { sciszenie_db: v }, true)}
                />
              </Modul>
            </div>
            {bladFiltra && <p className="text-sm text-destructive">Filtry: {bladFiltra}</p>}
          </div>
        )}

        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
            Anuluj
          </Button>
          <Button type="button" onClick={() => void zastosuj()} disabled={!biezacy || liczy}>
            {zmieniona ? 'Zastosuj' : 'Zamknij'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
