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
import { cut, decode, keep, toWav } from '@/lib/utils/audioEditing';

/** The sample editor of a profile.
 *  Stage 1: waveform, selection, cutting, loop, A/B, undo — on an AudioBuffer in the browser.
 *  Stage 2: filters (sibilance, clicks, treble, silence) — ffmpeg on the server, a
 *  NON-DESTRUCTIVE layer: cuts work on the raw sample, the filters are recomputed on every
 *  change (~0.1 s), and everything (cuts and slider moves) lives in one Ctrl+Z / Ctrl+Y history. */
interface SampleEditorProps {
  file: File | null | undefined;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onApply: (file: File) => void;
}

interface Filters {
  deesser: { on: boolean; strength: number; frequency: number };
  declick: { on: boolean; sensitivity: number };
  treble: { on: boolean; hz: number; db: number };
  gate: { on: boolean; threshold_db: number; reduction_db: number };
}

/** Everything off; the sliders start where they were measured on a whispering Vox sample. */
const FILTERS_START: Filters = {
  deesser: { on: false, strength: 0.7, frequency: 0.5 },
  declick: { on: false, sensitivity: 0.5 },
  treble: { on: false, hz: 8000, db: -6 },
  gate: { on: false, threshold_db: -45, reduction_db: -20 },
};

const anyOn = (f: Filters) => f.deesser.on || f.declick.on || f.treble.on || f.gate.on;

interface Version {
  buf: AudioBuffer;
  filters: Filters;
}

const SELECTION_COLOR = 'rgba(179, 154, 61, 0.28)';

const secs = (s: number) => `${s.toFixed(2)} s`;
const pct = (v: number) => `${Math.round(v * 100)}%`;

function Module({
  title,
  description,
  on,
  onToggle,
  children,
}: {
  title: string;
  description: string;
  on: boolean;
  onToggle: (on: boolean) => void;
  children: ReactNode;
}) {
  return (
    <div className={`space-y-2 rounded-md border p-3 ${on ? 'border-primary/60 bg-primary/5' : ''}`}>
      <label className="flex cursor-pointer items-center gap-2 text-sm font-medium">
        <Checkbox checked={on} onCheckedChange={(v) => onToggle(v === true)} />
        {title}
      </label>
      <p className="text-xs text-muted-foreground">{description}</p>
      {children}
    </div>
  );
}

function FilterSlider({
  label,
  value,
  display,
  min,
  max,
  step,
  disabled,
  onChange,
  onCommit,
}: {
  label: string;
  value: number;
  display: string;
  min: number;
  max: number;
  step: number;
  disabled: boolean;
  onChange: (v: number) => void;
  onCommit: (v: number) => void;
}) {
  return (
    <div className={`space-y-1 ${disabled ? 'opacity-50' : ''}`}>
      <div className="flex justify-between text-xs">
        <span>{label}</span>
        <span className="tabular-nums text-muted-foreground">{display}</span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onValueChange={([v]) => onChange(v)}
        onValueCommit={([v]) => onCommit(v)}
      />
    </div>
  );
}

export function SampleEditor({ file, open, onOpenChange, onApply }: SampleEditorProps) {
  const container = useRef<HTMLDivElement>(null);
  const ws = useRef<WaveSurfer | null>(null);
  const regions = useRef<RegionsPlugin | null>(null);
  const url = useRef<string | null>(null);
  const [history, setHistory] = useState<Version[]>([]);
  const [step, setStep] = useState(0);
  const [filters, setFilters] = useState<Filters>(FILTERS_START);
  const [result, setResult] = useState<{ source: AudioBuffer; filters: Filters; buf: AudioBuffer } | null>(null);
  const [busy, setBusy] = useState(false);
  const [filterError, setFilterError] = useState<string | null>(null);
  const [hearingOriginal, setHearingOriginal] = useState(false);
  const [loop, setLoop] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [selection, setSelection] = useState<{ start: number; end: number } | null>(null);
  const [zoom, setZoom] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const loopRef = useRef(loop);
  loopRef.current = loop;

  // Filtered versions ready to use: the same sample + the same settings = no server round trip.
  const bufferIds = useRef(new WeakMap<AudioBuffer, number>());
  const counter = useRef(0);
  const cache = useRef(new Map<string, AudioBuffer>());
  const cacheKey = (buf: AudioBuffer, f: Filters) => {
    let id = bufferIds.current.get(buf);
    if (id === undefined) {
      id = ++counter.current;
      bufferIds.current.set(buf, id);
    }
    return `${id}|${JSON.stringify(f)}`;
  };
  const render = async (buf: AudioBuffer, f: Filters, signal?: AbortSignal) => {
    if (!anyOn(f)) return buf;
    const k = cacheKey(buf, f);
    const ready = cache.current.get(k);
    if (ready) return ready;
    const wav = await apiClient.filterSample(toWav(buf), f, signal);
    const out = await decode(wav);
    cache.current.set(k, out);
    return out;
  };

  const original = history[0]?.buf;
  const current = history[step];
  const filtered =
    current && anyOn(filters) && result && result.source === current.buf ? result.buf : current?.buf;
  const shown = hearingOriginal ? original : filtered;
  const changed = step > 0 || anyOn(filters);

  // File → AudioBuffer on every open.
  useEffect(() => {
    if (!open || !file) return;
    let abandoned = false;
    setError(null);
    setFilterError(null);
    setHistory([]);
    setStep(0);
    setFilters(FILTERS_START);
    setResult(null);
    setHearingOriginal(false);
    setSelection(null);
    cache.current.clear();
    decode(file)
      .then((b) => !abandoned && setHistory([{ buf: b, filters: FILTERS_START }]))
      .catch((e) => !abandoned && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      abandoned = true;
    };
  }, [open, file]);

  // The waveform with selection — once per open (the container exists only in the open dialog).
  useEffect(() => {
    if (!open || !original || !container.current) return;
    const style = getComputedStyle(document.documentElement);
    const css = (name: string, fallback: string) => {
      const v = style.getPropertyValue(name).trim();
      return v ? `hsl(${v})` : fallback;
    };
    const plugin = RegionsPlugin.create();
    const w = WaveSurfer.create({
      container: container.current,
      height: 150,
      waveColor: css('--muted-foreground', '#888'),
      progressColor: css('--primary', '#b39a3d'),
      cursorColor: css('--foreground', '#fff'),
      cursorWidth: 2,
      plugins: [plugin],
    });
    plugin.enableDragSelection({ color: SELECTION_COLOR });
    plugin.on('region-created', (r) => {
      for (const other of plugin.getRegions()) if (other !== r) other.remove();
      setSelection({ start: r.start, end: r.end });
    });
    plugin.on('region-updated', (r) => setSelection({ start: r.start, end: r.end }));
    plugin.on('region-out', (r) => {
      if (loopRef.current) r.play();
    });
    w.on('play', () => setPlaying(true));
    w.on('pause', () => setPlaying(false));
    w.on('finish', () => {
      if (loopRef.current && plugin.getRegions().length === 0) {
        w.setTime(0);
        void w.play();
      }
    });
    ws.current = w;
    regions.current = plugin;
    return () => {
      w.destroy();
      ws.current = null;
      regions.current = null;
      setPlaying(false);
      if (url.current) URL.revokeObjectURL(url.current);
      url.current = null;
    };
  }, [open, original]);

  // The filters are recomputed after every change (with a 120 ms breather for slider drags).
  useEffect(() => {
    if (!current || !anyOn(filters)) {
      setBusy(false);
      return;
    }
    const ac = new AbortController();
    const t = setTimeout(() => {
      setBusy(true);
      render(current.buf, filters, ac.signal)
        .then((b) => {
          if (ac.signal.aborted) return;
          setResult({ source: current.buf, filters, buf: b });
          setFilterError(null);
        })
        .catch((e) => {
          if (!ac.signal.aborted) setFilterError(e instanceof Error ? e.message : String(e));
        })
        .finally(() => {
          if (!ac.signal.aborted) setBusy(false);
        });
    }, 120);
    return () => {
      clearTimeout(t);
      ac.abort();
    };
    // render only reads refs
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current, filters]);

  // Every version (edit, filter, undo, A/B) loads at once; playback carries on from the same spot.
  useEffect(() => {
    const w = ws.current;
    if (!w || !shown) return;
    const t = w.getCurrentTime();
    const wasPlaying = w.isPlaying();
    const fresh = URL.createObjectURL(toWav(shown));
    const old = url.current;
    url.current = fresh;
    w.load(fresh, [shown.getChannelData(0)], shown.duration)
      .then(() => {
        if (old) URL.revokeObjectURL(old);
        w.setTime(Math.min(t, shown.duration));
        if (zoom > 0) w.zoom(zoom);
        if (wasPlaying) void w.play();
      })
      .catch(() => {});
    // zoom deliberately left out of the dependencies — it has its own effect below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shown]);

  useEffect(() => {
    try {
      ws.current?.zoom(zoom);
    } catch {
      // the waveform has not loaded yet
    }
  }, [zoom]);

  const deselect = useCallback(() => {
    regions.current?.clearRegions();
    setSelection(null);
  }, []);

  const push = (version: Version) => {
    setHistory((h) => [...h.slice(0, step + 1), version]);
    setStep(step + 1);
  };

  /** save=false: a slider is moving (heard at once); save=true: released — a history entry. */
  const changeFilters = (next: Filters, save: boolean) => {
    setFilters(next);
    if (save && current) push({ buf: current.buf, filters: next });
  };
  const updateModule = <K extends keyof Filters>(name: K, change: Partial<Filters[K]>, save: boolean) =>
    changeFilters({ ...filters, [name]: { ...filters[name], ...change } }, save);

  const edit = (operation: (b: AudioBuffer, s: number, e: number) => AudioBuffer) => {
    if (!selection || !current || hearingOriginal) return;
    push({ buf: operation(current.buf, selection.start, selection.end), filters });
    deselect();
  };
  const goTo = (k: number) => {
    if (k < 0 || k >= history.length || k === step) return;
    setStep(k);
    setFilters(history[k].filters);
    deselect();
  };
  const undo = () => goTo(step - 1);
  const redo = () => goTo(step + 1);
  const playPause = () => {
    const w = ws.current;
    if (!w) return;
    if (w.isPlaying()) {
      w.pause();
      return;
    }
    const r = regions.current?.getRegions()[0];
    if (r) r.play(!loopRef.current);
    else void w.play();
  };

  // Keyboard: space play, Delete cut, K keep, B original/version, L loop, Ctrl+Z/Y.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return;
      const k = e.key.toLowerCase();
      if (e.ctrlKey && k === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if (e.ctrlKey && (k === 'y' || (k === 'z' && e.shiftKey))) {
        e.preventDefault();
        redo();
      } else if (e.ctrlKey || e.altKey || e.metaKey) {
        return;
      } else if (k === ' ') {
        e.preventDefault();
        // space on a focused button/slider would click it a second time
        target?.blur?.();
        playPause();
      } else if (k === 'delete' || k === 'backspace') {
        e.preventDefault();
        edit(cut);
      } else if (k === 'k') {
        edit(keep);
      } else if (k === 'b') {
        setHearingOriginal((v) => !v);
      } else if (k === 'l') {
        setLoop((v) => !v);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const apply = async () => {
    if (!current || !file || !changed) {
      onOpenChange(false);
      return;
    }
    setBusy(true);
    try {
      const ready = await render(current.buf, filters);
      const name = file.name.replace(/\.[^.]+$/, '') || 'sample';
      onApply(new File([toWav(ready)], `${name}-edited.wav`, { type: 'audio/wav' }));
      onOpenChange(false);
    } catch (e) {
      setFilterError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const editingBlocked = !selection || hearingOriginal;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="max-h-[92vh] max-w-5xl overflow-y-auto"
        onEscapeKeyDown={(e) => {
          if (selection) {
            e.preventDefault();
            deselect();
          }
        }}
        onInteractOutside={(e) => {
          if (changed) e.preventDefault();
        }}
      >
        <DialogHeader>
          <DialogTitle>Sample editor</DialogTitle>
          <DialogDescription>
            Drag over the waveform to select. Space — play · Delete — cut · K — keep only the
            selection · B — original/version · L — loop · Ctrl+Z / Ctrl+Y (cuts and filters) · Esc —
            deselect
          </DialogDescription>
        </DialogHeader>

        {error ? (
          <p className="text-sm text-destructive">Cannot read this sample: {error}</p>
        ) : (
          <div className="space-y-3">
            <div
              ref={container}
              className={`rounded-md border bg-muted/30 ${hearingOriginal ? 'ring-2 ring-amber-500' : ''}`}
            />
            <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-muted-foreground">
              <span className="flex items-center gap-2">
                {hearingOriginal ? (
                  <b className="text-amber-500">HEARING THE ORIGINAL</b>
                ) : (
                  <>
                    Length: {current ? secs(current.buf.duration) : '…'}
                    {original && step > 0 && ` (original ${secs(original.duration)})`}
                  </>
                )}
                {busy && <Loader2 className="h-4 w-4 animate-spin" />}
              </span>
              <span>
                {selection
                  ? `Selected: ${secs(selection.start)} – ${secs(selection.end)} (${secs(selection.end - selection.start)})`
                  : 'Nothing selected'}
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button type="button" size="icon" variant="outline" onClick={playPause} title="Play / pause (space)">
                {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <Button
                type="button"
                variant={loop ? 'default' : 'outline'}
                onClick={() => setLoop((v) => !v)}
                title="Loop (L) — the selection or the whole sample"
                className="gap-2"
              >
                <Repeat className="h-4 w-4" /> Loop
              </Button>
              <Button
                type="button"
                variant={hearingOriginal ? 'default' : 'outline'}
                onClick={() => setHearingOriginal((v) => !v)}
                disabled={!changed}
                title="Original / version (B)"
              >
                A/B
              </Button>
              <div className="mx-1 h-6 w-px bg-border" />
              <Button
                type="button"
                variant="outline"
                onClick={() => edit(cut)}
                disabled={editingBlocked}
                title="Cut the selection (Delete)"
                className="gap-2"
              >
                <Scissors className="h-4 w-4" /> Cut
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => edit(keep)}
                disabled={editingBlocked}
                title="Keep only the selection (K)"
                className="gap-2"
              >
                <Crop className="h-4 w-4" /> Keep selection
              </Button>
              <div className="mx-1 h-6 w-px bg-border" />
              <Button
                type="button"
                size="icon"
                variant="outline"
                onClick={undo}
                disabled={step === 0}
                title="Undo (Ctrl+Z)"
              >
                <Undo2 className="h-4 w-4" />
              </Button>
              <Button
                type="button"
                size="icon"
                variant="outline"
                onClick={redo}
                disabled={step >= history.length - 1}
                title="Redo (Ctrl+Y)"
              >
                <Redo2 className="h-4 w-4" />
              </Button>
              <div className="ml-auto flex w-48 items-center gap-2 text-xs text-muted-foreground">
                Zoom
                <Slider value={[zoom]} min={0} max={500} step={10} onValueChange={([v]) => setZoom(v)} />
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <Module
                title="Sibilance (de-esser)"
                description="Turns down harsh s / sh / ch only while they sound — the rest of the voice stays."
                on={filters.deesser.on}
                onToggle={(on) => updateModule('deesser', { on }, true)}
              >
                <FilterSlider
                  label="Strength"
                  value={filters.deesser.strength}
                  display={pct(filters.deesser.strength)}
                  min={0}
                  max={1}
                  step={0.01}
                  disabled={!filters.deesser.on}
                  onChange={(v) => updateModule('deesser', { strength: v }, false)}
                  onCommit={(v) => updateModule('deesser', { strength: v }, true)}
                />
                <FilterSlider
                  label="Frequency (lower also catches sh/ch, higher only sharp s)"
                  value={filters.deesser.frequency}
                  display={pct(filters.deesser.frequency)}
                  min={0}
                  max={1}
                  step={0.01}
                  disabled={!filters.deesser.on}
                  onChange={(v) => updateModule('deesser', { frequency: v }, false)}
                  onCommit={(v) => updateModule('deesser', { frequency: v }, true)}
                />
              </Module>

              <Module
                title="Clicks and mouth noise (de-click)"
                description="Finds short clicks and patches them with the surrounding sound."
                on={filters.declick.on}
                onToggle={(on) => updateModule('declick', { on }, true)}
              >
                <FilterSlider
                  label="Sensitivity"
                  value={filters.declick.sensitivity}
                  display={pct(filters.declick.sensitivity)}
                  min={0}
                  max={1}
                  step={0.01}
                  disabled={!filters.declick.on}
                  onChange={(v) => updateModule('declick', { sensitivity: v }, false)}
                  onCommit={(v) => updateModule('declick', { sensitivity: v }, true)}
                />
              </Module>

              <Module
                title="Darken the top (shelf)"
                description="Gently lowers everything above the threshold — for hiss and harshness across the recording."
                on={filters.treble.on}
                onToggle={(on) => updateModule('treble', { on }, true)}
              >
                <FilterSlider
                  label="From"
                  value={filters.treble.hz}
                  display={`${(filters.treble.hz / 1000).toFixed(1)} kHz`}
                  min={3000}
                  max={16000}
                  step={100}
                  disabled={!filters.treble.on}
                  onChange={(v) => updateModule('treble', { hz: v }, false)}
                  onCommit={(v) => updateModule('treble', { hz: v }, true)}
                />
                <FilterSlider
                  label="By"
                  value={filters.treble.db}
                  display={`${filters.treble.db.toFixed(1)} dB`}
                  min={-24}
                  max={0}
                  step={0.5}
                  disabled={!filters.treble.on}
                  onChange={(v) => updateModule('treble', { db: v }, false)}
                  onCommit={(v) => updateModule('treble', { db: v }, true)}
                />
              </Module>

              <Module
                title="Silence between words (gate)"
                description="Whatever is quieter than the threshold gets turned down — breaths and noise in the pauses."
                on={filters.gate.on}
                onToggle={(on) => updateModule('gate', { on }, true)}
              >
                <FilterSlider
                  label="Threshold"
                  value={filters.gate.threshold_db}
                  display={`${filters.gate.threshold_db} dB`}
                  min={-80}
                  max={-10}
                  step={1}
                  disabled={!filters.gate.on}
                  onChange={(v) => updateModule('gate', { threshold_db: v }, false)}
                  onCommit={(v) => updateModule('gate', { threshold_db: v }, true)}
                />
                <FilterSlider
                  label="Reduction"
                  value={filters.gate.reduction_db}
                  display={`${filters.gate.reduction_db} dB`}
                  min={-60}
                  max={0}
                  step={1}
                  disabled={!filters.gate.on}
                  onChange={(v) => updateModule('gate', { reduction_db: v }, false)}
                  onCommit={(v) => updateModule('gate', { reduction_db: v }, true)}
                />
              </Module>
            </div>
            {filterError && <p className="text-sm text-destructive">Filters: {filterError}</p>}
          </div>
        )}

        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button type="button" onClick={() => void apply()} disabled={!current || busy}>
            {changed ? 'Apply' : 'Close'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
