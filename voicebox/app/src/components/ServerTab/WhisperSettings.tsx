import { Loader2, RefreshCw, Unplug } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Toggle } from '@/components/ui/toggle';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { WhisperDevice, WhisperModel, WhisperQuant, WhisperState } from '@/lib/api/types';
import { useGenerationSettings } from '@/lib/hooks/useSettings';
import { SettingRow, SettingSection } from './SettingRow';

const NAMES: Record<WhisperModel, string> = {
  medium: 'Medium',
  large: 'Large v3',
  turbo: 'Large v3 Turbo',
};
const DEVICES: Record<WhisperDevice, string> = {
  gpu: 'the GPU',
  cpu: 'the CPU',
};

/**
 * Whisper (whisper.cpp) — its own container, a separate click from the voice.
 *
 * The choice goes INTO THE DATABASE like Vox's weights, and the dictation uses it
 * (`/transcribe` without `model`). `loaded_*` = what the process really holds —
 * it differs from the choice until a reload, and that has to be visible.
 */
export function WhisperSettings() {
  const { settings, update } = useGenerationSettings();
  const { toast } = useToast();
  const model = (settings?.whisper_model ?? 'large') as WhisperModel;
  const quant = (settings?.whisper_quant ?? 'q5') as WhisperQuant;
  const device = (settings?.whisper_device ?? 'gpu') as WhisperDevice;
  const onStartup = settings?.whisper_on_startup ?? false;
  const flash = settings?.whisper_fa ?? true;
  const [state, setState] = useState<WhisperState | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    apiClient.getWhisperState().then(setState).catch(() => {});
  }, []);

  const size = (m: WhisperModel, q: WhisperQuant) => {
    const f = state?.files.find((x) => x.model === m && x.quant === q);
    if (!f) return '';
    return f.on_disk ? ` — ${(f.size_mib / 1024).toFixed(1)} GB` : ' — no file';
  };

  const loaded = state?.loaded && state.loaded_model;
  const mismatch =
    !!loaded &&
    (state?.loaded_model !== model ||
      state?.loaded_quant !== quant ||
      state?.loaded_device !== device ||
      (state?.loaded_fa ?? true) !== flash);

  const load = useCallback(async () => {
    setBusy(true);
    try {
      // ⛔ NO ARGUMENTS — the server takes the choice from the DATABASE, like Vox's reload.
      const s = await apiClient.loadWhisper();
      setState(s);
      toast({
        title: `Whisper loaded — ${s.loaded_model} ${s.loaded_quant} on ${DEVICES[s.loaded_device ?? 'gpu']}`,
        description: s.ms ? `Up in ${(s.ms / 1000).toFixed(1)} s.` : undefined,
      });
    } catch (e) {
      toast({
        title: 'Could not load Whisper',
        description: e instanceof Error ? e.message : String(e),
        variant: 'destructive',
      });
    } finally {
      setBusy(false);
    }
  }, [toast]);

  const unload = useCallback(async () => {
    setBusy(true);
    try {
      setState(await apiClient.unloadWhisper());
    } catch (e) {
      toast({
        title: 'Could not unload Whisper',
        description: e instanceof Error ? e.message : String(e),
        variant: 'destructive',
      });
    } finally {
      setBusy(false);
    }
  }, [toast]);

  return (
    <SettingSection
      title="Whisper — speech recognition"
      description="whisper.cpp in its own container (ggml). It loads and unloads independently of the voice model, and the dictation can talk to it even when Voicebox is down. Dictation uses what is chosen here."
    >
      {state && !state.container && (
        <SettingRow title="The whisper container does not answer">
          <span className="text-sm text-amber-500">
            Start it (<code className="text-xs">docker start whisper</code>) — the choice below is saved anyway.
          </span>
        </SettingRow>
      )}

      <SettingRow
        title="Model"
        description="Turbo is Large v3 with a slimmed-down decoder — faster, almost the same quality."
        action={
          <Select
            value={model}
            onValueChange={(v) => update({ whisper_model: v as WhisperModel })}
            disabled={busy}
          >
            <SelectTrigger className="w-[190px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(Object.keys(NAMES) as WhisperModel[]).map((m) => (
                <SelectItem key={m} value={m}>
                  {NAMES[m]}
                  {size(m, quant)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        }
      />

      <SettingRow
        title="Quantization"
        description="q5 — smaller and faster. q8 — closer to the full weights."
        action={
          <Select
            value={quant}
            onValueChange={(v) => update({ whisper_quant: v as WhisperQuant })}
            disabled={busy}
          >
            <SelectTrigger className="w-[190px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="q5">q5{size(model, 'q5')}</SelectItem>
              <SelectItem value="q8">q8{size(model, 'q8')}</SelectItem>
            </SelectContent>
          </Select>
        }
      />

      <SettingRow
        title="Device"
        description="The CPU leaves the VRAM to the voice. With only a few cores in the container, Large will be slow there."
        action={
          <Select
            value={device}
            onValueChange={(v) => update({ whisper_device: v as WhisperDevice })}
            disabled={busy}
          >
            <SelectTrigger className="w-[190px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="gpu">GPU</SelectItem>
              <SelectItem value="cpu">CPU</SelectItem>
            </SelectContent>
          </Select>
        }
      />

      <SettingRow
        title="Flash attention"
        description={
          'Attention computed in tiles — faster and less VRAM; whisper.cpp has it on by default. ' +
          'A change takes effect after reloading Whisper.'
        }
        htmlFor="whisperFlash"
        action={
          <Toggle
            id="whisperFlash"
            checked={flash}
            onCheckedChange={(v) => update({ whisper_fa: v })}
          />
        }
      />

      <SettingRow
        title="Load on startup"
        description="On — the chosen model loads by itself whenever its container or Voicebox starts (in the background). Off — it waits for a click or the first dictation from the GUI."
        htmlFor="whisperOnStartup"
        action={
          <Toggle
            id="whisperOnStartup"
            checked={onStartup}
            onCheckedChange={(v) => update({ whisper_on_startup: v })}
          />
        }
      />

      <SettingRow
        title="State"
        description={
          loaded
            ? `On ${DEVICES[state?.loaded_device ?? 'gpu']}: ${NAMES[state!.loaded_model!]} ${state?.loaded_quant}.` +
              (mismatch ? ' The choice differs — reload, or the next dictation will.' : '')
            : 'Nothing is loaded. The first dictation loads the chosen model by itself.'
        }
        action={
          <div className="flex items-center gap-2">
            <Button
              variant={mismatch || !loaded ? 'default' : 'outline'}
              size="sm"
              onClick={load}
              disabled={busy}
            >
              {busy ? (
                <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
              ) : (
                <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
              )}
              {busy ? 'Working…' : loaded ? 'Reload' : 'Load'}
            </Button>
            <Button variant="outline" size="sm" onClick={unload} disabled={busy || !loaded}>
              <Unplug className="h-3.5 w-3.5 mr-1.5" />
              Unload
            </Button>
          </div>
        }
      />
    </SettingSection>
  );
}
