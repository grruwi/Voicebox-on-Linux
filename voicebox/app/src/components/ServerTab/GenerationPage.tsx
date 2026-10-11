import { FolderOpen, Languages, Loader2, Mic, RefreshCw, Unplug, Zap } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Toggle } from '@/components/ui/toggle';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { OmniWeights, VoxAcoustic, VoxBackend, VoxBaseLm } from '@/lib/api/types';
import { useGenerationSettings } from '@/lib/hooks/useSettings';
import { usePlatform } from '@/platform/PlatformContext';
import { useServerStore } from '@/stores/serverStore';
import { GpuPicker } from './GpuPicker';
import { SettingRow, SettingSection } from './SettingRow';
import { WhisperSettings } from './WhisperSettings';

const BACKEND_NAMES: Record<VoxBackend, string> = { hip: 'ROCm (HIP)', vulkan: 'Vulkan (RADV)' };

const TABS = ['general', 'vox', 'omni', 'whisper'] as const;
const TAB_KEY = 'generation-settings-tab';

export function GenerationPage() {
  const { t } = useTranslation();
  const platform = usePlatform();
  const serverUrl = useServerStore((state) => state.serverUrl);
  const { settings, update } = useGenerationSettings();
  const persistedMaxChunkChars = settings?.max_chunk_chars ?? 800;
  const persistedCrossfadeMs = settings?.crossfade_ms ?? 50;
  const normalizeAudio = settings?.normalize_audio ?? true;
  const autoplayOnGenerate = settings?.autoplay_on_generate ?? true;
  // Slider mirrors persist on commit (pointer-up / keyboard-release) only —
  // onValueChange would fire a PATCH for every pointer-move pixel and round-
  // trip mid-drag failures could leave persisted state out of sync with UI.
  const [maxChunkChars, setMaxChunkChars] = useState(persistedMaxChunkChars);
  const [crossfadeMs, setCrossfadeMs] = useState(persistedCrossfadeMs);
  useEffect(() => setMaxChunkChars(persistedMaxChunkChars), [persistedMaxChunkChars]);
  useEffect(() => setCrossfadeMs(persistedCrossfadeMs), [persistedCrossfadeMs]);
  // VoxCPM2 knobs live in the database — they take effect from the next generation,
  // no backend restart. A slider saves on RELEASE (`onValueCommit`), not while
  // dragging, otherwise a PATCH goes out for every pixel.
  const persistedVoxCfg = settings?.vox_cfg ?? 2.0;
  const persistedVoxTimesteps = settings?.vox_timesteps ?? 10;
  const persistedVoxTemperature = settings?.vox_temperature ?? 1.0;
  const [voxCfg, setVoxCfg] = useState(persistedVoxCfg);
  const [voxTimesteps, setVoxTimesteps] = useState(persistedVoxTimesteps);
  const [voxTemperature, setVoxTemperature] = useState(persistedVoxTemperature);
  useEffect(() => setVoxCfg(persistedVoxCfg), [persistedVoxCfg]);
  useEffect(() => setVoxTimesteps(persistedVoxTimesteps), [persistedVoxTimesteps]);
  useEffect(() => setVoxTemperature(persistedVoxTemperature), [persistedVoxTemperature]);
  // The seed is kept as TEXT, not a number — an empty field has to be distinguishable
  // from zero, and `<input type="number">` returns an empty string for both. Saved
  // when leaving the field (blur / Enter), not on every key.
  const persistedVoxSeed = settings?.vox_seed ?? null;
  const [voxSeed, setVoxSeed] = useState(
    persistedVoxSeed === null ? '' : String(persistedVoxSeed),
  );
  useEffect(() => {
    setVoxSeed(persistedVoxSeed === null ? '' : String(persistedVoxSeed));
  }, [persistedVoxSeed]);

  const saveSeed = useCallback(() => {
    const text = voxSeed.trim();
    // Empty = `null` = a random seed on every generation. `null` goes through the API
    // and CLEARS the column — otherwise there would be no way back to random.
    const value = text === '' ? null : Number.parseInt(text, 10);
    if (value === persistedVoxSeed) return;
    update({ vox_seed: value });
  }, [voxSeed, persistedVoxSeed, update]);
  const [opening, setOpening] = useState(false);
  const [generationsPath, setGenerationsPath] = useState<string | null>(null);
  const { toast } = useToast();
  // ⭐ The choice goes INTO THE DATABASE (`vox_base_lm` / `vox_acoustic`), so it survives
  // a restart and every road to the engine reads it. `loaded*` = what the engine really
  // holds — it differs from the choice until a reload, and that has to be visible.
  const selectedBaseLm = (settings?.vox_base_lm ?? 'q8') as VoxBaseLm;
  const selectedAcoustic = (settings?.vox_acoustic ?? 'f16') as VoxAcoustic;
  const selectedBackend = (settings?.vox_backend ?? 'vulkan') as VoxBackend;
  const [loadedBaseLm, setLoadedBaseLm] = useState<VoxBaseLm | null>(null);
  const [loadedAcoustic, setLoadedAcoustic] = useState<VoxAcoustic | null>(null);
  const [loadedBackend, setLoadedBackend] = useState<VoxBackend | null>(null);
  const selectedOmni = (settings?.omni_weights ?? 'q8') as OmniWeights;
  const [loadedOmni, setLoadedOmni] = useState<OmniWeights | null>(null);
  const [loadedOmniFlash, setLoadedOmniFlash] = useState<boolean | null>(null);
  // OmniVoice sliders: empty in the database = upstream's default (2.0 / 32) — that is what we show.
  const persistedOmniCfg = settings?.omni_cfg ?? 2.0;
  const persistedOmniSteps = settings?.omni_steps ?? 32;
  const [omniCfg, setOmniCfg] = useState(persistedOmniCfg);
  const [omniSteps, setOmniSteps] = useState(persistedOmniSteps);
  useEffect(() => setOmniCfg(persistedOmniCfg), [persistedOmniCfg]);
  useEffect(() => setOmniSteps(persistedOmniSteps), [persistedOmniSteps]);
  const persistedOmniSpeed = settings?.omni_speed ?? 1.0;
  const [omniSpeed, setOmniSpeed] = useState(persistedOmniSpeed);
  useEffect(() => setOmniSpeed(persistedOmniSpeed), [persistedOmniSpeed]);
  const omniChunkMode = settings?.omni_chunk_mode ?? 'none';
  const persistedOmniChunkChars = settings?.omni_chunk_chars ?? 160;
  const [omniChunkChars, setOmniChunkChars] = useState(persistedOmniChunkChars);
  useEffect(() => setOmniChunkChars(persistedOmniChunkChars), [persistedOmniChunkChars]);
  const persistedOmniSeed = settings?.omni_seed ?? null;
  const [omniSeed, setOmniSeed] = useState(persistedOmniSeed === null ? '' : String(persistedOmniSeed));
  useEffect(() => {
    setOmniSeed(persistedOmniSeed === null ? '' : String(persistedOmniSeed));
  }, [persistedOmniSeed]);
  const saveOmniSeed = useCallback(() => {
    const text = omniSeed.trim();
    const value = text === '' ? null : Number.parseInt(text, 10);
    if (value === persistedOmniSeed) return;
    update({ omni_seed: value });
  }, [omniSeed, persistedOmniSeed, update]);
  const [omniBusy, setOmniBusy] = useState(false);
  const selectedOmniFlash = settings?.omni_flash ?? true;
  const omniMismatch =
    loadedOmni !== null &&
    (loadedOmni !== selectedOmni || (loadedOmniFlash ?? true) !== selectedOmniFlash);
  const reloadOmni = useCallback(async () => {
    setOmniBusy(true);
    try {
      const s = await apiClient.reloadOmni();
      setLoadedOmni(s.loaded_omni ?? null);
      setLoadedOmniFlash(s.loaded_omni_flash ?? null);
      setLoadedBaseLm(s.loaded_base_lm ?? null);
      setLoadedAcoustic(s.loaded_acoustic ?? null);
      toast({
        title: `OmniVoice reloaded — ${s.loaded_omni ?? '?'}${s.loaded_omni_flash ? ' + flash attention' : ''}`,
        description: s.ms ? `Up in ${(s.ms / 1000).toFixed(1)} s. Vox left the card.` : undefined,
      });
    } catch (e) {
      toast({
        title: 'Could not reload OmniVoice',
        description: e instanceof Error ? e.message : String(e),
        variant: 'destructive',
      });
    } finally {
      setOmniBusy(false);
    }
  }, [toast]);
  const unloadOmni = useCallback(async () => {
    setOmniBusy(true);
    try {
      const s = await apiClient.unloadOmni();
      setLoadedOmni(s.loaded_omni ?? null);
      setLoadedOmniFlash(s.loaded_omni_flash ?? null);
    } catch (e) {
      toast({
        title: 'Could not unload OmniVoice',
        description: e instanceof Error ? e.message : String(e),
        variant: 'destructive',
      });
    } finally {
      setOmniBusy(false);
    }
  }, [toast]);
  // The last opened tab survives a page refresh.
  const [tab, setTab] = useState(() => {
    const saved = localStorage.getItem(TAB_KEY);
    return saved && (TABS as readonly string[]).includes(saved) ? saved : 'general';
  });
  const changeTab = useCallback((v: string) => {
    setTab(v);
    localStorage.setItem(TAB_KEY, v);
  }, []);
  const [ttsAvailable, setTtsAvailable] = useState(true);
  const [reloading, setReloading] = useState(false);
  const needsReload =
    (loadedBaseLm !== null && selectedBaseLm !== loadedBaseLm) ||
    (loadedAcoustic !== null && selectedAcoustic !== loadedAcoustic) ||
    (loadedBackend !== null && selectedBackend !== loadedBackend);

  useEffect(() => {
    apiClient
      .getTtsState()
      .then((s) => {
        setTtsAvailable(s.available);
        setLoadedBaseLm(s.loaded_base_lm ?? null);
        setLoadedAcoustic(s.loaded_acoustic ?? null);
        setLoadedBackend(s.loaded_backend ?? null);
        setLoadedOmni(s.loaded_omni ?? null);
        setLoadedOmniFlash(s.loaded_omni_flash ?? null);
      })
      .catch(() => {});
  }, []);

  const reloadModel = useCallback(async () => {
    setReloading(true);
    try {
      // ⛔ NO ARGUMENTS. The server takes the choice from the DATABASE, i.e. the very place
      // the panel has just saved. Passing values from here froze them in the `useCallback`
      // closure and the button sent the acoustic part from BEFORE the change.
      const s = await apiClient.reloadTts();
      setLoadedBaseLm(s.loaded_base_lm ?? null);
      setLoadedAcoustic(s.loaded_acoustic ?? null);
      setLoadedBackend(s.loaded_backend ?? null);
      setLoadedOmni(s.loaded_omni ?? null);
      setLoadedOmniFlash(s.loaded_omni_flash ?? null);
      toast({
        title: s.loaded_base_lm
          ? `Model reloaded — ${s.loaded_base_lm} / ${s.loaded_acoustic ?? '?'} / ${BACKEND_NAMES[s.loaded_backend ?? 'hip']}`
          : 'Model reloaded',
        description: s.ms ? `Up in ${(s.ms / 1000).toFixed(1)} s.` : undefined,
      });
    } catch (e) {
      toast({
        title: 'Could not reload the model',
        description: e instanceof Error ? e.message : String(e),
        variant: 'destructive',
      });
    } finally {
      setReloading(false);
    }
  }, [toast]);

  useEffect(() => {
    fetch(`${serverUrl}/health/filesystem`)
      .then((res) => res.json())
      .then((data) => {
        const genDir = data.directories?.find((d: { path: string }) =>
          d.path.includes('generations'),
        );
        if (genDir?.path) setGenerationsPath(genDir.path);
      })
      .catch(() => {});
  }, [serverUrl]);

  const openGenerationsFolder = useCallback(async () => {
    if (!generationsPath) return;
    setOpening(true);
    try {
      await platform.filesystem.openPath(generationsPath);
    } catch (e) {
      console.error('Failed to open generations folder:', e);
    } finally {
      setOpening(false);
    }
  }, [platform, generationsPath]);

  return (
    <div className="flex gap-8 items-start max-w-5xl">
      <div className="flex-1 min-w-0 max-w-2xl space-y-8">
      <Tabs value={tab} onValueChange={changeTab} className="space-y-6">
        <TabsList>
          <TabsTrigger value="general">General</TabsTrigger>
          <TabsTrigger value="vox">Vox</TabsTrigger>
          <TabsTrigger value="omni">OmniVoice</TabsTrigger>
          <TabsTrigger value="whisper">Whisper</TabsTrigger>
        </TabsList>

        <TabsContent value="general" className="space-y-8">
          <SettingSection
            title="Engines"
            description="The voice engines (audio.cpp, PyTorch) run in their own containers. By default only one model sits on the card at a time — the Offloader unloads the others before a model loads."
          >
            <SettingRow
              title={t('settings.generation.keepAllModels.title')}
              description={t('settings.generation.keepAllModels.description')}
              htmlFor="keepAllModels"
              action={
                <Toggle
                  id="keepAllModels"
                  checked={settings?.keep_all_models ?? false}
                  onCheckedChange={(v) => update({ keep_all_models: v })}
                />
              }
            />
          </SettingSection>
          <GpuPicker />
          <SettingSection
            title={t('settings.generation.title')}
            description={t('settings.generation.description')}
          >
          <SettingRow
            title={t('settings.generation.chunkLimit.title')}
            description={t('settings.generation.chunkLimit.description')}
            action={
              <span className="text-sm tabular-nums text-muted-foreground">
                {t('settings.generation.chunkLimit.value', { chars: maxChunkChars })}
              </span>
            }
          >
            <Slider
              id="maxChunkChars"
              value={[maxChunkChars]}
              onValueChange={([value]) => setMaxChunkChars(value)}
              onValueCommit={([value]) => update({ max_chunk_chars: value })}
              min={100}
              max={5000}
              step={50}
              aria-label={t('settings.generation.chunkLimit.title')}
            />
          </SettingRow>

          <SettingRow
            title={t('settings.generation.crossfade.title')}
            description={t('settings.generation.crossfade.description')}
            action={
              <span className="text-sm tabular-nums text-muted-foreground">
                {crossfadeMs === 0
                  ? t('settings.generation.crossfade.cut')
                  : t('settings.generation.crossfade.ms', { ms: crossfadeMs })}
              </span>
            }
          >
            <Slider
              id="crossfadeMs"
              value={[crossfadeMs]}
              onValueChange={([value]) => setCrossfadeMs(value)}
              onValueCommit={([value]) => update({ crossfade_ms: value })}
              min={0}
              max={200}
              step={10}
              aria-label={t('settings.generation.crossfade.title')}
            />
          </SettingRow>

          <SettingRow
            title="Backend"
            description={
              loadedBackend && loadedBackend !== selectedBackend
                ? `Which audio.cpp binary computes the voice — separate files, the unused one takes nothing. Running now: ${BACKEND_NAMES[loadedBackend]}.`
                : loadedBackend
                  ? `Which audio.cpp binary computes the voice — separate files, the unused one takes nothing. Running: ${BACKEND_NAMES[loadedBackend]}.`
                  : 'Which audio.cpp binary computes the voice (Vox and OmniVoice) — separate files, the unused one takes nothing.'
            }
            action={
              <Select
                value={selectedBackend}
                onValueChange={(v) => update({ vox_backend: v as VoxBackend })}
                disabled={!ttsAvailable || reloading}
              >
                <SelectTrigger className="w-[150px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="hip">ROCm (HIP)</SelectItem>
                  <SelectItem value="vulkan">Vulkan (RADV)</SelectItem>
                </SelectContent>
              </Select>
            }
          />

          <SettingRow
            title={t('settings.generation.normalize.title')}
            description={t('settings.generation.normalize.description')}
            htmlFor="normalizeAudio"
            action={
              <Toggle
                id="normalizeAudio"
                checked={normalizeAudio}
                onCheckedChange={(v) => update({ normalize_audio: v })}
              />
            }
          />

          <SettingRow
            title={t('settings.generation.autoplay.title')}
            description={t('settings.generation.autoplay.description')}
            htmlFor="autoplayOnGenerate"
            action={
              <Toggle
                id="autoplayOnGenerate"
                checked={autoplayOnGenerate}
                onCheckedChange={(v) => update({ autoplay_on_generate: v })}
              />
            }
          />

          <SettingRow
            title={t('settings.generation.folder.title')}
            description={generationsPath ?? t('settings.generation.folder.description')}
            action={
              <Button
                variant="outline"
                size="sm"
                onClick={openGenerationsFolder}
                disabled={opening || !generationsPath}
              >
                <FolderOpen className="h-3.5 w-3.5 mr-1.5" />
                {t('settings.generation.folder.open')}
              </Button>
            }
          />
          </SettingSection>
        </TabsContent>

        <TabsContent value="vox" className="space-y-8">
          <SettingSection
            title="Vox (VoxCPM2)"
            description="Both Vox engines (audio.cpp and PyTorch) use these settings. Flash attention is always on — every attention goes through FA in our audio.cpp patches."
          >
          <SettingRow
            title="Following the parentheses (cfg)"
            description="Higher — the model follows the description in parentheses more closely, but speaks flat. Lower — livelier, with more spread between generations."
            action={
              <span className="text-sm tabular-nums text-muted-foreground">
                {voxCfg.toFixed(1)}
              </span>
            }
          >
            <Slider
              id="voxCfg"
              value={[voxCfg]}
              onValueChange={([value]) => setVoxCfg(value)}
              onValueCommit={([value]) => update({ vox_cfg: value })}
              min={0.5}
              max={5}
              step={0.1}
              aria-label="Following the parentheses"
            />
          </SettingRow>

          <SettingRow
            title="Diffusion steps"
            description="The more steps, the better the quality. Each step costs about 0.06 s."
            action={
              <span className="text-sm tabular-nums text-muted-foreground">
                {voxTimesteps}
              </span>
            }
          >
            <Slider
              id="voxTimesteps"
              value={[voxTimesteps]}
              onValueChange={([value]) => setVoxTimesteps(value)}
              onValueCommit={([value]) => update({ vox_timesteps: value })}
              min={4}
              max={60}
              step={1}
              aria-label="Diffusion steps"
            />
          </SettingRow>

          <SettingRow
            title="Noise temperature"
            description={
              'How much randomness the model lets into the diffusion. 1.0 is the engine\'s value. Lower — ' +
              'calmer and more repeatable, but flat. Higher — livelier, at the price of the ' +
              'same line coming out differently every time.'
            }
            action={
              <span className="text-sm tabular-nums text-muted-foreground">
                {voxTemperature.toFixed(2)}
              </span>
            }
          >
            <Slider
              id="voxTemperature"
              value={[voxTemperature]}
              onValueChange={([value]) => setVoxTemperature(value)}
              onValueCommit={([value]) => update({ vox_temperature: value })}
              min={0.1}
              max={2}
              step={0.05}
              aria-label="Noise temperature"
            />
          </SettingRow>

          <SettingRow
            title="Random seed"
            description={
              'Empty — different every time. A number — the same line gives the same file ' +
              'down to the byte, so "generate again" stops changing anything.'
            }
            action={
              <Input
                id="voxSeed"
                value={voxSeed}
                onChange={(e) => setVoxSeed(e.target.value.replace(/\D/g, '').slice(0, 10))}
                onBlur={saveSeed}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') e.currentTarget.blur();
                }}
                placeholder="random"
                inputMode="numeric"
                className="w-[140px] text-right tabular-nums"
                aria-label="Random seed"
              />
            }
          />

          <SettingRow
            title="Weights — language part"
            description={
              !ttsAvailable
                ? 'BaseLM — the plan of the utterance: words, stress, intonation. No audio.cpp engine is configured.'
                : loadedBaseLm && loadedBaseLm !== selectedBaseLm
                  ? `BaseLM — the plan of the utterance: words, stress, intonation. On the card now: ${loadedBaseLm}.`
                  : loadedBaseLm
                    ? `BaseLM — the plan of the utterance: words, stress, intonation. On the card: ${loadedBaseLm}.`
                    : 'BaseLM — the plan of the utterance: words, stress, intonation.'
            }
            action={
              <Select
                value={selectedBaseLm}
                onValueChange={(v) => update({ vox_base_lm: v as VoxBaseLm })}
                disabled={!ttsAvailable || reloading}
              >
                <SelectTrigger className="w-[150px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="q8">q8 — 1.6 GB</SelectItem>
                  <SelectItem value="f16">f16 — 3.0 GB</SelectItem>
                </SelectContent>
              </Select>
            }
          />

          <SettingRow
            title="Weights — acoustic part"
            description={
              loadedAcoustic && loadedAcoustic !== selectedAcoustic
                ? `DiT, LocEnc, ResidualLM — the detail of the sound. The AudioVAE head is always f32. On the card now: ${loadedAcoustic}.`
                : loadedAcoustic
                  ? `DiT, LocEnc, ResidualLM — the detail of the sound. The AudioVAE head is always f32. On the card: ${loadedAcoustic}.`
                  : 'DiT, LocEnc, ResidualLM — the detail of the sound. The AudioVAE head is always f32.'
            }
            action={
              <div className="flex items-center gap-2">
                <Select
                  value={selectedAcoustic}
                  onValueChange={(v) => update({ vox_acoustic: v as VoxAcoustic })}
                  disabled={!ttsAvailable || reloading}
                >
                  <SelectTrigger className="w-[150px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="f16">f16 — 1.6 GB</SelectItem>
                    <SelectItem value="f32">f32 — 3.3 GB</SelectItem>
                  </SelectContent>
                </Select>
                <Button
                  variant={needsReload ? 'default' : 'outline'}
                  size="sm"
                  onClick={reloadModel}
                  disabled={!ttsAvailable || reloading}
                >
                  {reloading ? (
                    <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                  ) : (
                    <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
                  )}
                  {reloading ? 'Loading…' : 'Reload'}
                </Button>
              </div>
            }
          />

          </SettingSection>
        </TabsContent>

        <TabsContent value="omni" className="space-y-8">
          <SettingSection
            title="OmniVoice"
            description="The same audio.cpp engine as Vox, one model at a time — OmniVoice comes in with its first utterance, and Vox leaves."
          >
          <SettingRow
            title="Weights"
            description={
              loadedOmni && loadedOmni !== selectedOmni
                ? `On the card now: ${loadedOmni} — the new one comes in with the next OmniVoice utterance.`
                : loadedOmni
                  ? `On the card: ${loadedOmni}.`
                  : 'Upstream\'s ready GGUF files.'
            }
            action={
              <Select
                value={selectedOmni}
                onValueChange={(v) => update({ omni_weights: v as OmniWeights })}
                disabled={!ttsAvailable}
              >
                <SelectTrigger className="w-[150px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="q8">q8 — 1.3 GB</SelectItem>
                  <SelectItem value="f16">f16 — 1.6 GB</SelectItem>
                </SelectContent>
              </Select>
            }
          />


          <SettingRow
            title="Flash attention"
            description={
              'Attention computed in tiles instead of the full matrix — without it OmniVoice holds a ' +
              'T×T matrix (sample + text + speech, times two for cfg) and VRAM grows with the longest utterance. ' +
              'Upstream: "small output drift". A change takes effect with the next OmniVoice load.' +
              (loadedOmniFlash !== null ? ` Now: ${loadedOmniFlash ? 'on' : 'off'}.` : '')
            }
            htmlFor="omniFlash"
            action={
              <Toggle
                id="omniFlash"
                checked={settings?.omni_flash ?? true}
                onCheckedChange={(v) => update({ omni_flash: v })}
              />
            }
          />

          <SettingRow
            title="Following the description (guidance)"
            description="The same as Vox's cfg: higher — closer, but flat; lower — livelier. Upstream's default: 2.0."
            action={
              <span className="text-sm tabular-nums text-muted-foreground">
                {omniCfg.toFixed(1)}
              </span>
            }
          >
            <Slider
              id="omniCfg"
              value={[omniCfg]}
              onValueChange={([value]) => setOmniCfg(value)}
              onValueCommit={([value]) => update({ omni_cfg: value })}
              min={0.5}
              max={5}
              step={0.1}
              aria-label="Following the description"
            />
          </SettingRow>

          <SettingRow
            title="Steps"
            description="Rounds refining the WHOLE utterance at once (not a step per chunk like Vox). Upstream's default: 32."
            action={
              <span className="text-sm tabular-nums text-muted-foreground">{omniSteps}</span>
            }
          >
            <Slider
              id="omniSteps"
              value={[omniSteps]}
              onValueChange={([value]) => setOmniSteps(value)}
              onValueCommit={([value]) => update({ omni_steps: value })}
              min={4}
              max={64}
              step={1}
              aria-label="OmniVoice steps"
            />
          </SettingRow>

          <SettingRow
            title="Speech speed"
            description="A multiplier for the estimated utterance length — OmniVoice guesses it from the sample's pace, so a sample with pauses gives speech that is too slow. Higher = faster. Upstream's default: 1.0."
            action={
              <span className="text-sm tabular-nums text-muted-foreground">
                {omniSpeed.toFixed(2)}×
              </span>
            }
          >
            <Slider
              id="omniSpeed"
              value={[omniSpeed]}
              onValueChange={([value]) => setOmniSpeed(value)}
              onValueCommit={([value]) => update({ omni_speed: value })}
              min={0.5}
              max={2}
              step={0.05}
              aria-label="OmniVoice speech speed"
            />
          </SettingRow>

          <SettingRow
            title="Stream in chunks"
            description={
              'OmniVoice computes the whole utterance at once, so the sound only comes after all of it. ' +
              'Cut text plays from the first chunk, and the next ones are computed meanwhile (like ' +
              '--stream-by-line in omnivoice.cpp). Off — upstream cuts by itself only above 30 s of speech.'
            }
            action={
              <Select
                value={omniChunkMode}
                onValueChange={(v) => update({ omni_chunk_mode: v as 'none' | 'lines' | 'sentences' })}
              >
                <SelectTrigger className="w-[150px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">off</SelectItem>
                  <SelectItem value="sentences">by sentence</SelectItem>
                  <SelectItem value="lines">by line</SelectItem>
                </SelectContent>
              </Select>
            }
          />

          {omniChunkMode !== 'none' && (
            <SettingRow
              title="Max characters per chunk"
              description={
                omniChunkMode === 'sentences'
                  ? 'Sentences packed up to this limit — less = an earlier first sound, but more seams. Default 160 (the streaming example in upstream\'s docs).'
                  : 'A line ending with a period/question mark is a chunk; longer than the limit gets cut further. Default 160 (the streaming example in upstream\'s docs).'
              }
              action={
                <span className="text-sm tabular-nums text-muted-foreground">{omniChunkChars}</span>
              }
            >
              <Slider
                id="omniChunkChars"
                value={[omniChunkChars]}
                onValueChange={([value]) => setOmniChunkChars(value)}
                onValueCommit={([value]) => update({ omni_chunk_chars: value })}
                min={20}
                max={1000}
                step={10}
                aria-label="Max characters per chunk"
              />
            </SettingRow>
          )}

          <SettingRow
            title="Random seed"
            description="Empty — different every time. A number — the same text gives the same file."
            action={
              <Input
                id="omniSeed"
                value={omniSeed}
                onChange={(e) => setOmniSeed(e.target.value.replace(/\D/g, '').slice(0, 10))}
                onBlur={saveOmniSeed}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') e.currentTarget.blur();
                }}
                placeholder="random"
                className="w-[150px]"
              />
            }
          />

          <SettingRow
            title="State"
            description={
              loadedOmni
                ? `On the card: OmniVoice ${loadedOmni}${loadedOmniFlash ? ' + flash attention' : ''}.` +
                  (omniMismatch ? ' The choice differs — reload, or the next OmniVoice utterance will.' : '')
                : 'Not loaded — it comes in with the first OmniVoice utterance (Vox leaves then).'
            }
            action={
              <div className="flex items-center gap-2">
                <Button
                  variant={omniMismatch ? 'default' : 'outline'}
                  size="sm"
                  onClick={reloadOmni}
                  disabled={!ttsAvailable || omniBusy}
                >
                  {omniBusy ? (
                    <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                  ) : (
                    <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
                  )}
                  {omniBusy ? 'Working…' : loadedOmni ? 'Reload' : 'Load'}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={unloadOmni}
                  disabled={omniBusy || !loadedOmni}
                >
                  <Unplug className="h-3.5 w-3.5 mr-1.5" />
                  Unload
                </Button>
              </div>
            }
          />
          </SettingSection>
        </TabsContent>

        <TabsContent value="whisper" className="space-y-8">
          <WhisperSettings />
        </TabsContent>
      </Tabs>
      </div>

      <aside className="hidden lg:block w-[280px] shrink-0 space-y-6 sticky top-0">
        <div className="space-y-2">
          <h3 className="text-sm font-semibold">{t('settings.generation.sidebar.aboutTitle')}</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {t('settings.generation.sidebar.aboutBody')}
          </p>
        </div>

        <div className="space-y-3">
          <h3 className="text-sm font-semibold">{t('settings.generation.sidebar.differencesTitle')}</h3>
          <ul className="space-y-3 text-sm text-muted-foreground">
            <li className="flex gap-2.5">
              <Mic className="h-4 w-4 shrink-0 mt-0.5 text-accent" />
              <span className="leading-relaxed">
                <span className="text-foreground font-medium">
                  {t('settings.generation.sidebar.clone.title')}
                </span>{' '}
                {t('settings.generation.sidebar.clone.body')}
              </span>
            </li>
            <li className="flex gap-2.5">
              <Languages className="h-4 w-4 shrink-0 mt-0.5 text-accent" />
              <span className="leading-relaxed">
                <span className="text-foreground font-medium">
                  {t('settings.generation.sidebar.engines.title')}
                </span>{' '}
                {t('settings.generation.sidebar.engines.body')}
              </span>
            </li>
            <li className="flex gap-2.5">
              <Zap className="h-4 w-4 shrink-0 mt-0.5 text-accent" />
              <span className="leading-relaxed">
                <span className="text-foreground font-medium">{t('settings.generation.sidebar.agentReady.title')}</span>{' '}
                {t('settings.generation.sidebar.agentReady.body')}
              </span>
            </li>
          </ul>
        </div>
      </aside>
    </div>
  );
}
