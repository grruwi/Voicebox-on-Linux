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
import type { OmniWagi, VoxAkustyka, VoxBackend, VoxWariant } from '@/lib/api/types';

const NAZWA_BACKENDU: Record<VoxBackend, string> = { hip: 'ROCm (HIP)', vulkan: 'Vulkan (RADV)' };
import { useGenerationSettings } from '@/lib/hooks/useSettings';
import { usePlatform } from '@/platform/PlatformContext';
import { useServerStore } from '@/stores/serverStore';
import { SettingRow, SettingSection } from './SettingRow';
import { WhisperUstawienia } from './WhisperUstawienia';
import { WyborKartyGpu } from './WyborKartyGpu';

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
  // Pokretla VoxCPM2 (16.08). Byly w kodzie i w `ustawienia.env`, czyli kazda
  // zmiana wymagala restartu backendu. Teraz ida z baza — dzialaja od nastepnej
  // generacji. Suwak zapisuje sie na PUSZCZENIU (`onValueCommit`), nie w trakcie
  // ciagniecia, bo inaczej leci PATCH na kazdy piksel.
  const persistedVoxCfg = settings?.vox_cfg ?? 2.0;
  const persistedVoxTimesteps = settings?.vox_timesteps ?? 10;
  const persistedVoxTemperature = settings?.vox_temperature ?? 1.0;
  const [voxCfg, setVoxCfg] = useState(persistedVoxCfg);
  const [voxTimesteps, setVoxTimesteps] = useState(persistedVoxTimesteps);
  const [voxTemperature, setVoxTemperature] = useState(persistedVoxTemperature);
  useEffect(() => setVoxCfg(persistedVoxCfg), [persistedVoxCfg]);
  useEffect(() => setVoxTimesteps(persistedVoxTimesteps), [persistedVoxTimesteps]);
  useEffect(() => setVoxTemperature(persistedVoxTemperature), [persistedVoxTemperature]);
  // Ziarno trzymamy jako TEKST, nie liczbę — puste pole musi być odróżnialne od
  // zera, a `<input type="number">` zwraca pusty string i dla jednego, i dla
  // drugiego. Zapis na wyjściu z pola (blur / Enter), nie na każdym klawiszu.
  const persistedVoxSeed = settings?.vox_seed ?? null;
  const [voxSeed, setVoxSeed] = useState(
    persistedVoxSeed === null ? '' : String(persistedVoxSeed),
  );
  useEffect(() => {
    setVoxSeed(persistedVoxSeed === null ? '' : String(persistedVoxSeed));
  }, [persistedVoxSeed]);

  const zapiszSeed = useCallback(() => {
    const tekst = voxSeed.trim();
    // Puste = `null` = ziarno losowe przy każdej generacji. `null` przechodzi
    // przez API i CZYŚCI kolumnę — bez tego nie dałoby się wrócić do losowego.
    const wartosc = tekst === '' ? null : Number.parseInt(tekst, 10);
    if (wartosc === persistedVoxSeed) return;
    update({ vox_seed: wartosc });
  }, [voxSeed, persistedVoxSeed, update]);
  const [opening, setOpening] = useState(false);
  const [generationsPath, setGenerationsPath] = useState<string | null>(null);
  // Wagi VoxCPM2 (21.08). Wariant wybiera sie przy STARCIE `llama-tts-server`,
  // wiec do teraz zmiana quantu kosztowala restart calego Voicebota. To NIE idzie
  // do bazy: `wybrany` to zyczenie, `zaladowany` to stan procesu czytany z /proc.
  // Trwalosc po restarcie dalej daje `ustawienia.env`.
  const { toast } = useToast();
  // ⭐ Wybór idzie DO BAZY (`vox_base_lm` / `vox_akustyka`), więc przeżywa
  // restart i czytają go obie drogi do serwera. `zaladowany_*` to stan procesu
  // z /proc — rozjeżdża się z wyborem do chwili przeładowania i to ma być widać.
  const wybranyWariant = (settings?.vox_base_lm ?? 'q8') as VoxWariant;
  const wybranaAkustyka = (settings?.vox_akustyka ?? 'f16') as VoxAkustyka;
  const wybranyBackend = (settings?.vox_backend ?? 'vulkan') as VoxBackend;
  const [zaladowanyWariant, setZaladowanyWariant] = useState<VoxWariant | null>(null);
  const [zaladowanaAkustyka, setZaladowanaAkustyka] = useState<VoxAkustyka | null>(null);
  const [zaladowanyBackend, setZaladowanyBackend] = useState<VoxBackend | null>(null);
  const wybranyOmni = (settings?.omni_wagi ?? 'q8') as OmniWagi;
  const [zaladowanyOmni, setZaladowanyOmni] = useState<OmniWagi | null>(null);
  const [zaladowanyOmniFlash, setZaladowanyOmniFlash] = useState<boolean | null>(null);
  // Suwaki OmniVoice: puste w bazie = default upstreamu (2,0 / 32) — tyle pokazujemy.
  const persistedOmniCfg = settings?.omni_cfg ?? 2.0;
  const persistedOmniKroki = settings?.omni_kroki ?? 32;
  const [omniCfg, setOmniCfg] = useState(persistedOmniCfg);
  const [omniKroki, setOmniKroki] = useState(persistedOmniKroki);
  useEffect(() => setOmniCfg(persistedOmniCfg), [persistedOmniCfg]);
  useEffect(() => setOmniKroki(persistedOmniKroki), [persistedOmniKroki]);
  const persistedOmniTempo = settings?.omni_tempo ?? 1.0;
  const [omniTempo, setOmniTempo] = useState(persistedOmniTempo);
  useEffect(() => setOmniTempo(persistedOmniTempo), [persistedOmniTempo]);
  const omniChunkTryb = settings?.omni_chunk_tryb ?? 'brak';
  const persistedOmniChunkZnaki = settings?.omni_chunk_znaki ?? 160;
  const [omniChunkZnaki, setOmniChunkZnaki] = useState(persistedOmniChunkZnaki);
  useEffect(() => setOmniChunkZnaki(persistedOmniChunkZnaki), [persistedOmniChunkZnaki]);
  const persistedOmniSeed = settings?.omni_seed ?? null;
  const [omniSeed, setOmniSeed] = useState(persistedOmniSeed === null ? '' : String(persistedOmniSeed));
  useEffect(() => {
    setOmniSeed(persistedOmniSeed === null ? '' : String(persistedOmniSeed));
  }, [persistedOmniSeed]);
  const zapiszOmniSeed = useCallback(() => {
    const tekst = omniSeed.trim();
    const wartosc = tekst === '' ? null : Number.parseInt(tekst, 10);
    if (wartosc === persistedOmniSeed) return;
    update({ omni_seed: wartosc });
  }, [omniSeed, persistedOmniSeed, update]);
  const [omniPracuje, setOmniPracuje] = useState(false);
  const omniFlashWybrany = settings?.omni_flash ?? true;
  const omniRozjazd =
    zaladowanyOmni !== null &&
    (zaladowanyOmni !== wybranyOmni || (zaladowanyOmniFlash ?? true) !== omniFlashWybrany);
  const przeladujOmni = useCallback(async () => {
    setOmniPracuje(true);
    try {
      const s = await apiClient.reloadOmni();
      setZaladowanyOmni(s.zaladowany_omni ?? null);
      setZaladowanyOmniFlash(s.zaladowany_omni_flash ?? null);
      setZaladowanyWariant(s.zaladowany_base_lm ?? null);
      setZaladowanaAkustyka(s.zaladowany_akustyka ?? null);
      toast({
        title: `OmniVoice przeładowany — ${s.zaladowany_omni ?? '?'}${s.zaladowany_omni_flash ? ' + flash attention' : ''}`,
        description: s.ms ? `Wstał w ${(s.ms / 1000).toFixed(1)} s. Vox zszedł z karty.` : undefined,
      });
    } catch (e) {
      toast({
        title: 'Nie udało się przeładować OmniVoice',
        description: e instanceof Error ? e.message : String(e),
        variant: 'destructive',
      });
    } finally {
      setOmniPracuje(false);
    }
  }, [toast]);
  const rozladujOmni = useCallback(async () => {
    setOmniPracuje(true);
    try {
      const s = await apiClient.rozladujOmni();
      setZaladowanyOmni(s.zaladowany_omni ?? null);
      setZaladowanyOmniFlash(s.zaladowany_omni_flash ?? null);
    } catch (e) {
      toast({
        title: 'Nie udało się rozładować OmniVoice',
        description: e instanceof Error ? e.message : String(e),
        variant: 'destructive',
      });
    } finally {
      setOmniPracuje(false);
    }
  }, [toast]);
  // Ostatnio otwarta zakladka przezywa odswiezenie strony.
  const [zakladka, setZakladka] = useState(() => localStorage.getItem('ustawienia-generacji-zakladka') ?? 'ogolne');
  const zmienZakladke = useCallback((v: string) => {
    setZakladka(v);
    localStorage.setItem('ustawienia-generacji-zakladka', v);
  }, []);
  const [ttsDostepny, setTtsDostepny] = useState(true);
  const [przeladowuje, setPrzeladowuje] = useState(false);
  const doPrzeladowania =
    (zaladowanyWariant !== null && wybranyWariant !== zaladowanyWariant) ||
    (zaladowanaAkustyka !== null && wybranaAkustyka !== zaladowanaAkustyka) ||
    (zaladowanyBackend !== null && wybranyBackend !== zaladowanyBackend);

  useEffect(() => {
    apiClient
      .getTtsStan()
      .then((s) => {
        setTtsDostepny(s.dostepny);
        setZaladowanyWariant(s.zaladowany_base_lm ?? s.wariant ?? null);
        setZaladowanaAkustyka(s.zaladowany_akustyka ?? null);
        setZaladowanyBackend(s.zaladowany_backend ?? null);
        setZaladowanyOmni(s.zaladowany_omni ?? null);
      setZaladowanyOmniFlash(s.zaladowany_omni_flash ?? null);
        setZaladowanyOmniFlash(s.zaladowany_omni_flash ?? null);
      })
      .catch(() => {});
  }, []);

  const przeladujModel = useCallback(async () => {
    setPrzeladowuje(true);
    try {
      // ⛔ BEZ ARGUMENTOW. Serwer bierze wybor z BAZY (`tts_serwer.wybrane()`),
      // czyli z tego samego miejsca, ktore panel wlasnie zapisal. Podawanie
      // wartosci stad zamrazalo je w domknieciu `useCallback` i przycisk
      // wysylal AKUSTYKE SPRZED zmiany — serwer wstawal z f16, choc w panelu
      // stalo f32, i wygladalo to na „f32 sie nie da wybrac".
      const s = await apiClient.reloadTts();
      setZaladowanyWariant(s.zaladowany_base_lm ?? s.wariant ?? null);
      setZaladowanaAkustyka(s.zaladowany_akustyka ?? null);
      setZaladowanyBackend(s.zaladowany_backend ?? null);
      setZaladowanyOmni(s.zaladowany_omni ?? null);
      setZaladowanyOmniFlash(s.zaladowany_omni_flash ?? null);
      toast({
        title: `Model przeładowany — ${s.zaladowany_base_lm ?? s.wariant} / ${s.zaladowany_akustyka ?? '?'} / ${NAZWA_BACKENDU[s.zaladowany_backend ?? 'hip']}`,
        description: s.ms ? `Wstał w ${(s.ms / 1000).toFixed(1)} s.` : undefined,
      });
    } catch (e) {
      toast({
        title: 'Nie udało się przeładować modelu',
        description: e instanceof Error ? e.message : String(e),
        variant: 'destructive',
      });
    } finally {
      setPrzeladowuje(false);
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
      <Tabs value={zakladka} onValueChange={zmienZakladke} className="space-y-6">
        <TabsList>
          <TabsTrigger value="ogolne">Ogólne</TabsTrigger>
          <TabsTrigger value="vox">Vox</TabsTrigger>
          <TabsTrigger value="omni">OmniVoice</TabsTrigger>
          <TabsTrigger value="whisper">Whisper</TabsTrigger>
        </TabsList>

        <TabsContent value="ogolne" className="space-y-8">
          <WyborKartyGpu />
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
              zaladowanyBackend && zaladowanyBackend !== wybranyBackend
                ? `Która binarka audio.cpp liczy głos — osobne pliki, nieużywana nic nie zajmuje. Teraz chodzi ${NAZWA_BACKENDU[zaladowanyBackend]}.`
                : zaladowanyBackend
                  ? `Która binarka audio.cpp liczy głos — osobne pliki, nieużywana nic nie zajmuje. Chodzi: ${NAZWA_BACKENDU[zaladowanyBackend]}.`
                  : 'Która binarka audio.cpp liczy głos (Vox i OmniVoice) — osobne pliki, nieużywana nic nie zajmuje.'
            }
            action={
              <Select
                value={wybranyBackend}
                onValueChange={(v) => update({ vox_backend: v as VoxBackend })}
                disabled={!ttsDostepny || przeladowuje}
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
            description="Flash attention zawsze włączony — wszystkie miejsca uwagi idą przez FA w naszych łatkach audio.cpp."
          >
          <SettingRow
            title="Trzymanie się nawiasu (cfg)"
            description="Wyżej — model dokładniej trzyma się opisu w nawiasie, ale mówi płasko. Niżej — żywiej, z większym rozrzutem między generacjami."
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
              aria-label="Trzymanie się nawiasu"
            />
          </SettingRow>

          <SettingRow
            title="Kroki dyfuzji"
            description="Im więcej kroków, tym lepsza jakość. Każdy krok to około 0,06 s."
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
              aria-label="Kroki dyfuzji"
            />
          </SettingRow>

          <SettingRow
            title="Temperatura szumu"
            description={
              'Ile losowości model wpuszcza w dyfuzję. 1,0 to wartość silnika. Niżej — ' +
              'spokojniej i powtarzalniej, ale płasko. Wyżej — żywiej, za cenę tego, że ' +
              'ta sama linijka wychodzi za każdym razem inaczej.'
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
              aria-label="Temperatura szumu"
            />
          </SettingRow>

          <SettingRow
            title="Ziarno losowe (seed)"
            description={
              'Puste — za każdym razem inaczej. Liczba — ta sama linijka daje ten sam ' +
              'plik co do bajta, więc „wygeneruj jeszcze raz" przestaje cokolwiek zmieniać. '
            }
            action={
              <Input
                id="voxSeed"
                value={voxSeed}
                onChange={(e) => setVoxSeed(e.target.value.replace(/\D/g, '').slice(0, 10))}
                onBlur={zapiszSeed}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') e.currentTarget.blur();
                }}
                placeholder="losowo"
                inputMode="numeric"
                className="w-[140px] text-right tabular-nums"
                aria-label="Ziarno losowe"
              />
            }
          />

          <SettingRow
            title="Wagi — część językowa"
            description={
              !ttsDostepny
                ? 'BaseLM — plan wypowiedzi: słowa, akcent, intonacja. Serwer TTS nie chodzi — backend woła CLI.'
                : zaladowanyWariant && zaladowanyWariant !== wybranyWariant
                  ? `BaseLM — plan wypowiedzi: słowa, akcent, intonacja. Na karcie siedzi ${zaladowanyWariant}.`
                  : zaladowanyWariant
                    ? `BaseLM — plan wypowiedzi: słowa, akcent, intonacja. Na karcie: ${zaladowanyWariant}.`
                    : 'BaseLM — plan wypowiedzi: słowa, akcent, intonacja. Nic nie jest załadowane.'
            }
            action={
              <Select
                value={wybranyWariant}
                onValueChange={(v) => update({ vox_base_lm: v as VoxWariant })}
                disabled={!ttsDostepny || przeladowuje}
              >
                <SelectTrigger className="w-[150px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="q8">q8 — 1,6 GB</SelectItem>
                  <SelectItem value="f16">f16 — 3,0 GB</SelectItem>
                  <SelectItem value="f32">f32 — 6,0 GB</SelectItem>
                </SelectContent>
              </Select>
            }
          />

          <SettingRow
            title="Wagi — część foniczna"
            description={
              zaladowanaAkustyka && zaladowanaAkustyka !== wybranaAkustyka
                ? `DiT, LocEnc, ResidualLM — detal brzmienia. Głowica (AudioVAE) zawsze f32. Na karcie siedzi ${zaladowanaAkustyka}.`
                : zaladowanaAkustyka
                  ? `DiT, LocEnc, ResidualLM — detal brzmienia. Głowica (AudioVAE) zawsze f32. Na karcie: ${zaladowanaAkustyka}.`
                  : 'DiT, LocEnc, ResidualLM — detal brzmienia. Głowica (AudioVAE) zawsze f32.'
            }
            action={
              <div className="flex items-center gap-2">
                <Select
                  value={wybranaAkustyka}
                  onValueChange={(v) => update({ vox_akustyka: v as VoxAkustyka })}
                  disabled={!ttsDostepny || przeladowuje}
                >
                  <SelectTrigger className="w-[150px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="f16">f16 — 1,6 GB</SelectItem>
                    <SelectItem value="f32">f32 — 3,3 GB</SelectItem>
                  </SelectContent>
                </Select>
                <Button
                  variant={doPrzeladowania ? 'default' : 'outline'}
                  size="sm"
                  onClick={przeladujModel}
                  disabled={!ttsDostepny || przeladowuje}
                >
                  {przeladowuje ? (
                    <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                  ) : (
                    <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
                  )}
                  {przeladowuje ? 'Ładuję…' : 'Przeładuj'}
                </Button>
              </div>
            }
          />

          </SettingSection>
        </TabsContent>

        <TabsContent value="omni" className="space-y-8">
          <SettingSection
            title="OmniVoice"
            description="Ten sam silnik audio.cpp co Vox, jeden model naraz — OmniVoice wchodzi przy pierwszej wypowiedzi nim, Vox wtedy schodzi."
          >
          <SettingRow
            title="Wagi"
            description={
              zaladowanyOmni && zaladowanyOmni !== wybranyOmni
                ? `Ten sam silnik co Vox, jeden model naraz — OmniVoice wchodzi przy pierwszej wypowiedzi nim, Vox wtedy schodzi. Na karcie siedzi ${zaladowanyOmni}, nowy wejdzie przy następnej.`
                : zaladowanyOmni
                  ? `Ten sam silnik co Vox, jeden model naraz — OmniVoice wchodzi przy pierwszej wypowiedzi nim, Vox wtedy schodzi. Na karcie: ${zaladowanyOmni}.`
                  : 'Ten sam silnik co Vox, jeden model naraz — OmniVoice wchodzi przy pierwszej wypowiedzi nim, Vox wtedy schodzi.'
            }
            action={
              <Select
                value={wybranyOmni}
                onValueChange={(v) => update({ omni_wagi: v as OmniWagi })}
                disabled={!ttsDostepny}
              >
                <SelectTrigger className="w-[150px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="q8">q8 — 1,3 GB</SelectItem>
                  <SelectItem value="f16">f16 — 1,6 GB</SelectItem>
                </SelectContent>
              </Select>
            }
          />


          <SettingRow
            title="Flash attention"
            description={
              'Uwaga liczona kafelkami zamiast pełnej macierzy — bez tego OmniVoice trzyma macierz ' +
              'T×T (próbka + tekst + mowa, razy dwa przez cfg) i VRAM rośnie z najdłuższą wypowiedzią. ' +
              'Upstream: „mały dryf wyjścia”. Zmiana wchodzi przy następnym załadowaniu OmniVoice.' +
              (zaladowanyOmniFlash !== null ? ` Teraz: ${zaladowanyOmniFlash ? 'włączony' : 'wyłączony'}.` : '')
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
            title="Trzymanie się opisu (guidance)"
            description="To samo co cfg u Voxa: wyżej — dokładniej, ale płasko; niżej — żywiej. Domyślne upstreamu: 2,0."
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
              aria-label="Trzymanie się opisu"
            />
          </SettingRow>

          <SettingRow
            title="Kroki"
            description="Rundy dopracowywania CAŁEJ wypowiedzi naraz (nie krok na kawałek jak u Voxa). Domyślne upstreamu: 32."
            action={
              <span className="text-sm tabular-nums text-muted-foreground">{omniKroki}</span>
            }
          >
            <Slider
              id="omniKroki"
              value={[omniKroki]}
              onValueChange={([value]) => setOmniKroki(value)}
              onValueCommit={([value]) => update({ omni_kroki: value })}
              min={4}
              max={64}
              step={1}
              aria-label="Kroki OmniVoice"
            />
          </SettingRow>

          <SettingRow
            title="Tempo mowy"
            description="Mnożnik przy szacowaniu długości wypowiedzi — OmniVoice zgaduje ją z tempa próbki, więc próbka z pauzami daje mowę za wolną. Wyżej = szybciej. Domyślne upstreamu: 1,0."
            action={
              <span className="text-sm tabular-nums text-muted-foreground">
                {omniTempo.toFixed(2)}×
              </span>
            }
          >
            <Slider
              id="omniTempo"
              value={[omniTempo]}
              onValueChange={([value]) => setOmniTempo(value)}
              onValueCommit={([value]) => update({ omni_tempo: value })}
              min={0.5}
              max={2}
              step={0.05}
              aria-label="Tempo mowy OmniVoice"
            />
          </SettingRow>

          <SettingRow
            title="Strumień kawałkami"
            description={
              'OmniVoice liczy całą wypowiedź naraz, więc dźwięk przychodzi dopiero po wszystkim. ' +
              'Pocięty tekst gra od pierwszego kawałka, a kolejne liczą się w trakcie (jak --stream-by-line ' +
              'w omnivoice.cpp). Wyłączone — upstream tnie sam dopiero powyżej 30 s mowy.'
            }
            action={
              <Select
                value={omniChunkTryb}
                onValueChange={(v) => update({ omni_chunk_tryb: v as 'brak' | 'linie' | 'zdania' })}
              >
                <SelectTrigger className="w-[150px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="brak">wyłączone</SelectItem>
                  <SelectItem value="zdania">po zdaniach</SelectItem>
                  <SelectItem value="linie">po liniach</SelectItem>
                </SelectContent>
              </Select>
            }
          />

          {omniChunkTryb !== 'brak' && (
            <SettingRow
              title="Maks. znaków w kawałku"
              description={
                omniChunkTryb === 'zdania'
                  ? 'Zdania pakowane do tego limitu — mniej = wcześniejszy pierwszy dźwięk, ale więcej szwów. Domyślne 160 (przykład strumienia z docs upstreamu).'
                  : 'Linia kończąca się kropką/pytajnikiem to kawałek; dłuższe od limitu tnie dalej. Domyślne 160 (przykład strumienia z docs upstreamu).'
              }
              action={
                <span className="text-sm tabular-nums text-muted-foreground">{omniChunkZnaki}</span>
              }
            >
              <Slider
                id="omniChunkZnaki"
                value={[omniChunkZnaki]}
                onValueChange={([value]) => setOmniChunkZnaki(value)}
                onValueCommit={([value]) => update({ omni_chunk_znaki: value })}
                min={20}
                max={1000}
                step={10}
                aria-label="Maks. znaków w kawałku"
              />
            </SettingRow>
          )}

          <SettingRow
            title="Ziarno losowe (seed)"
            description="Puste — za każdym razem inaczej. Liczba — ten sam tekst daje ten sam plik."
            action={
              <Input
                id="omniSeed"
                value={omniSeed}
                onChange={(e) => setOmniSeed(e.target.value.replace(/\D/g, '').slice(0, 10))}
                onBlur={zapiszOmniSeed}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') e.currentTarget.blur();
                }}
                placeholder="losowo"
                className="w-[150px]"
              />
            }
          />

          <SettingRow
            title="Stan"
            description={
              zaladowanyOmni
                ? `Na karcie: OmniVoice ${zaladowanyOmni}${zaladowanyOmniFlash ? ' + flash attention' : ''}.` +
                  (omniRozjazd ? ' Wybór się różni — przeładuj albo zrobi to następna wypowiedź OmniVoice.' : '')
                : 'Nie jest załadowany — wejdzie przy pierwszej wypowiedzi OmniVoice (Vox wtedy schodzi).'
            }
            action={
              <div className="flex items-center gap-2">
                <Button
                  variant={omniRozjazd ? 'default' : 'outline'}
                  size="sm"
                  onClick={przeladujOmni}
                  disabled={!ttsDostepny || omniPracuje}
                >
                  {omniPracuje ? (
                    <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                  ) : (
                    <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
                  )}
                  {omniPracuje ? 'Pracuję…' : zaladowanyOmni ? 'Przeładuj' : 'Załaduj'}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={rozladujOmni}
                  disabled={omniPracuje || !zaladowanyOmni}
                >
                  <Unplug className="h-3.5 w-3.5 mr-1.5" />
                  Rozładuj
                </Button>
              </div>
            }
          />
          </SettingSection>
        </TabsContent>

        <TabsContent value="whisper" className="space-y-8">
          <WhisperUstawienia />
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
