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
import type { WhisperKwant, WhisperModel, WhisperStan, WhisperUrzadzenie } from '@/lib/api/types';
import { useGenerationSettings } from '@/lib/hooks/useSettings';
import { SettingRow, SettingSection } from './SettingRow';

const NAZWY: Record<WhisperModel, string> = {
  medium: 'Medium',
  large: 'Large v3',
  turbo: 'Large v3 Turbo',
};
const URZADZENIA: Record<WhisperUrzadzenie, string> = {
  gpu: 'karcie',
  cpu: 'procesorze',
};

/**
 * Whisper (whisper.cpp) — osobny proces, osobny klik od głosu.
 *
 * Wybór idzie DO BAZY jak wagi voxa i to z niego korzysta dyktafon
 * (`/transcribe` bez `model`). `zaladowany_*` to stan procesu z /proc —
 * rozjeżdża się z wyborem do chwili przeładowania i to ma być widać.
 */
export function WhisperUstawienia() {
  const { settings, update } = useGenerationSettings();
  const { toast } = useToast();
  const model = (settings?.whisper_model ?? 'large') as WhisperModel;
  const kwant = (settings?.whisper_kwant ?? 'q5') as WhisperKwant;
  const urzadzenie = (settings?.whisper_urzadzenie ?? 'gpu') as WhisperUrzadzenie;
  const przyStarcie = settings?.whisper_przy_starcie ?? false;
  const flash = settings?.whisper_fa ?? true;
  const [stan, setStan] = useState<WhisperStan | null>(null);
  const [pracuje, setPracuje] = useState(false);

  useEffect(() => {
    apiClient.getWhisperStan().then(setStan).catch(() => {});
  }, []);

  const rozmiar = (m: WhisperModel, k: WhisperKwant) => {
    const p = stan?.pliki.find((x) => x.model === m && x.kwant === k);
    if (!p) return '';
    return p.jest_na_dysku ? ` — ${(p.rozmiar_mib / 1024).toFixed(1).replace('.', ',')} GB` : ' — brak pliku';
  };

  const zaladowany = stan?.chodzi && stan.zaladowany_model;
  const rozjazd =
    !!zaladowany &&
    (stan?.zaladowany_model !== model ||
      stan?.zaladowany_kwant !== kwant ||
      stan?.zaladowane_urzadzenie !== urzadzenie ||
      (stan?.zaladowany_fa ?? true) !== flash);

  const zaladuj = useCallback(async () => {
    setPracuje(true);
    try {
      // ⛔ BEZ ARGUMENTÓW — serwer bierze wybór z BAZY, tak jak przeładowanie voxa.
      const s = await apiClient.zaladujWhisper();
      setStan(s);
      toast({
        title: `Whisper załadowany — ${s.zaladowany_model} ${s.zaladowany_kwant} na ${URZADZENIA[s.zaladowane_urzadzenie ?? 'gpu']}`,
        description: s.ms ? `Wstał w ${(s.ms / 1000).toFixed(1)} s.` : undefined,
      });
    } catch (e) {
      toast({
        title: 'Nie udało się załadować Whispera',
        description: e instanceof Error ? e.message : String(e),
        variant: 'destructive',
      });
    } finally {
      setPracuje(false);
    }
  }, [toast]);

  const rozladuj = useCallback(async () => {
    setPracuje(true);
    try {
      setStan(await apiClient.rozladujWhisper());
    } catch (e) {
      toast({
        title: 'Nie udało się rozładować Whispera',
        description: e instanceof Error ? e.message : String(e),
        variant: 'destructive',
      });
    } finally {
      setPracuje(false);
    }
  }, [toast]);

  return (
    <SettingSection
      title="Whisper — rozpoznawanie mowy"
      description="whisper.cpp na tym samym backendzie co głos (ggml, ROCm). Osobny proces: ładuje się i zdejmuje niezależnie od modelu głosu. Dyktafon bierze to, co wybrane tutaj."
    >
      <SettingRow
        title="Model"
        description="Turbo to Large v3 z odchudzonym dekoderem — szybszy, prawie ta sama jakość."
        action={
          <Select
            value={model}
            onValueChange={(v) => update({ whisper_model: v as WhisperModel })}
            disabled={pracuje}
          >
            <SelectTrigger className="w-[190px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(Object.keys(NAZWY) as WhisperModel[]).map((m) => (
                <SelectItem key={m} value={m}>
                  {NAZWY[m]}
                  {rozmiar(m, kwant)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        }
      />

      <SettingRow
        title="Kwantyzacja"
        description="q5 — mniejszy i szybszy. q8 — bliżej pełnych wag."
        action={
          <Select
            value={kwant}
            onValueChange={(v) => update({ whisper_kwant: v as WhisperKwant })}
            disabled={pracuje}
          >
            <SelectTrigger className="w-[190px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="q5">q5{rozmiar(model, 'q5')}</SelectItem>
              <SelectItem value="q8">q8{rozmiar(model, 'q8')}</SelectItem>
            </SelectContent>
          </Select>
        }
      />

      <SettingRow
        title="Urządzenie"
        description="Procesor zostawia VRAM głosowi. W kontenerze ma rdzenie 6, 7, 14, 15 — Large będzie tam wolny."
        action={
          <Select
            value={urzadzenie}
            onValueChange={(v) => update({ whisper_urzadzenie: v as WhisperUrzadzenie })}
            disabled={pracuje}
          >
            <SelectTrigger className="w-[190px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="gpu">Karta (HIP)</SelectItem>
              <SelectItem value="cpu">Procesor</SelectItem>
            </SelectContent>
          </Select>
        }
      />

      <SettingRow
        title="Flash attention"
        description={
          'Uwaga liczona kafelkami — szybciej i mniej VRAM-u; whisper.cpp ma go domyślnie włączonego. ' +
          'Zmiana wchodzi po przeładowaniu Whispera.'
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
        title="Ładuj przy starcie Voiceboxa"
        description="Włączone — backend sam podnosi wybrany model po każdym starcie (w tle). Wyłączone — czeka na klik albo pierwsze dyktowanie z GUI."
        htmlFor="whisperPrzyStarcie"
        action={
          <Toggle
            id="whisperPrzyStarcie"
            checked={przyStarcie}
            onCheckedChange={(v) => update({ whisper_przy_starcie: v })}
          />
        }
      />

      <SettingRow
        title="Stan"
        description={
          zaladowany
            ? `Na ${URZADZENIA[stan?.zaladowane_urzadzenie ?? 'gpu']}: ${NAZWY[stan!.zaladowany_model!]} ${stan?.zaladowany_kwant}.` +
              (rozjazd ? ' Wybór się różni — przeładuj albo zrobi to następne dyktowanie.' : '')
            : 'Nic nie jest załadowane. Pierwsze dyktowanie załaduje wybrany model samo.'
        }
        action={
          <div className="flex items-center gap-2">
            <Button
              variant={rozjazd || !zaladowany ? 'default' : 'outline'}
              size="sm"
              onClick={zaladuj}
              disabled={pracuje}
            >
              {pracuje ? (
                <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
              ) : (
                <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
              )}
              {pracuje ? 'Pracuję…' : zaladowany ? 'Przeładuj' : 'Załaduj'}
            </Button>
            <Button variant="outline" size="sm" onClick={rozladuj} disabled={pracuje || !zaladowany}>
              <Unplug className="h-3.5 w-3.5 mr-1.5" />
              Rozładuj
            </Button>
          </div>
        }
      />
    </SettingSection>
  );
}
