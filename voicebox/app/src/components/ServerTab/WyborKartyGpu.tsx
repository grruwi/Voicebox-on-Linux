import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { AlertTriangle, Check, Loader2 } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { apiClient } from '@/lib/api/client';
import type { StanKarty } from '@/lib/api/types';
import { useServerStore } from '@/stores/serverStore';
import { SettingRow, SettingSection } from './SettingRow';

/**
 * Wybor karty, na ktorej maja siedziec modele.
 *
 * ⛔ Wybor jest TWARDY, nie "preferencja": niewybrana karta staje sie
 * NIEWIDZIALNA naraz dla ROCm i dla Vulkana, wiec nic nie ma jak sie na nia
 * rozlac. Bez tego ggml rozklada jeden model po obu kartach — zmierzone 02.09:
 * 5728 MiB na jednej i 2545 MiB na drugiej w JEDNYM procesie.
 *
 * ⚠️ Wchodzi po restarcie Voiceboxa. Zmienne widocznosci czyta sie przy starcie
 * procesu i nie da sie ich przestawic w locie.
 */
export function WyborKartyGpu() {
  const serverUrl = useServerStore((s) => s.serverUrl);
  const queryClient = useQueryClient();
  const [blad, setBlad] = useState<string | null>(null);

  const { data, isLoading } = useQuery<StanKarty>({
    queryKey: ['karta-gpu', serverUrl],
    queryFn: () => apiClient.getKarta(),
    retry: 1,
  });

  const zapisz = useMutation({
    mutationFn: (indeks: number | null) => apiClient.setKarta(indeks),
    onMutate: () => setBlad(null),
    onSuccess: (nowy) => queryClient.setQueryData(['karta-gpu', serverUrl], nowy),
    onError: (e: unknown) => setBlad(e instanceof Error ? e.message : 'nie udało się zapisać'),
  });

  // Jedna karta w maszynie = nie ma czego wybierac. Sekcja znika calkowicie,
  // zeby nie sugerowac decyzji, ktorej nie ma.
  if (isLoading || !data || data.karty.length < 2) return null;

  const opis = (indeks: number | null) =>
    indeks === null ? 'bez ograniczenia' : `karta ${indeks}`;

  return (
    <SettingSection
      title="Karta dla modeli"
      description="Na której karcie mają siedzieć modele TTS i STT. Wybrana karta zostaje jedyną widoczną — reszta jest odcięta i nic się na nią nie rozleje."
    >
      {data.wymaga_restartu && (
        <SettingRow title="Zmiana czeka na restart">
          <div className="flex items-start gap-2 text-sm text-amber-500">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
            <span>
              Po restarcie silniki dostaną <strong>{opis(data.rozwiazana)}</strong> ({data.powod}),
              a działający Voicebox widzi jeszcze <strong>{opis(data.aktywna)}</strong>. Wejdzie po{' '}
              <code className="text-xs">voicebox down &amp;&amp; voicebox up</code>.
            </span>
          </div>
        </SettingRow>
      )}

      {data.karty.map((k) => {
        const wybrana = data.wybrana === k.indeks;
        const aktywna = data.aktywna === k.indeks;
        // Zapis wskazuje kartę, której tu nie ma — launcher spadł na tę.
        const zastepcza = !wybrana && data.rozwiazana === k.indeks;
        return (
          <SettingRow
            key={k.pci}
            title={k.nazwa}
            description={`${k.pci} · ${(k.vram_mib / 1024).toFixed(0)} GB${aktywna ? ' · używana teraz' : ''}${zastepcza ? ' · ta pójdzie po restarcie' : ''}`}
            action={
              <Button
                size="sm"
                variant={wybrana ? 'default' : 'outline'}
                disabled={zapisz.isPending}
                onClick={() => zapisz.mutate(k.indeks)}
              >
                {zapisz.isPending && zapisz.variables === k.indeks ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : wybrana ? (
                  <>
                    <Check className="h-3.5 w-3.5 mr-1.5" />
                    wybrana
                  </>
                ) : (
                  'wybierz'
                )}
              </Button>
            }
          />
        );
      })}

      <SettingRow
        title="Bez ograniczenia"
        description="Obie karty widoczne. ⛔ Wtedy ggml dzieli model na połówki i wozi dane przez PCIe — wolniej i zjada VRAM po obu stronach."
        action={
          <Button
            size="sm"
            variant={data.wybrana === null ? 'default' : 'ghost'}
            disabled={zapisz.isPending}
            onClick={() => zapisz.mutate(null)}
          >
            {data.wybrana === null ? (
              <>
                <Check className="h-3.5 w-3.5 mr-1.5" />
                ustawione
              </>
            ) : (
              'zdejmij'
            )}
          </Button>
        }
      />

      {blad && (
        <SettingRow title="Błąd">
          <span className="text-sm text-destructive">{blad}</span>
        </SettingRow>
      )}
    </SettingSection>
  );
}
