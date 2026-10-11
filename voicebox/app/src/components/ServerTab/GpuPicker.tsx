import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { AlertTriangle, Check, Loader2 } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { apiClient } from '@/lib/api/client';
import type { GpuState } from '@/lib/api/types';
import { useServerStore } from '@/stores/serverStore';
import { SettingRow, SettingSection } from './SettingRow';

/**
 * The card the models run on.
 *
 * ⛔ The choice is HARD, not a "preference": the other card becomes INVISIBLE to ROCm
 * and to Vulkan at once, so nothing can spill onto it. Without it ggml spreads one
 * model across both cards — measured: 5728 MiB on one and 2545 MiB on the other in
 * ONE process.
 *
 * It takes effect with the next model load: the engine containers start every model
 * with the visibility variables of the chosen card.
 */
export function GpuPicker() {
  const serverUrl = useServerStore((s) => s.serverUrl);
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | null>(null);

  const { data, isLoading } = useQuery<GpuState>({
    queryKey: ['gpu', serverUrl],
    queryFn: () => apiClient.getGpu(),
    retry: 1,
  });

  const save = useMutation({
    mutationFn: (index: number | null) => apiClient.setGpu(index),
    onMutate: () => setError(null),
    onSuccess: (state) => queryClient.setQueryData(['gpu', serverUrl], state),
    onError: (e: unknown) => setError(e instanceof Error ? e.message : 'could not save'),
  });

  // One card in the machine = nothing to choose. The section disappears completely,
  // so as not to suggest a decision that does not exist.
  if (isLoading || !data || data.cards.length < 2) return null;

  const selected = data.cards.find((c) => c.pci === data.selected_pci) ?? null;
  // The file points at a card that is not here (pulled out, another machine).
  const missing = data.selected_pci !== null && selected === null;

  return (
    <SettingSection
      title="GPU for the models"
      description="Which card the TTS and STT models run on. The chosen card becomes the only visible one — the rest is cut off and nothing spills onto it. Takes effect with the next model load."
    >
      {missing && (
        <SettingRow title="The chosen card is not here">
          <div className="flex items-start gap-2 text-sm text-amber-500">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
            <span>
              Saved: <code className="text-xs">{data.selected_pci}</code>. The engines get{' '}
              <strong>{data.resolved === null ? 'every card' : `card ${data.resolved}`}</strong> ({data.reason}).
            </span>
          </div>
        </SettingRow>
      )}

      {data.cards.map((card) => {
        const isSelected = selected?.index === card.index;
        return (
          <SettingRow
            key={card.pci}
            title={card.name}
            description={`${card.pci} · ${(card.vram_mib / 1024).toFixed(0)} GB`}
            action={
              <Button
                size="sm"
                variant={isSelected ? 'default' : 'outline'}
                disabled={save.isPending}
                onClick={() => save.mutate(card.index)}
              >
                {save.isPending && save.variables === card.index ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : isSelected ? (
                  <>
                    <Check className="h-3.5 w-3.5 mr-1.5" />
                    selected
                  </>
                ) : (
                  'select'
                )}
              </Button>
            }
          />
        );
      })}

      <SettingRow
        title="No restriction"
        description="Every card visible. ⛔ ggml then splits a model in halves and ferries data over PCIe — slower, and it eats VRAM on both sides."
        action={
          <Button
            size="sm"
            variant={data.selected_pci === null ? 'default' : 'ghost'}
            disabled={save.isPending}
            onClick={() => save.mutate(null)}
          >
            {data.selected_pci === null ? (
              <>
                <Check className="h-3.5 w-3.5 mr-1.5" />
                set
              </>
            ) : (
              'lift'
            )}
          </Button>
        }
      />

      {error && (
        <SettingRow title="Error">
          <span className="text-sm text-destructive">{error}</span>
        </SettingRow>
      )}
    </SettingSection>
  );
}
