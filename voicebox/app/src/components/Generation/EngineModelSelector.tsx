import { useEffect } from 'react';
import type { UseFormReturn } from 'react-hook-form';
import { FormControl } from '@/components/ui/form';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { TtsEngine, VoiceProfileResponse } from '@/lib/api/types';
import { getLanguageOptionsForEngine } from '@/lib/constants/languages';
import type { GenerationFormValues } from '@/lib/hooks/useGenerationForm';

/**
 * The engines of this build and their display metadata.
 * Adding a new engine means adding one entry here.
 */
const ENGINE_OPTIONS: readonly { value: TtsEngine; label: string }[] = [
  { value: 'voxcpm', label: 'VoxCPM2' },
  { value: 'omnivoice', label: 'OmniVoice' },
  { value: 'voxcpm_pt', label: 'VoxCPM2 PyTorch' },
];

const ENGINE_DESCRIPTIONS: Record<TtsEngine, string> = {
  voxcpm: '48 kHz cloning, (style prefix) control — audio.cpp',
  omnivoice: '24 kHz cloning, 600+ languages — same audio.cpp engine as VoxCPM2, one at a time',
  voxcpm_pt: 'Same weights as VoxCPM2, PyTorch instead of ggml',
};

/** Every engine of this build clones from reference audio. */
const CLONING_ENGINES = new Set<string>(ENGINE_OPTIONS.map((o) => o.value));

function getAvailableOptions(selectedProfile?: VoiceProfileResponse | null) {
  if (!selectedProfile) return ENGINE_OPTIONS;
  return ENGINE_OPTIONS.filter((opt) => isProfileCompatibleWithEngine(selectedProfile, opt.value));
}

export function applyEngineSelection(form: UseFormReturn<GenerationFormValues>, value: string) {
  form.setValue('engine', value as TtsEngine);
  // If the current language isn't supported by the new engine, reset to the first available
  const currentLang = form.getValues('language');
  const available = getLanguageOptionsForEngine(value);
  if (!available.some((l) => l.value === currentLang)) {
    form.setValue('language', available[0]?.value ?? 'en');
  }
}

interface EngineModelSelectorProps {
  form: UseFormReturn<GenerationFormValues>;
  compact?: boolean;
  selectedProfile?: VoiceProfileResponse | null;
}

export function EngineModelSelector({ form, compact, selectedProfile }: EngineModelSelectorProps) {
  const engine = form.watch('engine') || 'voxcpm';
  const availableOptions = getAvailableOptions(selectedProfile);

  const currentEngineAvailable = availableOptions.some((opt) => opt.value === engine);

  useEffect(() => {
    if (!currentEngineAvailable && availableOptions.length > 0) {
      applyEngineSelection(form, availableOptions[0].value);
    }
  }, [availableOptions, currentEngineAvailable, form]);

  const itemClass = compact ? 'text-xs text-muted-foreground' : undefined;
  const triggerClass = compact
    ? 'h-8 text-xs bg-card border-border rounded-full hover:bg-background/50 transition-all'
    : undefined;

  return (
    <Select value={engine} onValueChange={(v) => applyEngineSelection(form, v)}>
      <FormControl>
        <SelectTrigger className={triggerClass}>
          <SelectValue />
        </SelectTrigger>
      </FormControl>
      <SelectContent side={compact ? 'top' : undefined}>
        {availableOptions.map((opt) => (
          <SelectItem key={opt.value} value={opt.value} className={itemClass}>
            {opt.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

/** Returns a human-readable description for the currently selected engine. */
export function getEngineDescription(engine: string): string {
  return ENGINE_DESCRIPTIONS[engine as TtsEngine] ?? '';
}

/**
 * Check if a profile is compatible with the currently selected engine.
 * Useful for UI hints.
 */
export function isProfileCompatibleWithEngine(
  profile: VoiceProfileResponse,
  engine: string,
): boolean {
  const voiceType = profile.voice_type || 'cloned';
  if (voiceType === 'preset') return profile.preset_engine === engine;
  if (voiceType === 'cloned') return CLONING_ENGINES.has(engine);
  return true; // designed — future
}
