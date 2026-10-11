import { AudioWaveform } from 'lucide-react';
import { Button } from '@/components/ui/button';

/** Guzik obok Whispera przy probce: otwiera edytor probki (11.10). */
export interface EdytorProps {
  onEdit?: () => void;
}

export function PrzyciskEdytor({ onEdit, disabled = false }: EdytorProps & { disabled?: boolean }) {
  if (!onEdit) return null;
  return (
    <Button
      type="button"
      variant="outline"
      onClick={onEdit}
      disabled={disabled}
      className="flex items-center gap-2"
      title="Edytor próbki: cięcie, pętla, A/B, cofanie"
    >
      <AudioWaveform className="h-4 w-4" />
      Edytuj
    </Button>
  );
}
