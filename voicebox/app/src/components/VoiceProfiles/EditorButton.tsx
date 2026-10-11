import { AudioWaveform } from 'lucide-react';
import { Button } from '@/components/ui/button';

/** The button next to Whisper by a sample: opens the sample editor. */
export interface EditorProps {
  onEdit?: () => void;
}

export function EditorButton({ onEdit, disabled = false }: EditorProps & { disabled?: boolean }) {
  if (!onEdit) return null;
  return (
    <Button
      type="button"
      variant="outline"
      onClick={onEdit}
      disabled={disabled}
      className="flex items-center gap-2"
      title="Sample editor: cut, loop, A/B, undo"
    >
      <AudioWaveform className="h-4 w-4" />
      Edit
    </Button>
  );
}
