import { Cpu } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { HealthResponse } from '@/lib/api/types';
import { useServerHealth } from '@/lib/hooks/useServer';
import { SettingSection } from './SettingRow';

function AppleLogo({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M18.71 19.5c-.83 1.24-1.71 2.45-3.05 2.47-1.34.03-1.77-.79-3.29-.79-1.53 0-2 .77-3.27.82-1.31.05-2.3-1.32-3.14-2.53C4.25 17 2.94 12.45 4.7 9.39c.87-1.52 2.43-2.48 4.12-2.51 1.28-.02 2.5.87 3.29.87.78 0 2.26-1.07 3.8-.91.65.03 2.47.26 3.64 1.98-.09.06-2.17 1.28-2.15 3.81.03 3.02 2.65 4.03 2.68 4.04-.03.07-.42 1.44-1.38 2.83M13 3.5c.73-.83 1.94-1.46 2.94-1.5.13 1.17-.34 2.35-1.04 3.19-.69.85-1.83 1.51-2.95 1.42-.15-1.15.41-2.35 1.05-3.11z" />
    </svg>
  );
}

function GpuIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="4" y="6" width="16" height="12" rx="2" />
      <path d="M2 10h2M2 14h2M20 10h2M20 14h2" />
      <path d="M9 10h6M9 14h4" />
    </svg>
  );
}

function GpuInfoCard({ health }: { health: HealthResponse }) {
  const { t } = useTranslation();
  const hasGpu = health.gpu_available && health.gpu_type;

  const gpuName = hasGpu
    ? health.gpu_type!.replace(/^(CUDA|ROCm|MPS|Metal|XPU|DirectML)\s*\((.+)\)$/, '$2') ||
      health.gpu_type!
    : null;
  const gpuBackend = hasGpu ? health.gpu_type!.replace(/\s*\(.+\)$/, '') : null;
  const isApple = gpuBackend === 'MPS' || gpuBackend === 'Metal';
  const showBackendVariant = 
    health.backend_variant && 
    health.backend_variant !== 'cpu' &&
    health.backend_variant.toLowerCase() !== gpuBackend?.toLowerCase();

  return (
    <div className="rounded-lg border border-border/60 p-4">
      <div className="flex items-center gap-3">
        {hasGpu ? (
          isApple ? (
            <AppleLogo className="h-5 w-5 shrink-0 text-muted-foreground" />
          ) : (
            <GpuIcon className="h-5 w-5 shrink-0 text-accent" />
          )
        ) : (
          <Cpu className="h-5 w-5 shrink-0 text-muted-foreground" />
        )}
        <div className="flex-1 min-w-0 space-y-0.5">
          <div className="text-sm font-medium">{hasGpu ? gpuName : t('settings.gpu.cpuOnly')}</div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
            {hasGpu ? (
              <>
                <span>{gpuBackend}</span>
                {showBackendVariant && (
                  <>
                    <span className="text-border">|</span>
                    <span className="uppercase">{health.backend_variant}</span>
                  </>
                )}
                {health.vram_used_mb != null && health.vram_used_mb > 0 && (
                  <>
                    <span className="text-border">|</span>
                    <span>
                      {t('settings.gpu.vramUsed', { mb: health.vram_used_mb.toFixed(0) })}
                    </span>
                  </>
                )}
              </>
            ) : (
              <span>{t('settings.gpu.noAcceleration')}</span>
            )}
          </div>
        </div>
        {hasGpu && (
          <div className="flex items-center gap-2 rounded-full border border-accent/30 px-2.5 py-0.5">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent/60" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-accent shadow-[0_0_4px_1px_hsl(var(--accent)/0.4)]" />
            </span>
            <span className="text-[10px] font-medium text-muted-foreground">
              {t('settings.gpu.active')}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export function GpuPage() {
  const { t } = useTranslation();
  const { data: health } = useServerHealth();

  if (!health) return null;

  return (
    <div className="space-y-8 max-w-2xl">
      <GpuInfoCard health={health} />

      <SettingSection
        title="GPU backends"
        description="The engines run in their own containers, each with its own GPU build: audio.cpp (Vulkan or ROCm, switched in Settings → Generation) and PyTorch (ROCm). Voicebox downloads no GPU backends of its own. The card for the models is chosen in Settings → Generation."
      >
        {null}
      </SettingSection>

      <p className="text-xs text-muted-foreground/60 leading-relaxed">{t('settings.gpu.footer')}</p>
    </div>
  );
}
