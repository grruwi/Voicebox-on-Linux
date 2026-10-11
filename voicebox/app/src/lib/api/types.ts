// API Types matching backend Pydantic models
import type { LanguageCode } from '@/lib/constants/languages';

export type VoiceType = 'cloned' | 'preset' | 'designed';

export interface VoiceProfileCreate {
  name: string;
  description?: string;
  language: LanguageCode;
  voice_type?: VoiceType;
  preset_engine?: string;
  preset_voice_id?: string;
  design_prompt?: string;
  default_engine?: string;
  /** Free-form character prompt used by compose and the `/generate` personality-rewrite path. */
  personality?: string;
}

export interface VoiceProfileResponse {
  id: string;
  name: string;
  description?: string;
  language: string;
  avatar_path?: string;
  effects_chain?: EffectConfig[];
  voice_type: VoiceType;
  preset_engine?: string;
  preset_voice_id?: string;
  design_prompt?: string;
  default_engine?: string;
  personality?: string | null;
  generation_count: number;
  sample_count: number;
  created_at: string;
  updated_at: string;
}

/** Response returned by /profiles/{id}/compose. */
export interface PersonalityTextResponse {
  text: string;
  model_size: string;
}

export interface PresetVoice {
  voice_id: string;
  name: string;
  gender: 'male' | 'female';
  language: string;
}

export interface ProfileSampleCreate {
  reference_text: string;
}

export interface ProfileSampleResponse {
  id: string;
  profile_id: string;
  audio_path: string;
  reference_text: string;
}

export interface EffectConfig {
  type: string;
  enabled: boolean;
  params: Record<string, number>;
}

export interface GenerationRequest {
  profile_id: string;
  text: string;
  language: LanguageCode;
  seed?: number;
  model_size?: '1.7B' | '0.6B' | '1B' | '3B';
  engine?: TtsEngine;
  /** Built-in voice of a preset engine; overrides the one stored in the profile. */
  preset_voice_id?: string;
  instruct?: string;
  /** When true and the profile has a personality prompt, input text is rewritten in-character before TTS. */
  personality?: boolean;
  max_chunk_chars?: number;
  crossfade_ms?: number;
  normalize?: boolean;
  effects_chain?: EffectConfig[];
}

export interface GenerationVersionResponse {
  id: string;
  generation_id: string;
  label: string;
  audio_path: string;
  effects_chain?: EffectConfig[];
  source_version_id?: string;
  is_default: boolean;
  created_at: string;
}

export interface GenerationResponse {
  id: string;
  profile_id: string;
  text: string;
  language: string;
  audio_path?: string;
  duration?: number;
  seed?: number;
  instruct?: string;
  engine?: string;
  model_size?: string;
  status: 'loading_model' | 'generating' | 'completed' | 'failed';
  error?: string;
  is_favorited?: boolean;
  created_at: string;
  versions?: GenerationVersionResponse[];
  active_version_id?: string;
}

export interface HistoryQuery {
  profile_id?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

export interface HistoryResponse extends GenerationResponse {
  profile_name: string;
  versions?: GenerationVersionResponse[];
  active_version_id?: string;
}

export interface HistoryListResponse {
  items: HistoryResponse[];
  total: number;
}

export type WhisperModelSize = 'base' | 'small' | 'medium' | 'large' | 'turbo';

export type Qwen3ModelSize = '0.6B' | '1.7B' | '4B';

export type CaptureSource = 'dictation' | 'recording' | 'file';

/**
 * Snapshot of the accessibility-focused UI element at chord-start. Emitted
 * from Rust as part of the ``dictate:start`` payload so the frontend can
 * pass it back to ``paste_final_text`` once the final text is ready.
 */
export interface FocusSnapshot {
  pid: number;
  bundle_id: string | null;
  role: string | null;
}

export interface RefinementFlags {
  smart_cleanup: boolean;
  self_correction: boolean;
  preserve_technical: boolean;
}

export interface CaptureResponse {
  id: string;
  audio_path: string;
  source: CaptureSource;
  language?: string | null;
  duration_ms?: number | null;
  transcript_raw: string;
  transcript_refined?: string | null;
  stt_model?: string | null;
  llm_model?: string | null;
  refinement_flags?: RefinementFlags | null;
  created_at: string;
}

export interface CaptureListResponse {
  items: CaptureResponse[];
  total: number;
}

/**
 * Response of ``POST /captures``. Adds ``auto_refine`` and ``allow_auto_paste``
 * — the server's current settings captured at request time — so the client
 * can decide whether to chain a refine call and whether to fire the
 * synthetic-paste pipeline without relying on its own (possibly stale) copy
 * of capture_settings.
 */
export interface CaptureCreateResponse extends CaptureResponse {
  auto_refine: boolean;
  allow_auto_paste: boolean;
}

export interface CaptureRefineRequest {
  flags?: RefinementFlags;
  model_size?: Qwen3ModelSize;
}

export interface CaptureRetranscribeRequest {
  model?: WhisperModelSize;
  language?: LanguageCode;
}

export interface CaptureSettings {
  stt_model: WhisperModelSize;
  language: string;
  auto_refine: boolean;
  llm_model: Qwen3ModelSize;
  smart_cleanup: boolean;
  self_correction: boolean;
  preserve_technical: boolean;
  allow_auto_paste: boolean;
  default_playback_voice_id: string | null;
  /** Whether the global keyboard hotkey is armed. Off by default — turning
   *  this on triggers the macOS Input Monitoring TCC prompt. */
  hotkey_enabled: boolean;
  /** keytap key names. Defaults are platform-specific right-hand modifiers. */
  chord_push_to_talk_keys: string[];
  /** keytap key names. Toggle adds Space to the platform-specific PTT chord. */
  chord_toggle_to_talk_keys: string[];
}

export type CaptureSettingsUpdate = Partial<CaptureSettings>;

/**
 * One row in the dictation readiness checklist. ``model_name`` is the
 * canonical id understood by ``POST /models/download`` so the UI can wire a
 * one-click "Download" button without a second lookup.
 */
export interface ModelReadiness {
  ready: boolean;
  model_name: string;
  display_name: string;
  size: string;
  size_mb?: number | null;
}

/** Backend half of the dictation readiness check. The frontend combines this
 *  with TCC permission state into the full checklist used by useDictationReadiness. */
export interface CaptureReadinessResponse {
  stt: ModelReadiness;
  llm: ModelReadiness;
}

export interface GenerationSettings {
  max_chunk_chars: number;
  crossfade_ms: number;
  normalize_audio: boolean;
  autoplay_on_generate: boolean;
  /** How strictly VoxCPM2 follows the steering parentheses. */
  vox_cfg: number;
  vox_base_lm?: VoxBaseLm;
  vox_acoustic?: VoxAcoustic;
  /** The audio.cpp engine's backend — separate binaries, a change = a reload. */
  vox_backend?: VoxBackend;
  /** OmniVoice: upstream's ready GGUF — q8 (upstream's default) or f16. */
  omni_weights?: OmniWeights;
  /** OmniVoice: flash attention (perf_mode) — on by default. */
  omni_flash?: boolean;
  /** OmniVoice's own sliders; `null` = upstream's default (2.0 / 32 / random). */
  omni_cfg?: number | null;
  omni_steps?: number | null;
  omni_seed?: number | null;
  /** Pseudo-streaming: none | lines (endline) | sentences (tag_aware). */
  omni_chunk_mode?: 'none' | 'lines' | 'sentences';
  /** Max characters per chunk; `null` = 160 (the streaming example in upstream's docs). */
  omni_chunk_chars?: number | null;
  /** OmniVoice speech speed (`speed`); `null` = upstream's 1.0. */
  omni_speed?: number | null;
  /** Diffusion steps: 30 for reference samples, 10 for chatting, 8 is the floor. */
  vox_timesteps: number;
  /** Noise temperature in the diffusion. 1.0 = the engine's value. */
  vox_temperature: number;
  /** Random seed. `null` = different every time (the intended default). */
  vox_seed: number | null;
  /** How many decoder steps to collect BEFORE the first stream chunk.
   *  A step is ~0.17 s of sound, so this is directly the silence before the first word. */
  vox_stream_min?: number;
  /** Every how many steps the next chunks go out — i.e. how thick the portions are. */
  vox_stream_every?: number;
  /** Whisper (whisper.cpp): the dictation model, quantization, GPU or CPU. */
  whisper_model?: WhisperModel;
  whisper_quant?: WhisperQuant;
  whisper_device?: WhisperDevice;
  /** Whisper loads by itself when its container starts (with the choice above). */
  whisper_on_startup?: boolean;
  /** whisper-server flash attention — on by default. */
  whisper_fa?: boolean;
  /** true = the Offloader is off, models of different engines may share the card. */
  keep_all_models?: boolean;
}

/** The TTS engines of this build. */
export type TtsEngine = 'voxcpm' | 'voxcpm_pt' | 'omnivoice';

export type WhisperModel = 'medium' | 'large' | 'turbo';
export type WhisperQuant = 'q5' | 'q8';
export type WhisperDevice = 'gpu' | 'cpu';

/** The `whisper` container's state. `loaded_*` = what the process REALLY holds, not the settings. */
export interface WhisperState {
  loaded: boolean;
  /** false = the `whisper` container does not answer */
  container: boolean;
  loaded_model: WhisperModel | null;
  loaded_quant: WhisperQuant | null;
  loaded_device: WhisperDevice | null;
  loaded_fa?: boolean | null;
  selected_fa?: boolean;
  selected_model: WhisperModel;
  selected_quant: WhisperQuant;
  selected_device: WhisperDevice;
  files: {
    model: WhisperModel;
    quant: WhisperQuant;
    file: string;
    on_disk: boolean;
    size_mib: number;
  }[];
  ms?: number | null;
}

export type GenerationSettingsUpdate = Partial<GenerationSettings>;

/** VoxCPM2's language part: q8 — 1.6 GB, faster; f16 — 3.0 GB. */
export type VoxBaseLm = 'q8' | 'f16';
/** The rest of the generator (AudioVAE shapes the wave). There is no q8. */
export type VoxAcoustic = 'f16' | 'f32';
/** ROCm (HIP) or Vulkan (RADV) — which audio.cpp binary computes the voice. */
export type VoxBackend = 'hip' | 'vulkan';
/** OmniVoice weights file: q8 (q8_0, 1.3 GB) or f16 (1.6 GB). */
export type OmniWeights = 'q8' | 'f16';

export interface WeightVariant {
  key: string;
  file: string;
  on_disk: boolean;
  size_mib: number;
}

/** The voice engines' state. `loaded_*` = what the engine REALLY holds, `selected_*` =
 *  what the panel picked; they differ until a reload. */
export interface TtsState {
  /** false = no audio.cpp engine configured at all */
  available: boolean;
  vox_loaded: boolean;
  /** How long the last (re)load took — only in reload replies. */
  ms?: number | null;
  loaded_base_lm?: VoxBaseLm | null;
  loaded_acoustic?: VoxAcoustic | null;
  selected_base_lm?: VoxBaseLm | null;
  selected_acoustic?: VoxAcoustic | null;
  loaded_backend?: VoxBackend | null;
  selected_backend?: VoxBackend | null;
  /** OmniVoice lives in the same engine — one model at a time, so either it or Vox. */
  omni_loaded?: boolean;
  loaded_omni?: OmniWeights | null;
  loaded_omni_flash?: boolean | null;
  selected_omni?: OmniWeights | null;
  omni?: WeightVariant[];
  base_lm?: WeightVariant[];
  acoustic?: WeightVariant[];
  /** Stream granularity the engine REALLY started with vs the panel's. */
  loaded_stream_min?: number | null;
  loaded_stream_every?: number | null;
  selected_stream_min?: number | null;
  selected_stream_every?: number | null;
}

export interface TranscriptionRequest {
  language?: LanguageCode;
  model?: WhisperModelSize;
}

export interface TranscriptionResponse {
  text: string;
  duration: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_downloaded?: boolean;
  model_size?: string;
  gpu_available: boolean;
  gpu_type?: string;
  vram_used_mb?: number;
  backend_type?: string;
  backend_variant?: string; // "cpu", "cuda", or "rocm"
  supports_rocm?: boolean; // AMD GPU on Windows or Linux (/dev/kfd) — the ROCm backend is applicable
}

export interface CudaDownloadProgress {
  model_name: string;
  current: number;
  total: number;
  progress: number;
  filename?: string;
  status: 'downloading' | 'extracting' | 'complete' | 'error';
  timestamp: string;
  error?: string;
}

export interface CudaStatus {
  available: boolean; // CUDA binary exists on disk
  active: boolean; // Currently running the CUDA binary
  binary_path: string | null;
  cuda_libs_version: string | null;
  download_supported: boolean; // Platform has a matching release asset
  unsupported_reason: string | null;
  downloading: boolean; // Download in progress
  download_progress?: CudaDownloadProgress;
}

export interface RocmDownloadProgress {
  model_name: string;
  current: number;
  total: number;
  progress: number;
  filename?: string;
  status: 'downloading' | 'extracting' | 'complete' | 'error';
  timestamp: string;
  error?: string;
}

export interface RocmStatus {
  available: boolean; // ROCm binary exists on disk
  active: boolean; // Currently running the ROCm binary
  binary_path?: string;
  rocm_libs_version?: string;
  downloading: boolean; // Download in progress
  download_progress?: RocmDownloadProgress;
}

export interface ModelProgress {
  model_name: string;
  current: number;
  total: number;
  progress: number;
  filename?: string;
  status: 'downloading' | 'extracting' | 'complete' | 'error';
  timestamp: string;
  error?: string;
}

export interface ModelStatus {
  model_name: string;
  display_name: string;
  hf_repo_id?: string; // HuggingFace repository ID
  downloaded: boolean;
  downloading: boolean; // True if download is in progress
  size_mb?: number;
  loaded: boolean;
}

export interface HuggingFaceModelInfo {
  id: string;
  author: string;
  lastModified: string;
  pipeline_tag?: string;
  library_name?: string;
  downloads: number;
  likes: number;
  tags: string[];
  cardData?: {
    license?: string;
    language?: string[];
    pipeline_tag?: string;
  };
}

export interface ModelStatusListResponse {
  models: ModelStatus[];
}

export interface ModelDownloadRequest {
  model_name: string;
}

export interface ActiveDownloadTask {
  model_name: string;
  status: string;
  started_at: string;
  error?: string;
  progress?: number; // 0-100 percentage
  current?: number; // bytes downloaded
  total?: number; // total bytes
  filename?: string; // current file being downloaded
}

export interface ActiveGenerationTask {
  task_id: string;
  profile_id: string;
  text_preview: string;
  started_at: string;
}

export interface ActiveTasksResponse {
  downloads: ActiveDownloadTask[];
  generations: ActiveGenerationTask[];
}

export interface StoryCreate {
  name: string;
  description?: string;
}

export interface StoryResponse {
  id: string;
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
  item_count: number;
}

export interface StoryItemDetail {
  id: string;
  story_id: string;
  generation_id: string;
  version_id?: string;
  start_time_ms: number;
  track: number;
  trim_start_ms: number;
  trim_end_ms: number;
  created_at: string;
  profile_id: string;
  profile_name: string;
  text: string;
  language: string;
  audio_path: string;
  duration: number;
  seed?: number;
  instruct?: string;
  engine?: string;
  volume: number;
  generation_created_at: string;
  versions?: GenerationVersionResponse[];
  active_version_id?: string;
}

export interface StoryItemVolumeUpdate {
  volume: number;
}

export interface StoryItemVersionUpdate {
  version_id: string | null;
}

export interface StoryDetailResponse {
  id: string;
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
  items: StoryItemDetail[];
}

export interface StoryItemCreate {
  generation_id: string;
  start_time_ms?: number;
  track?: number;
}

export interface StoryItemUpdateTime {
  generation_id: string;
  start_time_ms: number;
}

export interface StoryItemBatchUpdate {
  updates: StoryItemUpdateTime[];
}

export interface StoryItemReorder {
  generation_ids: string[];
}

export interface StoryItemMove {
  start_time_ms: number;
  track: number;
}

export interface StoryItemTrim {
  trim_start_ms: number;
  trim_end_ms: number;
}

export interface StoryItemSplit {
  split_time_ms: number;
}

// Effects

export interface EffectPresetResponse {
  id: string;
  name: string;
  description?: string;
  effects_chain: EffectConfig[];
  is_builtin: boolean;
  created_at: string;
}

export interface EffectPresetCreate {
  name: string;
  description?: string;
  effects_chain: EffectConfig[];
}

export interface EffectPresetUpdate {
  name?: string;
  description?: string;
  effects_chain?: EffectConfig[];
}

export interface AvailableEffectParam {
  default: number;
  min: number;
  max: number;
  step: number;
  description: string;
}

export interface AvailableEffect {
  type: string;
  label: string;
  description: string;
  params: Record<string, AvailableEffectParam>;
}

export interface AvailableEffectsResponse {
  effects: AvailableEffect[];
}

export interface ApplyEffectsRequest {
  effects_chain: EffectConfig[];
  source_version_id?: string;
  label?: string;
  set_as_default?: boolean;
}

/* ─── MCP ─────────────────────────────────────────────────────────────── */

export interface MCPClientBinding {
  client_id: string;
  label: string | null;
  profile_id: string | null;
  default_engine: string | null;
  default_personality: boolean;
  last_seen_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface MCPClientBindingUpsert {
  client_id: string;
  label?: string | null;
  profile_id?: string | null;
  default_engine?: string | null;
  default_personality?: boolean;
}

export interface MCPClientBindingListResponse {
  items: MCPClientBinding[];
}

/* ─── Cloud (backup & sync) ───────────────────────────────────────────── */

export interface CloudLoginStartResponse {
  authorize_url: string;
}

export interface CloudStatus {
  connected: boolean;
  device_name: string | null;
  account_user_id: string | null;
  key_prefix: string | null;
  connected_at: string | null;
  dashboard_url: string;
}

// ── the GPU for the models ────────────────────────────────────────────────────
// The choice is stored as a PCI address and takes effect with the next model load
// (the engine containers start each model with the visibility variables).
export interface GpuCard {
  index: number;
  pci: string;
  name: string;
  vram_mib: number;
}

export interface GpuState {
  cards: GpuCard[];
  /** What the file holds (a PCI address); null = every card visible. */
  selected_pci: string | null;
  /** The card the engines get on THIS machine — after a card is pulled or a move it may differ. */
  resolved: number | null;
  /** Where this choice comes from, in plain words. */
  reason: string;
}
