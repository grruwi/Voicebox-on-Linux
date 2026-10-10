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
  engine?:
    | 'qwen'
    | 'qwen_custom_voice'
    | 'voxcpm'
    | 'omnivoice'
    | 'voxcpm_pt'
    | 'fish_s2_pt'
    | 'fish_s2'
    | 'fish_s1'
    | 'luxtts'
    | 'chatterbox'
    | 'chatterbox_turbo'
    | 'tada'
    | 'kokoro'
    | 'supertonic';
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
  /** Jak sztywno VoxCPM2 trzyma się nawiasu sterującego. */
  vox_cfg: number;
  vox_base_lm?: VoxWariant;
  vox_akustyka?: VoxAkustyka;
  /** Backend silnika audio.cpp — osobne binarki, zmiana = przeładowanie. */
  vox_backend?: VoxBackend;
  /** OmniVoice: gotowy GGUF z upstreamu — q8 (domyślny upstreamu) albo f16. */
  omni_wagi?: OmniWagi;
  /** OmniVoice: flash attention (perf_mode) — domyślnie włączony. */
  omni_flash?: boolean;
  /** Własne suwaki OmniVoice; `null` = default upstreamu (2,0 / 32 / losowo). */
  omni_cfg?: number | null;
  omni_kroki?: number | null;
  omni_seed?: number | null;
  /** Pseudo-strumień: brak | linie (endline) | zdania (tag_aware). */
  omni_chunk_tryb?: 'brak' | 'linie' | 'zdania';
  /** Maks. znaków w kawałku; `null` = 160 (przykład strumienia z docs upstreamu). */
  omni_chunk_znaki?: number | null;
  /** Tempo mowy OmniVoice (`speed`); `null` = 1,0 upstreamu. */
  omni_tempo?: number | null;
  /** Kroki dyfuzji: 30 do próbek referencyjnych, 10 do gadania, 8 to próg. */
  vox_timesteps: number;
  /** Temperatura szumu w dyfuzji. 1.0 = wartość silnika. */
  vox_temperature: number;
  /** Ziarno losowe. `null` = za każdym razem inaczej (tak ma być domyślnie). */
  vox_seed: number | null;
  /** Ile kroków dekodera uzbierać PRZED pierwszym kawałkiem strumienia.
   *  Krok to ~0,17 s dźwięku, więc to wprost cisza przed pierwszym słowem. */
  vox_stream_min?: number;
  /** Co ile kroków lecą kolejne kawałki — czyli jak grube są porcje. */
  vox_stream_co?: number;
  /** Whisper (whisper.cpp): model dyktafonu, kwantyzacja, karta albo procesor. */
  whisper_model?: WhisperModel;
  whisper_kwant?: WhisperKwant;
  whisper_urzadzenie?: WhisperUrzadzenie;
  /** Backend ładuje whispera sam przy starcie (w tle, z wyborem wyżej). */
  whisper_przy_starcie?: boolean;
  /** Flash attention whisper-servera — domyślnie włączony. */
  whisper_fa?: boolean;
}

export type WhisperModel = 'medium' | 'large' | 'turbo';
export type WhisperKwant = 'q5' | 'q8';
export type WhisperUrzadzenie = 'gpu' | 'cpu';

/** Stan procesu `whisper-server`. `zaladowany_*` czytane z /proc, nie z ustawień. */
export interface WhisperStan {
  chodzi: boolean;
  zaladowany_model: WhisperModel | null;
  zaladowany_kwant: WhisperKwant | null;
  zaladowane_urzadzenie: WhisperUrzadzenie | null;
  zaladowany_fa?: boolean | null;
  wybrany_fa?: boolean;
  wybrany_model: WhisperModel;
  wybrany_kwant: WhisperKwant;
  wybrane_urzadzenie: WhisperUrzadzenie;
  pliki: {
    model: WhisperModel;
    kwant: WhisperKwant;
    plik: string;
    jest_na_dysku: boolean;
    rozmiar_mib: number;
  }[];
  ms?: number | null;
}

export type GenerationSettingsUpdate = Partial<GenerationSettings>;

/** Część językowa VoxCPM2: q8 — 1,6 GB, szybszy; f16 — 3,0 GB; f32 — 6,0 GB. */
export type VoxWariant = 'q8' | 'f16' | 'f32';
/** AudioVAE, czyli to, co lepi falę. Wariantu q8 nie ma. */
export type VoxAkustyka = 'f16' | 'f32';
/** ROCm (HIP) albo Vulkan (RADV) — która binarka audio.cpp liczy głos. */
export type VoxBackend = 'hip' | 'vulkan';
/** OmniVoice — plik wag: q8 (q8_0, 1,3 GB) albo f16 (1,6 GB). */
export type OmniWagi = 'q8' | 'f16';

export interface WariantWag {
  klucz: string;
  plik: string;
  jest_na_dysku: boolean;
  rozmiar_mib: number;
}

/** Stan procesu `llama-tts-server`. `zaladowany_*` to wagi, które FAKTYCZNIE
 *  trzyma — czytane z /proc, nie z ustawień. */
export interface TtsStan {
  dostepny: boolean;
  chodzi: boolean;
  wariant: VoxWariant | null;
  ms?: number | null;
  zaladowany_base_lm?: VoxWariant | null;
  zaladowany_akustyka?: VoxAkustyka | null;
  wybrany_base_lm?: VoxWariant | null;
  wybrany_akustyka?: VoxAkustyka | null;
  zaladowany_backend?: VoxBackend | null;
  wybrany_backend?: VoxBackend | null;
  /** OmniVoice siedzi w tym samym silniku — jeden model naraz, więc albo on, albo vox. */
  omni_chodzi?: boolean;
  zaladowany_omni?: OmniWagi | null;
  zaladowany_omni_flash?: boolean | null;
  wybrany_omni?: OmniWagi | null;
  omni?: WariantWag[];
  base_lm?: WariantWag[];
  akustyka?: WariantWag[];
  /** Ziarnistość, z którą serwer FAKTYCZNIE wstał — z /proc/<pid>/environ,
   *  bo te dwie liczby idą środowiskiem, nie argumentem wiersza poleceń. */
  zaladowany_stream_min?: number | null;
  zaladowany_stream_co?: number | null;
  wybrany_stream_min?: number | null;
  wybrany_stream_co?: number | null;
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

// ── wybor karty graficznej dla modeli ───────────────────────────────────────
// `wybrana` to zapis w pliku, `aktywna` to co widzi ZYWY proces backendu.
// Rozjazd = ktos kliknal i nie zrestartowal jeszcze Voiceboxa.
export interface KartaGpu {
  indeks: number;
  pci: string;
  nazwa: string;
  vram_mib: number;
}

export interface StanKarty {
  karty: KartaGpu[];
  wybrana: number | null;
  aktywna: number | null;
  /** Karta, którą silniki dostaną po restarcie — po wyjęciu karty lub przenosinach ≠ `wybrana`. */
  rozwiazana: number | null;
  /** Skąd ten wybór, po ludzku. */
  powod: string;
  wymaga_restartu: boolean;
}
